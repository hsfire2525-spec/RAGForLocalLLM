"""人手抽出検査。

外部judgeが使えない以上、自由記述の妥当性を確かめる手段は人手しかない。
**それを軽量に回せることが決定的に重要**（docs/design/design.md §6.4）。

そのための仕掛けが2つある。

1. **過去判定の再利用。** 判定は「質問 × 回答文字列」に紐づく。
   構成を変えても同じ回答が出れば再判定は不要で、これがないと
   人手コストが実験回数に比例して増える
2. **層別サンプリング。** 全件を見ないので、質問タイプが偏ると
   「表参照は全滅していたが、たまたま抽出されなかった」が起きる

判定は4値。自動採点と同じ区分にしてあるので、**自動指標と人手の
一致率**をそのまま計算できる。乖離が大きい質問タイプは、データセット側を
機械採点しやすい形に直す判断材料になる。
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ragforlocalllm.eval.normalize import normalize_answer

Verdict = Literal["correct", "incorrect", "valid_abstention", "invalid_abstention"]

VERDICTS: tuple[Verdict, ...] = (
    "correct",
    "incorrect",
    "valid_abstention",
    "invalid_abstention",
)

VERDICT_LABELS: dict[Verdict, str] = {
    "correct": "正答",
    "incorrect": "誤答",
    "valid_abstention": "妥当な棄権",
    "invalid_abstention": "不当な棄権",
}

# 自動採点の4値と人手判定の4値の対応。一致率の計算に使う。
AUTO_TO_VERDICT: dict[str, Verdict] = {
    "correct": "correct",
    "incorrect": "incorrect",
    "correct_abstention": "valid_abstention",
    "unjustified_abstention": "invalid_abstention",
}

DEFAULT_STORE = Path("data/reviews/human_judgments.jsonl")


def answer_key(qid: str, answer_text: str | None) -> str:
    """判定を再利用する単位。質問 × 正規化した回答文字列。"""
    return f"{qid}\t{normalize_answer(answer_text or '')}"


@dataclass(frozen=True)
class Judgment:
    qid: str
    answer: str
    verdict: Verdict
    comment: str = ""
    reviewed_at: str = ""
    run: str = ""

    @property
    def key(self) -> str:
        return answer_key(self.qid, self.answer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "answer": self.answer,
            "verdict": self.verdict,
            "comment": self.comment,
            "reviewed_at": self.reviewed_at,
            "run": self.run,
        }


class JudgmentStore:
    """人手判定の永続化。ラン間で共有する。

    **ランレコードとは別に持つ。** ランごとの ``human_review.jsonl`` に
    しか残さないと、次のランで同じ回答をもう一度判定させることになる。
    """

    def __init__(self, path: Path | str = DEFAULT_STORE) -> None:
        self.path = Path(path)
        self._by_key: dict[str, Judgment] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                judgment = Judgment(
                    qid=payload["qid"],
                    answer=payload.get("answer", ""),
                    verdict=payload["verdict"],
                    comment=payload.get("comment", ""),
                    reviewed_at=payload.get("reviewed_at", ""),
                    run=payload.get("run", ""),
                )
                self._by_key[judgment.key] = judgment

    def __len__(self) -> int:
        return len(self._by_key)

    def get(self, qid: str, answer_text: str | None) -> Judgment | None:
        return self._by_key.get(answer_key(qid, answer_text))

    def put(self, judgment: Judgment) -> None:
        self._by_key[judgment.key] = judgment
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(judgment.as_dict(), ensure_ascii=False) + "\n")

    def record(
        self,
        qid: str,
        answer_text: str | None,
        verdict: Verdict,
        *,
        comment: str = "",
        run: str = "",
    ) -> Judgment:
        judgment = Judgment(
            qid=qid,
            answer=answer_text or "",
            verdict=verdict,
            comment=comment,
            reviewed_at=datetime.now().isoformat(timespec="seconds"),
            run=run,
        )
        self.put(judgment)
        return judgment


def stratified_sample(
    rows: Sequence[dict[str, Any]],
    n: int,
    *,
    stratify_by: str | None = "question_type",
    seed: int = 20260731,
) -> list[dict[str, Any]]:
    """層をラウンドロビンで回して n 件選ぶ。

    単純無作為だと少数の質問タイプが丸ごと抜け落ちる。層別の目的は
    「見ていない領域を作らない」ことなので、層内はシャッフルし、
    層はサイズの大きい順ではなく巡回で消費する。
    """
    if n <= 0 or not rows:
        return []
    rng = random.Random(seed)

    if stratify_by is None:
        pool = list(rows)
        rng.shuffle(pool)
        return pool[:n]

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(stratify_by, "unknown")), []).append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    names = sorted(groups)
    while len(selected) < n and any(groups[name] for name in names):
        for name in names:
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop())
    return selected


def agreement(rows: Sequence[dict[str, Any]], store: JudgmentStore) -> dict[str, Any]:
    """自動採点と人手判定の一致率。

    **自動指標が信頼できる範囲を把握するための指標。** 一致率が低い
    質問タイプは、データセットを機械採点しやすい形に直すか、
    その層だけ人手検査の比率を上げる判断になる。
    """
    matched = 0
    total = 0
    disagreements: list[dict[str, Any]] = []
    by_type: dict[str, list[int]] = {}

    for row in rows:
        judgment = store.get(row["qid"], row.get("answer"))
        if judgment is None:
            continue
        total += 1
        auto = AUTO_TO_VERDICT.get(str(row.get("outcome")), "incorrect")
        agreed = auto == judgment.verdict
        matched += int(agreed)
        by_type.setdefault(str(row.get("question_type", "other")), []).append(int(agreed))
        if not agreed:
            disagreements.append(
                {
                    "qid": row["qid"],
                    "auto": auto,
                    "human": judgment.verdict,
                    "answer": row.get("answer"),
                    "comment": judgment.comment,
                }
            )

    return {
        "n_reviewed": total,
        "agreement_rate": round(matched / total, 4) if total else None,
        "by_question_type": {
            name: {"n": len(flags), "agreement": round(sum(flags) / len(flags), 4)}
            for name, flags in sorted(by_type.items())
        },
        "disagreements": disagreements,
    }

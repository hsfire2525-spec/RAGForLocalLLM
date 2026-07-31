"""gold QA の起草支援と、引用文の検証。

作成手順は「候補の機械生成 → 人手での確定 → 機械での検証 → 凍結」
（docs/design/design.md §10.4）。このモジュールは1番目と3番目を担う。
2番目（ドメイン判断）は人がやるしかない。

**3番目が特に重要。** 手打ちした引用文が抽出テキストと一致しないのが
最大の失敗要因で、しかもこれは静かに検索メトリクスを壊す。
gold を凍結する前に、全 evidence が解決することを機械で確かめる。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ragforlocalllm.core.types import Chunk
from ragforlocalllm.eval.dataset import GoldDataset
from ragforlocalllm.eval.resolve import Resolver

_SENTENCE = re.compile(r"[^。！？\n]+[。！？]?")

MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 40


@dataclass(frozen=True)
class DraftCandidate:
    """人が質問と回答を書き込むための下書き1件。"""

    qid: str
    chunk: Chunk
    quote: str
    passage: str

    def as_gold_row(self) -> dict[str, Any]:
        """``GoldItem`` として妥当な JSON。質問と回答だけが未記入。

        ``TODO`` を残しておくと ``rag gold`` が未完成として警告する。
        空文字にすると検証を通ってしまい、未完成のまま実験に使われうる。
        """
        evidence: dict[str, Any] = {"page": self.chunk.page, "quote": self.quote}
        if self.chunk.section_path:
            evidence["section_path"] = self.chunk.section_path
        return {
            "qid": self.qid,
            "question": "TODO",
            "answer": "TODO",
            "answer_type": "short",
            "question_type": "other",
            "evidence": [evidence],
            "answerable": True,
            "notes": self.passage,
        }


def draft_candidates(
    chunks: Sequence[Chunk], n: int, *, seed: int = 20260731, passage_chars: int = 300
) -> list[DraftCandidate]:
    """チャンクを層別にサンプリングして下書き候補を作る。

    層は節の第1階層（無ければページ帯）。**文書の特定部分に偏った
    gold は、その部分に有利な構成を選んでしまう。**
    """
    import random

    rng = random.Random(seed)
    groups: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(_stratum(chunk), []).append(chunk)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[Chunk] = []
    names = sorted(groups)
    while len(selected) < n and any(groups[name] for name in names):
        for name in names:
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop())

    candidates: list[DraftCandidate] = []
    for i, chunk in enumerate(selected, start=1):
        quote = pick_quote(chunk.text)
        if quote is None:
            continue
        candidates.append(
            DraftCandidate(
                qid=f"q{i:03d}",
                chunk=chunk,
                quote=quote,
                passage=chunk.text[:passage_chars].replace("\n", " "),
            )
        )
    return candidates


def _stratum(chunk: Chunk) -> str:
    if chunk.section_path:
        return chunk.section_path.split(" > ")[0]
    page = chunk.page
    return f"p{page // 10:02d}x" if page is not None else "unknown"


def pick_quote(text: str) -> str | None:
    """識別に足りる長さの、なるべく短い引用文を選ぶ。

    引用文は**識別に必要な最小限**に留める（コーパス本体はコミット
    できない前提を尊重する）。長すぎる引用はチャンク境界を跨ぎやすく、
    解決に失敗する確率も上がる。
    """
    # 1行目は見出しであることが多く、他の節と重複しやすい
    body = "\n".join(text.splitlines()[1:]) or text
    for match in _SENTENCE.finditer(body):
        sentence = match.group().strip()
        if len(sentence) < MIN_QUOTE_CHARS:
            continue
        return sentence[:MAX_QUOTE_CHARS]
    stripped = body.strip()
    return stripped[:MAX_QUOTE_CHARS] if len(stripped) >= MIN_QUOTE_CHARS else None


# ----------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationIssue:
    qid: str
    quote: str
    status: str
    hint: str


def verify_quotes(gold: GoldDataset, chunks: Sequence[Chunk]) -> list[VerificationIssue]:
    """全 evidence の引用文が実際の抽出テキストに解決するか確かめる。

    **凍結前に必ず通す。** ここで落とさないと、解決できない引用が
    検索メトリクスの欠測として紛れ込む。
    """
    resolver = Resolver(chunks, fallback_to_page=False)
    issues: list[VerificationIssue] = []
    for item in gold:
        if not item.answerable:
            continue
        resolution = resolver.resolve_item(item)
        for match in resolution.matches:
            if match.resolved:
                continue
            quote = match.evidence.quote or ""
            if match.quote_status == "split_across_chunks":
                hint = "チャンク境界で分断。引用文を短くするかオーバーラップを増やす。"
            elif match.quote_status == "missing_from_corpus":
                hint = "抽出テキストに存在しません。転記ミスか Loader の欠落です。"
            else:
                hint = "page / section_path が抽出結果と一致しません。"
            issues.append(
                VerificationIssue(
                    qid=item.qid,
                    quote=quote,
                    status=match.quote_status or "unresolved",
                    hint=hint,
                )
            )
    return issues


def todo_qids(gold: GoldDataset) -> list[str]:
    """質問または回答が未記入のまま残っている項目。"""
    return [
        item.qid
        for item in gold
        if "TODO" in item.question or "TODO" in item.answer or not item.question.strip()
    ]

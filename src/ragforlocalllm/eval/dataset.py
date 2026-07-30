"""評価データセット（gold QA）のスキーマ。

**根拠は chunk_id ではなく「ページ + 引用文」でアンカーする。**
chunk_id は Chunker の設定に依存するため、チャンク戦略を変えるたびに
gold が無効になり、本リポジトリの主目的（チャンク戦略の比較）と
両立しない。引用文は評価時にチャンク集合へ解決する
（docs/design/design.md §6.2、解決器は Phase 1）。

引用文は**識別に必要な最小限の長さ**に留める。コーパス本体は
リポジトリにコミットしない前提のため。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal[
    "definition",  # 用語定義
    "numeric",  # 数値・期限・件数
    "enumeration",  # 列挙
    "procedure",  # 手順
    "responsibility",  # 責任・体制
    "reference",  # 参照解決
    "table",  # 表の参照
    "unanswerable",  # コーパスに答えが無い
    "other",
]

AnswerType = Literal["short", "numeric", "list", "long"]
"""機械採点の方式を決める。

外部judgeが使えないため、データセットは機械採点可能な形
（short / numeric / list）を中心に構成する。``long`` は少数に留め、
人手抽出検査で扱う（docs/design/design.md §6.1）。
"""


class Evidence(BaseModel):
    """回答の根拠となる箇所。

    ``quote`` は原文からの短い抜粋。評価時にチャンク本文へ
    部分文字列一致（NFKC正規化 + 空白除去）で解決する。
    """

    model_config = ConfigDict(extra="forbid")

    page: int | None = None
    quote: str | None = None
    section_path: str | None = None

    @model_validator(mode="after")
    def _require_anchor(self) -> Evidence:
        if self.quote is None and self.page is None and self.section_path is None:
            raise ValueError("Evidence には quote / page / section_path のいずれかが必要です")
        return self


class GoldItem(BaseModel):
    """gold QA 1件。"""

    model_config = ConfigDict(extra="forbid")

    qid: str
    question: str
    answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    """表記ゆれ・別名。exact match とエイリアス照合に使う。"""
    answer_type: AnswerType = "short"
    evidence: list[Evidence] = Field(default_factory=list)
    answerable: bool = True
    """False の質問を10〜20%含める。棄権性能を測れないと、
    「何でも答える」構成が高スコアになってしまう。"""
    question_type: QuestionType = "other"
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> GoldItem:
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.qid}: answerable=true には evidence が必要です")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.qid}: answerable=false に evidence は指定できません")
        if self.answer_type == "list" and not self.answer.strip():
            raise ValueError(f"{self.qid}: answer_type=list には answer が必要です")
        return self

    @property
    def accepted_answers(self) -> list[str]:
        return [self.answer, *self.answer_aliases]

    @property
    def answer_items(self) -> list[str]:
        """列挙型の回答を要素に分解する（集合F1用）。"""
        if self.answer_type != "list":
            return [self.answer]
        return [part.strip() for part in self.answer.replace("、", ",").split(",") if part.strip()]


class GoldDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[GoldItem]
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[GoldItem]:  # type: ignore[override]
        return iter(self.items)

    @property
    def answerable_count(self) -> int:
        return sum(1 for item in self.items if item.answerable)

    @property
    def unanswerable_ratio(self) -> float:
        if not self.items:
            return 0.0
        return 1.0 - self.answerable_count / len(self.items)

    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.question_type] = counts.get(item.question_type, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, object]:
        return {
            "n_items": len(self.items),
            "answerable": self.answerable_count,
            "unanswerable": len(self.items) - self.answerable_count,
            "unanswerable_ratio": round(self.unanswerable_ratio, 3),
            "by_question_type": self.by_type(),
        }


def load_gold(path: str | Path) -> GoldDataset:
    """JSONL 形式の gold データセットを読む。

    1行1件。空行と ``#`` 始まりのコメント行は無視する。
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"gold データセットがありません: {source}")

    items: list[GoldItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{lineno}: JSON として読めません: {exc}") from exc
        try:
            item = GoldItem.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"{source}:{lineno}: 検証に失敗しました\n{exc}") from exc
        if item.qid in seen:
            raise ValueError(f"{source}:{lineno}: qid が重複しています: {item.qid}")
        seen.add(item.qid)
        items.append(item)

    return GoldDataset(items=items, source=source)

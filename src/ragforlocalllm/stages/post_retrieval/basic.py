"""検索結果の後処理（LLM非依存の低コスト手法）。

この段が最重要（docs/design/design.md §3.2(7)）。4B級モデルの回答
精度は「渡されたコンテキストの質」でほぼ決まるため、リランクや
圧縮はここに集約する。ここではリランカー（追加依存が必要）の前に
入れられる、依存なしの手法を実装する。
"""

from __future__ import annotations

import unicodedata

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import QueryState, ScoredChunk


@register("post_retrieval", "top_k")
class TopK:
    """上位k件に切る。リランク後の絞り込みに使う。"""

    def __init__(self, k: int = 5) -> None:
        if k <= 0:
            raise ValueError("k は正の整数である必要があります")
        self.k = k

    def process(self, state: QueryState) -> QueryState:
        state.contexts = state.contexts[: self.k]
        return state


@register("post_retrieval", "reorder_lost_in_middle")
class ReorderLostInMiddle:
    """重要度の高い文書を先頭と末尾に配置する。

    系列中央の情報が使われにくくなる現象（lost-in-the-middle）は
    小さいモデルでより顕著に出る。並べ替えだけなのでコストはゼロ。
    """

    def process(self, state: QueryState) -> QueryState:
        ranked = sorted(state.contexts, key=lambda c: c.score, reverse=True)
        head: list[ScoredChunk] = []
        tail: list[ScoredChunk] = []
        for i, item in enumerate(ranked):
            (head if i % 2 == 0 else tail).append(item)
        state.contexts = head + list(reversed(tail))
        return state


@register("post_retrieval", "dedupe")
class Dedupe:
    """近似重複の除去。

    正規化後の文字bi-gram Jaccard 係数が閾値を超えるものを重複とみなし、
    スコアの高い側を残す。コンテキスト長を節約でき、4B級に効く。
    """

    def __init__(self, threshold: float = 0.9, ngram: int = 2) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold は 0 より大きく 1 以下である必要があります")
        self.threshold = threshold
        self.ngram = ngram

    def process(self, state: QueryState) -> QueryState:
        kept: list[ScoredChunk] = []
        signatures: list[set[str]] = []
        for item in sorted(state.contexts, key=lambda c: c.score, reverse=True):
            signature = self._ngrams(item.chunk.text)
            if any(self._jaccard(signature, seen) >= self.threshold for seen in signatures):
                continue
            kept.append(item)
            signatures.append(signature)
        state.contexts = kept
        return state

    def _ngrams(self, text: str) -> set[str]:
        cleaned = "".join(unicodedata.normalize("NFKC", text).split())
        if len(cleaned) < self.ngram:
            return {cleaned} if cleaned else set()
        return {cleaned[i : i + self.ngram] for i in range(len(cleaned) - self.ngram + 1)}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


@register("post_retrieval", "add_section_path")
class AddSectionPath:
    """チャンク本文の先頭に ``section_path`` を付与する。

    チャンク単体では何の話か分からない箇所を低コストで補う。
    元のチャンクは書き換えず、複製に対して行う。
    """

    def __init__(self, template: str = "【{section_path}】\n") -> None:
        self.template = template

    def process(self, state: QueryState) -> QueryState:
        updated: list[ScoredChunk] = []
        for item in state.contexts:
            section_path = item.chunk.section_path
            if not section_path:
                updated.append(item)
                continue
            chunk = item.chunk.model_copy(
                update={"text": self.template.format(section_path=section_path) + item.chunk.text}
            )
            updated.append(
                ScoredChunk(chunk=chunk, score=item.score, provenance=dict(item.provenance))
            )
        state.contexts = updated
        return state

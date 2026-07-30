"""クエリの前処理（LLM非依存）。

LLM依存の手法（HyDE、multi-query 等）は Phase 3 で追加し、
ここでは「低コストで効く」ルールベースの候補を置く
（docs/design/design.md §3.2(5)）。
"""

from __future__ import annotations

import unicodedata

from ragforlocalllm.core.registry import register


@register("query_transform", "identity")
class IdentityTransform:
    """素通し。ベースライン。"""

    def transform(self, query: str) -> list[str]:
        return [query]


@register("query_transform", "normalize")
class NormalizeTransform:
    """NFKC正規化と空白整理のみを行う。

    全角英数・半角カナの混在を吸収する。日本語コーパスでは
    表記ゆれの影響が大きいため、低コストな第一候補。
    """

    def __init__(self, nfkc: bool = True, strip_spaces: bool = True) -> None:
        self.nfkc = nfkc
        self.strip_spaces = strip_spaces

    def transform(self, query: str) -> list[str]:
        text = query
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.strip_spaces:
            text = " ".join(text.split())
        return [text]


@register("query_transform", "synonym_expand")
class SynonymExpandTransform:
    """用語辞書によるクエリ拡張。

    元クエリに加えて、辞書で置換した派生クエリを返す。
    セキュリティ用語の同義語（標的型攻撃 / スピアフィッシング 等）を
    手作業辞書で展開する用途。辞書は設定に直接書ける。
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        nfkc: bool = True,
        max_queries: int = 4,
    ) -> None:
        self.synonyms = synonyms or {}
        self.nfkc = nfkc
        self.max_queries = max_queries

    def transform(self, query: str) -> list[str]:
        text = unicodedata.normalize("NFKC", query) if self.nfkc else query
        out = [text]
        for term, alternatives in self.synonyms.items():
            if term not in text:
                continue
            for alt in alternatives:
                variant = text.replace(term, alt)
                if variant not in out:
                    out.append(variant)
                if len(out) >= self.max_queries:
                    return out
        return out

"""生成後の機械的検証（LLM非依存）。

引用整合性チェックと棄権判定は、LLMの能力とは独立に効き、
低コストでハルシネーションを検出できる（docs/design/design.md §3.2(10)）。
NLI による根拠検証は Phase 3（追加依存が必要）。
"""

from __future__ import annotations

import re

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import QueryState

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
DEFAULT_ABSTAIN_PHRASES = ("分かりません", "わかりません", "不明です", "記載がありません")


@register("post_generation", "parse_citations")
class ParseCitations:
    """回答本文の ``[n]`` を実際のチャンクIDへ解決する。

    存在しない番号を引用していた場合は ``invalid_citations`` として
    trace に残す。これはハルシネーションの直接的な兆候になる。
    """

    def __init__(self, strip_from_text: bool = False) -> None:
        self.strip_from_text = strip_from_text

    def process(self, state: QueryState) -> QueryState:
        answer, prompt = state.answer, state.prompt
        if answer is None or prompt is None:
            return state

        available = prompt.context_chunk_ids
        resolved: list[str] = []
        invalid: list[str] = []
        for marker in CITATION_PATTERN.findall(answer.text):
            index = int(marker) - 1  # プロンプト上の番号は 1 始まり
            if 0 <= index < len(available):
                chunk_id = available[index]
                if chunk_id not in resolved:
                    resolved.append(chunk_id)
            else:
                invalid.append(marker)

        answer.citations = resolved
        if self.strip_from_text:
            answer.text = CITATION_PATTERN.sub("", answer.text).strip()

        if invalid:
            state.trace[-1].info["invalid_citations"] = invalid if state.trace else None
        return state


@register("post_generation", "abstain_on_phrase")
class AbstainOnPhrase:
    """定型の棄権表現を検出して ``abstained`` を立てる。

    正答率・誤答率・棄権率の3値を算出するために必要。単一の精度
    指標だけを見ると、棄権を捨てて誤答を増やす構成が勝ってしまう。
    """

    def __init__(self, phrases: tuple[str, ...] | list[str] = DEFAULT_ABSTAIN_PHRASES) -> None:
        self.phrases = tuple(phrases)

    def process(self, state: QueryState) -> QueryState:
        answer = state.answer
        if answer is None:
            return state
        stripped = CITATION_PATTERN.sub("", answer.text).strip()
        if any(phrase in stripped for phrase in self.phrases):
            answer.abstained = True
        return state


@register("post_generation", "abstain_without_citation")
class AbstainWithoutCitation:
    """有効な引用が無い回答を棄権に置き換える。

    根拠を示せていない回答は誤答である可能性が高い。実用上の
    誤答率を下げる、最も安価な手段のひとつ。
    """

    def __init__(self, message: str = "分かりません") -> None:
        self.message = message

    def process(self, state: QueryState) -> QueryState:
        answer = state.answer
        if answer is None or answer.abstained:
            return state
        if not answer.citations:
            answer.raw_text = answer.raw_text or answer.text
            answer.text = self.message
            answer.abstained = True
        return state

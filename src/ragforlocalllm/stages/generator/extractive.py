"""LLMを使わない抽出型ジェネレータ。

「LLMの賢さに依存しない工夫を優先する」（原則5）を評価するうえで、
**LLMを一切使わない下限のベースライン**は重要な参照点になる。
検索とリランクだけでどこまで到達できるかが分かり、LLM投入による
改善分を切り分けられる。CI でも動く（外部プロセス不要）。

このベースラインは**棄権できることが必須**。棄権しないベースラインは
誤答率が常に最大になり、比較対象として機能しない。
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Answer, Prompt, TokenUsage

# 文境界は句点等に加えて改行でも切る。日本語の箇条書き・見出し・表の行は
# 句点で終わらないため、句点のみで分割すると見出しと本文が同一単位になり、
# 回答に構造的なノイズが混入する。
_UNIT_SPLIT = re.compile(r"(?<=[。！？!?])|\n+")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s")
_LIST_MARKER = re.compile(r"^\s*(?:[-*・]|\d+[.)])\s*")


@register("generator", "extractive")
class ExtractiveGenerator:
    """コンテキストから、質問と最も語が重なる文を抜き出して返す。

    スコアは文字bi-gramの Dice 係数（形態素解析器に依存させないため）。
    ``min_score`` を下回る場合は棄権する。
    """

    def __init__(
        self,
        max_sentences: int = 2,
        min_score: float = 0.15,
        skip_headings: bool = True,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score は 0.0〜1.0 である必要があります")
        self.max_sentences = max_sentences
        self.min_score = min_score
        self.skip_headings = skip_headings

    def generate(self, prompt: Prompt, schema: dict[str, Any] | None = None) -> Answer:
        started = time.perf_counter()
        question = _extract_question(prompt)
        question_grams = _bigrams(question)

        scored: list[tuple[float, str, str]] = []  # (score, unit, marker)
        for marker, text in _extract_contexts(prompt):
            for unit in _units(text, skip_headings=self.skip_headings):
                score = _dice(question_grams, _bigrams(unit))
                if score >= self.min_score:
                    scored.append((score, unit, marker))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[: self.max_sentences]
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        if not picked:
            # 根拠となる文が見つからなければ棄権する。
            return Answer(
                text="分かりません",
                raw_text="",
                abstained=True,
                model="extractive",
                usage=TokenUsage(prompt_tokens=prompt.token_estimate, completion_tokens=None),
                latency_ms=latency_ms,
            )

        body = " ".join(unit for _, unit, _ in picked)
        markers = sorted({marker for _, _, marker in picked}, key=int)
        return Answer(
            text=body + " " + " ".join(f"[{m}]" for m in markers),
            raw_text=body,
            citations=markers,
            abstained=False,
            model="extractive",
            usage=TokenUsage(prompt_tokens=prompt.token_estimate, completion_tokens=None),
            latency_ms=latency_ms,
        )


# ----------------------------------------------------------------------
# プロンプトの解析
# ----------------------------------------------------------------------


def _extract_question(prompt: Prompt) -> str:
    user = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
    if "# 質問" in user:
        return user.split("# 質問", 1)[1].strip()
    return user.strip()


def _extract_contexts(prompt: Prompt) -> list[tuple[str, str]]:
    """プロンプト本文から ``[n]`` 見出し単位のコンテキストを取り出す。"""
    user = next((m.content for m in reversed(prompt.messages) if m.role == "user"), "")
    body = user.split("# 質問", 1)[0]
    parts = re.split(r"^\[(\d+)\]", body, flags=re.MULTILINE)
    out: list[tuple[str, str]] = []
    # parts = ["前置き", "1", "本文1", "2", "本文2", ...]
    for i in range(1, len(parts) - 1, 2):
        out.append((parts[i], parts[i + 1].strip()))
    return out


def _units(text: str, *, skip_headings: bool) -> list[str]:
    """回答候補となる文の単位に分割する。

    Markdown 見出しは構造情報であり回答本体ではないため、既定で除外する
    （質問と語が重なりやすく、除外しないと回答が見出しで埋まる）。
    """
    units: list[str] = []
    for raw in _UNIT_SPLIT.split(text):
        if raw is None:
            continue
        unit = raw.strip()
        if not unit:
            continue
        if skip_headings and _MD_HEADING.match(unit):
            continue
        units.append(_LIST_MARKER.sub("", unit))
    return units


# ----------------------------------------------------------------------
# スコア
# ----------------------------------------------------------------------


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(unicodedata.normalize("NFKC", text).split())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _dice(a: set[str], b: set[str]) -> float:
    """Dice 係数。長い文が有利になりすぎないよう正規化する。"""
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))

"""トークン数の算定。

LM Studio の OpenAI 互換APIにトークン化エンドポイントはないため、
(a) 対応する HF モデルのトークナイザで正確に数えるか、
(b) 文字数ベースの保守的な推定にフォールバックする。

**どちらの方式を使ったかを必ず記録する**（数値の比較可能性に影響する。
docs/design/design.md §3.2(8)）。
"""

from __future__ import annotations

from typing import Literal, Protocol

Method = Literal["tokenizer", "char_heuristic"]


class TokenCounter(Protocol):
    @property
    def method(self) -> Method: ...

    def count(self, text: str) -> int: ...


class CharHeuristicCounter:
    """文字数ベースの推定。

    日本語は多くの多言語トークナイザで概ね 1文字あたり 1トークン前後に
    なる。安全側（過大評価）に寄せることで、コンテキスト予算を超過して
    出力が切れる事故を防ぐ。
    """

    def __init__(self, tokens_per_char: float = 1.1, ascii_tokens_per_char: float = 0.3) -> None:
        self.tokens_per_char = tokens_per_char
        self.ascii_tokens_per_char = ascii_tokens_per_char

    @property
    def method(self) -> Method:
        return "char_heuristic"

    def count(self, text: str) -> int:
        ascii_chars = sum(1 for ch in text if ch.isascii())
        other_chars = len(text) - ascii_chars
        estimate = ascii_chars * self.ascii_tokens_per_char + other_chars * self.tokens_per_char
        return int(estimate) + 1


class TransformersCounter:
    """HF トークナイザによる正確な計数。``transformers`` が必要。"""

    def __init__(self, tokenizer_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - 環境依存
            raise RuntimeError(
                "transformers が必要です（`uv sync --extra models`）。"
                "利用できない場合は char_heuristic を使ってください。"
            ) from exc
        self.tokenizer_name = tokenizer_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def method(self) -> Method:
        return "tokenizer"

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def make_counter(tokenizer: str | None = None) -> TokenCounter:
    """``tokenizer`` が指定され、かつ利用可能ならそれを使う。

    失敗時は例外を投げずに文字数推定へ落とす（実験が止まるより、
    方式が記録された近似値で進めるほうがよい）。
    """
    if tokenizer:
        try:
            return TransformersCounter(tokenizer)
        except Exception:
            return CharHeuristicCounter()
    return CharHeuristicCounter()

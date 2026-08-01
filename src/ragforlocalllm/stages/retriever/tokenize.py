"""日本語の分割。BM25 の語彙単位を決める。

**日本語では空白が単語境界にならない**ため、BM25 に素の空白分割を渡すと
文書全体が1トークンになり機能しない。分割方式は検索性能を直接左右する
実験軸なので、設定で選べるようにし、**使った方式を必ず記録する**
（docs/design/design.md §3.2(6)、§6.5）。

2方式を用意している。

- ``char_ngram`` … 文字bi-gram。追加依存なし・完全に決定的。
  **未知語や記号混じりの語に強い。** 「表4」「付録6」のような参照は
  形態素解析だと「表」「4」に割れて弁別力を失うが、bi-gram なら
  「表4」がそのまま特徴になる。ベースラインの実測では、まさにこの型の
  質問で dense 検索が失敗していた（design.md §9 Phase 2）
- ``sudachi`` … 形態素解析。言語的に妥当な単位で、語彙の一致が素直。
  ``uv sync --extra ja`` が必要

どちらが有効かはコーパス依存であり、決め打ちにせず比較する。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal, Protocol

TokenizerName = Literal["char_ngram", "sudachi"]

# 記号だけ・空白だけのトークンは BM25 の語彙から落とす
_NOISE = re.compile(r"^[\s　。、，．・！？（）「」『』【】〔〕：；…‥\-—–_/\\|]+$")


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...
    def describe(self) -> dict[str, Any]: ...


class CharNgramTokenizer:
    """文字n-gram。既定は bi-gram。

    正規化（NFKC + 小文字化）してから切る。「ＥＤＲ」と「EDR」を
    同じ語として扱わないと、表記ゆれで一致を落とす。
    """

    def __init__(self, n: int = 2) -> None:
        if n < 1:
            raise ValueError("n は 1 以上である必要があります")
        self.n = n

    def tokenize(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).lower()
        # 空白と改行は語をまたぐ区切りなので、ここで断つ
        segments = [s for s in re.split(r"\s+", normalized) if s and not _NOISE.match(s)]
        tokens: list[str] = []
        for segment in segments:
            if len(segment) < self.n:
                tokens.append(segment)
                continue
            tokens.extend(segment[i : i + self.n] for i in range(len(segment) - self.n + 1))
        return tokens

    def describe(self) -> dict[str, Any]:
        return {"tokenizer": "char_ngram", "n": self.n}


class SudachiTokenizer:
    """SudachiPy による形態素解析。

    ``mode`` は A（最短）/ B / C（最長）。検索では B が無難だが、
    複合語の扱いが変わるため実験軸として選べるようにしてある。
    """

    def __init__(
        self, mode: str = "B", drop_pos: tuple[str, ...] = ("助詞", "助動詞", "補助記号")
    ) -> None:
        if mode not in ("A", "B", "C"):
            raise ValueError("mode は A / B / C のいずれかです")
        self.mode = mode
        self.drop_pos = tuple(drop_pos)
        self._tokenizer: Any | None = None
        self._split_mode: Any | None = None

    def _ensure(self) -> tuple[Any, Any]:
        if self._tokenizer is None:
            try:
                from sudachipy import dictionary, tokenizer
            except ImportError as exc:  # pragma: no cover - 任意依存
                raise RuntimeError("SudachiPy が必要です: uv sync --extra ja") from exc
            self._tokenizer = dictionary.Dictionary().create()
            self._split_mode = getattr(tokenizer.Tokenizer.SplitMode, self.mode)
        return self._tokenizer, self._split_mode

    def tokenize(self, text: str) -> list[str]:
        tok, mode = self._ensure()
        normalized = unicodedata.normalize("NFKC", text)
        out: list[str] = []
        for morpheme in tok.tokenize(normalized, mode):
            if morpheme.part_of_speech()[0] in self.drop_pos:
                continue
            # 正規化形を使う。活用や表記ゆれを吸収する。
            surface = morpheme.normalized_form().lower()
            if surface.strip() and not _NOISE.match(surface):
                out.append(surface)
        return out

    def describe(self) -> dict[str, Any]:
        return {"tokenizer": "sudachi", "mode": self.mode, "drop_pos": list(self.drop_pos)}


def build_tokenizer(name: TokenizerName = "char_ngram", **kwargs: Any) -> Tokenizer:
    if name == "char_ngram":
        return CharNgramTokenizer(**kwargs)
    if name == "sudachi":
        return SudachiTokenizer(**kwargs)
    raise ValueError(f"未知のトークナイザです: {name}（char_ngram / sudachi）")

"""固定長チャンカー（ベースライン）と日本語向け再帰分割。

日本語では単語境界が空白で区切られないため、英語向けの既定区切り
文字（``" "`` 等）はほぼ機能しない。区切り文字を明示的に与える
（docs/design/design.md §3.2(2)）。
"""

from __future__ import annotations

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Chunk, Document

# 粗い順に試す。日本語の句点・読点・閉じ括弧を含める。
JA_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "）",
    "」",
    "、",
    " ",
    "",
)


def _make_chunk(doc: Document, index: int, text: str, start: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#c{index:04d}",
        doc_id=doc.doc_id,
        text=text,
        metadata={
            "char_start": start,
            "char_end": start + len(text),
            "n_chars": len(text),
        },
    )


@register("chunker", "fixed")
class FixedChunker:
    """文字数ベースの固定長分割。境界は考慮しない、素のベースライン。"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size は正の整数である必要があります")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap は 0 以上 chunk_size 未満である必要があります")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size]
            if not piece.strip():
                continue
            chunks.append(_make_chunk(doc, len(chunks), piece, start))
            if start + self.chunk_size >= len(text):
                break
        return chunks


@register("chunker", "recursive_ja")
class RecursiveJapaneseChunker:
    """日本語の区切り文字を粗い順に試す再帰分割。

    ``chunk_size`` を超えないよう、より細かい区切り文字へ降りていく。
    どの区切りでも収まらない場合のみ文字単位で切る。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: tuple[str, ...] | list[str] = JA_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size は正の整数である必要があります")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap は 0 以上 chunk_size 未満である必要があります")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = tuple(separators)

    def split(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []
        pieces = self._split_text(doc.text, self.separators)
        merged = self._merge(pieces)

        chunks: list[Chunk] = []
        cursor = 0
        for piece in merged:
            # 元テキスト上の位置を復元する（評価アンカーの解決に使う）
            found = doc.text.find(piece, cursor)
            start = found if found >= 0 else cursor
            chunks.append(_make_chunk(doc, len(chunks), piece, start))
            cursor = start + max(len(piece) - self.overlap, 1)
        return chunks

    def _split_text(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if not separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        # 区切り文字は直前の断片に残す（「。」で切ると文末が失われるため）
        rejoined = [p + sep for p in parts[:-1]] + [parts[-1]]

        out: list[str] = []
        for part in rejoined:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                out.append(part)
            else:
                out.extend(self._split_text(part, rest))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        """chunk_size に収まる範囲で断片を結合し、overlap を付与する。"""
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            if buffer and len(buffer) + len(piece) > self.chunk_size:
                merged.append(buffer)
                tail = buffer[-self.overlap :] if self.overlap else ""
                buffer = tail + piece
            else:
                buffer += piece
        if buffer.strip():
            merged.append(buffer)
        return merged

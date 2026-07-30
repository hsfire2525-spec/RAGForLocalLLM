"""プレーンテキスト / Markdown ローダー。

PDF ローダー（Phase 2、`pymupdf` 等）と違って追加依存がないため、
単体テストと疎通確認の既定として使う。
"""

from __future__ import annotations

from pathlib import Path

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


@register("loader", "text")
class TextLoader:
    """テキストファイル1件、またはディレクトリ配下のテキスト群を読む。

    Markdown の見出し (`#`) を拾って ``section_path`` を組み立てるのは
    Chunker 側の責務とし、ここでは生テキストと最小のメタデータのみ返す。
    """

    def __init__(self, encoding: str = "utf-8", recursive: bool = True) -> None:
        self.encoding = encoding
        self.recursive = recursive

    def load(self, path: Path) -> list[Document]:
        path = Path(path)
        if path.is_dir():
            pattern = "**/*" if self.recursive else "*"
            files = sorted(
                p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES
            )
            if not files:
                raise FileNotFoundError(f"{path} 配下にテキストファイルがありません")
        else:
            files = [path]

        docs: list[Document] = []
        for file in files:
            text = file.read_text(encoding=self.encoding)
            docs.append(
                Document(
                    doc_id=file.stem,
                    text=text,
                    metadata={
                        "source": str(file),
                        "title": file.stem,
                        "n_chars": len(text),
                        "loader": "text",
                    },
                )
            )
        return docs

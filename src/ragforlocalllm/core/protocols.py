"""各段のインターフェース。

実装はこの Protocol を満たし、registry に登録して設定から名前解決する。
設計方針は docs/design/design.md §4.2 を参照。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ragforlocalllm.core.types import Answer, Chunk, Document, Prompt, QueryState, ScoredChunk


@runtime_checkable
class Loader(Protocol):
    """文書ファイル → Document。"""

    def load(self, path: Path) -> list[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """Document → チャンク列。"""

    def split(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """テキスト → ベクトル。

    query 用と passage 用を分けることで、multilingual-e5 系の
    ``query: `` / ``passage: `` プレフィックス規約を実装側に閉じ込める。
    呼び出し側がプレフィックスを意識する設計にすると、付け忘れても
    動作してしまい精度だけが落ちるという発見しにくい失敗を招く。
    """

    @property
    def dim(self) -> int: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class Index(Protocol):
    """構築済みインデックス。"""

    def search(self, vectors: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        """各クエリベクトルに対し (chunk_id, score) の降順リストを返す。"""
        ...

    def save(self, directory: Path) -> None: ...


@runtime_checkable
class Indexer(Protocol):
    """チャンク列 → インデックス。"""

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> Index: ...

    def load(self, directory: Path) -> Index: ...


@runtime_checkable
class QueryTransform(Protocol):
    """クエリの前処理。複数クエリへ展開してよい。"""

    def transform(self, query: str) -> list[str]: ...


@runtime_checkable
class Retriever(Protocol):
    """検索。"""

    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]: ...


@runtime_checkable
class PostRetrievalStep(Protocol):
    """検索結果の後処理。

    リランク・重複除去・圧縮・並べ替えを同一インターフェースにし、
    リストとして合成する。順序自体が実験軸になる。
    """

    def process(self, state: QueryState) -> QueryState: ...


@runtime_checkable
class PromptBuilder(Protocol):
    """コンテキスト + 質問 → プロンプト。"""

    def build(self, state: QueryState) -> Prompt: ...


@runtime_checkable
class Generator(Protocol):
    """LLM推論。"""

    def generate(self, prompt: Prompt, schema: dict[str, Any] | None = None) -> Answer: ...

    def describe(self) -> dict[str, Any]:
        """ランレコードに記録するモデル情報（モデルID・量子化・コンテキスト長等）。"""
        ...


@runtime_checkable
class PostGenerationStep(Protocol):
    """生成後の検証と修正。PostRetrievalStep と同様にリストで合成する。"""

    def process(self, state: QueryState) -> QueryState: ...

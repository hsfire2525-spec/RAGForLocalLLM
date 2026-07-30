"""numpy による全探索インデックス。

FAISS を導入する前のベースライン、かつ CI 用。数千〜数万チャンク
規模では全探索でも十分速く、近似の影響を排除できるため、FAISS 側の
実装を検証する際の参照実装としても使う。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


class NumpyFlatIndex:
    """内積による全探索。ベクトルは正規化済みを前提とする（= cosine）。"""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"チャンク数 ({len(chunks)}) とベクトル数 ({vectors.shape[0]}) が一致しません"
            )
        self.chunks = chunks
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.by_id = {c.chunk_id: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, vectors: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if len(self.chunks) == 0:
            return [[] for _ in range(vectors.shape[0])]
        query = np.ascontiguousarray(vectors, dtype=np.float32)
        scores = query @ self.vectors.T
        k = min(top_k, len(self.chunks))
        results: list[list[tuple[str, float]]] = []
        for row in scores:
            # argpartition で上位k件を取り、その中だけ降順に整える
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            results.append([(self.chunks[i].chunk_id, float(row[i])) for i in top])
        return results

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "kind": "numpy_flat",
                    "n_chunks": len(self.chunks),
                    "dim": int(self.vectors.shape[1]),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> NumpyFlatIndex:
        directory = Path(directory)
        vectors = np.load(directory / VECTORS_FILE)
        chunks = [
            Chunk.model_validate_json(line)
            for line in (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(chunks, vectors)


@register("indexer", "numpy_flat")
class NumpyFlatIndexer:
    def __init__(self, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError("numpy_flat は現在 metric='cosine' のみ対応しています")
        self.metric = metric

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> NumpyFlatIndex:
        return NumpyFlatIndex(chunks, vectors)

    def load(self, directory: Path) -> NumpyFlatIndex:
        return NumpyFlatIndex.load(directory)

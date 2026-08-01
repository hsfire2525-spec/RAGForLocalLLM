"""FAISS インデックス。

``numpy_flat`` と**同じ結果を返すこと**を前提に置く。既定の
``IndexFlatIP`` は近似ではない全探索なので、正規化済みベクトルに対して
numpy 実装と数値誤差の範囲で一致する。これにより「FAISS に変えたら
数値が動いた」が近似の影響なのか実装の誤りなのかを切り分けられる。

``IndexHNSWFlat`` は近似検索。数万チャンク規模までは全探索で十分速く、
近似を入れると**検索の取りこぼしが精度低下として現れて手法比較を汚す**
ため、既定は ``flat`` にしてある。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Chunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "index_meta.json"


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - 任意依存
        raise RuntimeError("faiss が必要です: uv sync --extra retrieval") from exc
    return faiss


class FaissIndex:
    """FAISS による検索。ベクトルは正規化済みを前提とする（= cosine）。

    ``chunks`` を公開するのは必須。**gold引用の解決にチャンク本文が要る**
    ため、これが無いと評価が回らない（``eval/runner.py`` 参照）。
    """

    def __init__(self, chunks: list[Chunk], index: Any, *, factory: str = "flat") -> None:
        if len(chunks) != index.ntotal:
            raise ValueError(
                f"チャンク数 ({len(chunks)}) とベクトル数 ({index.ntotal}) が一致しません"
            )
        self.chunks = chunks
        self.index = index
        self.factory = factory
        self.by_id = {c.chunk_id: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, vectors: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if not self.chunks:
            return [[] for _ in range(vectors.shape[0])]
        query = np.ascontiguousarray(vectors, dtype=np.float32)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, k)
        results: list[list[tuple[str, float]]] = []
        for row_scores, row_indices in zip(scores, indices, strict=True):
            # FAISS は該当が k 件に満たないと -1 を返す
            results.append(
                [
                    (self.chunks[i].chunk_id, float(s))
                    for s, i in zip(row_scores, row_indices, strict=True)
                    if i >= 0
                ]
            )
        return results

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    def save(self, directory: Path) -> None:
        faiss = _import_faiss()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "kind": "faiss",
                    "factory": self.factory,
                    "n_chunks": len(self.chunks),
                    "dim": int(self.index.d),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> FaissIndex:
        faiss = _import_faiss()
        directory = Path(directory)
        index = faiss.read_index(str(directory / INDEX_FILE))
        chunks = [
            Chunk.model_validate_json(line)
            for line in (directory / CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta_path = directory / META_FILE
        factory = "flat"
        if meta_path.exists():
            factory = str(json.loads(meta_path.read_text(encoding="utf-8")).get("factory", "flat"))
        return cls(chunks, index, factory=factory)


@register("indexer", "faiss")
class FaissIndexer:
    """FAISS インデックスの構築。

    Parameters
    ----------
    metric:
        ``cosine`` のみ。正規化済みベクトルの内積として実装する。
    factory:
        ``flat`` — 全探索（既定、近似なし）。
        ``hnsw`` — 近似。大規模化したときのみ使う。
    hnsw_m / hnsw_ef_construction / hnsw_ef_search:
        ``factory=hnsw`` のときのパラメータ。
    """

    def __init__(
        self,
        metric: str = "cosine",
        factory: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ) -> None:
        if metric != "cosine":
            raise ValueError("faiss インデクサは現在 metric='cosine' のみ対応しています")
        if factory not in ("flat", "hnsw"):
            raise ValueError("factory は flat / hnsw のいずれかです")
        self.metric = metric
        self.factory = factory
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

    def build(self, chunks: list[Chunk], vectors: np.ndarray) -> FaissIndex:
        faiss = _import_faiss()
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = int(matrix.shape[1])

        if self.factory == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(matrix)
        return FaissIndex(chunks, index, factory=self.factory)

    def load(self, directory: Path) -> FaissIndex:
        return FaissIndex.load(directory)

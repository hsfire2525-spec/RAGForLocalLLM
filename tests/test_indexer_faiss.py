"""FAISS インデックスの検証。

**要点は「numpy_flat と同じ結果を返すこと」。** これが保証されていないと、
インデクサを差し替えたときの数値変化が、近似の影響なのか実装の誤りなのか
切り分けられない。既定の ``IndexFlatIP`` は近似ではないので一致するはず。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragforlocalllm.core.registry import build
from ragforlocalllm.core.types import Chunk

faiss = pytest.importorskip("faiss")

from ragforlocalllm.stages.indexer.faiss_index import FaissIndex, FaissIndexer  # noqa: E402
from ragforlocalllm.stages.indexer.numpy_flat import NumpyFlatIndexer  # noqa: E402


def make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i:03d}",
            doc_id="doc",
            text=f"本文 {i}",
            metadata={"page": i // 3 + 1, "page_start": i // 3 + 1, "page_end": i // 3 + 1},
        )
        for i in range(n)
    ]


def normalized(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def data() -> tuple[list[Chunk], np.ndarray]:
    rng = np.random.default_rng(42)
    return make_chunks(30), normalized(rng, 30, 16)


# ----------------------------------------------------------------------


def test_registered_under_expected_name() -> None:
    assert isinstance(build("indexer", {"type": "faiss"}), FaissIndexer)


def test_flat_matches_numpy_exhaustive_search(data: tuple[list[Chunk], np.ndarray]) -> None:
    """**近似なしなので numpy 全探索と一致すること。**"""
    chunks, vectors = data
    rng = np.random.default_rng(7)
    queries = normalized(rng, 5, 16)

    faiss_hits = FaissIndexer().build(chunks, vectors).search(queries, 5)
    numpy_hits = NumpyFlatIndexer().build(chunks, vectors).search(queries, 5)

    for a, b in zip(faiss_hits, numpy_hits, strict=True):
        assert [cid for cid, _ in a] == [cid for cid, _ in b]
        for (_, sa), (_, sb) in zip(a, b, strict=True):
            assert sa == pytest.approx(sb, abs=1e-5)


def test_search_returns_scores_in_descending_order(data: tuple[list[Chunk], np.ndarray]) -> None:
    chunks, vectors = data
    hits = FaissIndexer().build(chunks, vectors).search(vectors[:1], 5)[0]
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    # 自分自身が最上位（正規化済みなので内積は 1.0）
    assert hits[0][0] == "c000"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_top_k_larger_than_corpus_is_clamped() -> None:
    chunks, vectors = make_chunks(3), normalized(np.random.default_rng(1), 3, 8)
    hits = FaissIndexer().build(chunks, vectors).search(vectors[:1], 10)[0]
    assert len(hits) == 3


def test_empty_index_returns_empty_results() -> None:
    index = FaissIndexer().build([], np.zeros((0, 8), dtype=np.float32))
    assert len(index) == 0
    assert index.search(np.zeros((2, 8), dtype=np.float32), 5) == [[], []]


def test_chunks_are_exposed_for_gold_resolution(data: tuple[list[Chunk], np.ndarray]) -> None:
    """gold引用の解決にチャンク本文が要る。公開が無いと評価が回らない。"""
    chunks, vectors = data
    index = FaissIndexer().build(chunks, vectors)
    assert len(index.chunks) == len(chunks)
    assert index.get("c005") is not None
    assert index.get("missing") is None


def test_mismatched_lengths_are_rejected(data: tuple[list[Chunk], np.ndarray]) -> None:
    chunks, vectors = data
    index = faiss.IndexFlatIP(16)
    index.add(vectors)
    with pytest.raises(ValueError, match="一致しません"):
        FaissIndex(chunks[:5], index)


# ----------------------------------------------------------------------
# 永続化
# ----------------------------------------------------------------------


def test_roundtrip_preserves_results_and_metadata(
    data: tuple[list[Chunk], np.ndarray], tmp_path: Path
) -> None:
    chunks, vectors = data
    original = FaissIndexer().build(chunks, vectors)
    original.save(tmp_path)

    loaded = FaissIndexer().load(tmp_path)
    assert len(loaded) == len(original)
    assert loaded.search(vectors[:2], 5) == original.search(vectors[:2], 5)
    # チャンクのメタデータ（ページ・節）が失われないこと
    assert loaded.get("c004") is not None
    assert loaded.get("c004").page == chunks[4].page  # type: ignore[union-attr]


def test_hnsw_is_opt_in(data: tuple[list[Chunk], np.ndarray]) -> None:
    """近似は明示的に選ぶもの。既定で入ると手法比較が汚れる。"""
    chunks, vectors = data
    assert FaissIndexer().factory == "flat"
    index = FaissIndexer(factory="hnsw", hnsw_m=8).build(chunks, vectors)
    assert index.factory == "hnsw"
    assert len(index.search(vectors[:1], 3)[0]) == 3


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="metric"):
        FaissIndexer(metric="l2")
    with pytest.raises(ValueError, match="factory"):
        FaissIndexer(factory="ivf")

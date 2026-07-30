"""キャッシュの単体テスト。

埋め込みキャッシュが壊れると実験サイクルの速度が落ちるだけでなく、
古い値を返した場合は結果そのものが誤る。実 Cache（NullCache ではない）
に対する往復を固定する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ragforlocalllm.core.cache import Cache, NullCache, content_key


def test_json_roundtrip(tmp_path: Path) -> None:
    with Cache(tmp_path) as cache:
        assert cache.get_json("ns", "missing") is None
        cache.put_json("ns", "k", {"a": 1, "b": ["x"]})
        assert cache.get_json("ns", "k") == {"a": 1, "b": ["x"]}


def test_json_survives_reopen(tmp_path: Path) -> None:
    with Cache(tmp_path) as cache:
        cache.put_json("ns", "k", 42)
    with Cache(tmp_path) as reopened:
        assert reopened.get_json("ns", "k") == 42


def test_array_roundtrip(tmp_path: Path) -> None:
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    with Cache(tmp_path) as cache:
        assert cache.get_array("embeddings", "missing") is None
        cache.put_array("embeddings", "k", vectors)
        loaded = cache.get_array("embeddings", "k")

    assert loaded is not None
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, vectors)


def test_array_leaves_no_partial_file(tmp_path: Path) -> None:
    """一時ファイルが残らないこと（np.save の拡張子自動付加に注意）。"""
    with Cache(tmp_path) as cache:
        cache.put_array("embeddings", "k", np.zeros((2, 2), dtype=np.float32))
    leftovers = list(Path(tmp_path).rglob("*.part*"))
    assert leftovers == []


def test_namespaces_are_isolated(tmp_path: Path) -> None:
    with Cache(tmp_path) as cache:
        cache.put_json("a", "k", 1)
        cache.put_json("b", "k", 2)
        assert cache.get_json("a", "k") == 1
        assert cache.get_json("b", "k") == 2


def test_stats_counts_entries(tmp_path: Path) -> None:
    with Cache(tmp_path) as cache:
        cache.put_json("llm", "k1", "x")
        cache.put_json("llm", "k2", "y")
        cache.put_array("embeddings", "v", np.zeros((1, 2), dtype=np.float32))
        stats = cache.stats()
    assert stats["llm"] == 2
    assert stats["_blobs"] == 1


def test_content_key_is_order_independent_for_dicts() -> None:
    a = content_key("passages", {"model": "e5", "dim": 8})
    b = content_key("passages", {"dim": 8, "model": "e5"})
    assert a == b


def test_content_key_differs_for_different_content() -> None:
    assert content_key("passages", ["a"]) != content_key("passages", ["b"])


def test_null_cache_never_stores() -> None:
    cache = NullCache()
    cache.put_json("ns", "k", 1)
    cache.put_array("ns", "k", np.zeros((1, 1), dtype=np.float32))
    assert cache.get_json("ns", "k") is None
    assert cache.get_array("ns", "k") is None
    assert cache.stats() == {}

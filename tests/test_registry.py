from __future__ import annotations

import pytest

from ragforlocalllm import stages  # noqa: F401 - レジストリ登録
from ragforlocalllm.core import registry
from ragforlocalllm.core.registry import RegistryError


def test_expected_components_are_registered() -> None:
    assert "recursive_ja" in registry.available("chunker")
    assert "dense" in registry.available("retriever")
    assert "extractive" in registry.available("generator")
    assert "openai_compat" in registry.available("generator")
    assert "reorder_lost_in_middle" in registry.available("post_retrieval")


def test_build_passes_kwargs() -> None:
    chunker = registry.build("chunker", {"type": "fixed", "chunk_size": 100, "overlap": 10})
    assert chunker.chunk_size == 100
    assert chunker.overlap == 10


def test_unknown_type_lists_candidates() -> None:
    with pytest.raises(RegistryError) as exc:
        registry.build("chunker", {"type": "recursive_jp"})
    message = str(exc.value)
    assert "登録済み" in message
    # 近い名前を候補として出す
    assert "recursive_ja" in message


def test_unknown_kwarg_is_rejected_with_suggestion() -> None:
    with pytest.raises(RegistryError) as exc:
        registry.build("chunker", {"type": "fixed", "chunk_sizes": 100})
    message = str(exc.value)
    assert "受け付けません" in message
    assert "chunk_size" in message


def test_missing_required_kwarg_is_reported() -> None:
    with pytest.raises(RegistryError, match="必須"):
        registry.build("prompt", {"type": "template"})  # path が必須


def test_injected_key_conflict_is_rejected() -> None:
    """設定側から注入対象のキーを上書きしようとした場合に落ちること。"""
    with pytest.raises(RegistryError, match="注入されます"):
        registry.build(
            "retriever",
            {"type": "dense", "embedder": "bogus"},
            embedder=object(),
            index=object(),
        )


def test_missing_type_key() -> None:
    with pytest.raises(RegistryError, match="'type' がありません"):
        registry.build("chunker", {"chunk_size": 100})

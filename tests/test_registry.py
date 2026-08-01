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


def test_injections_are_filtered_to_what_the_component_declares() -> None:
    """**注入は「使うものだけ」渡す。**

    合成する側（hybrid 検索器）は子ごとに必要な依存を知らずに、一律で
    embedder と index を渡す。BM25 は埋め込み器を使わないため、受け取ると
    宣言していない依存は落とさないと合成が成立しない。
    """

    @registry.register("injection_test", "needs_index_only")
    class NeedsIndexOnly:
        def __init__(self, index: object, top_k: int = 5) -> None:
            self.index = index
            self.top_k = top_k

    built = registry.build("injection_test", {"type": "needs_index_only"}, embedder="E", index="I")
    assert built.index == "I"
    assert not hasattr(built, "embedder")


def test_component_declaring_a_dependency_still_receives_it() -> None:
    @registry.register("injection_test", "needs_both")
    class NeedsBoth:
        def __init__(self, embedder: object, index: object) -> None:
            self.embedder = embedder
            self.index = index

    built = registry.build("injection_test", {"type": "needs_both"}, embedder="E", index="I")
    assert (built.embedder, built.index) == ("E", "I")


def test_config_key_colliding_with_an_injection_is_still_an_error() -> None:
    """注入を絞っても、設定側からの上書きは許さない。"""

    @registry.register("injection_test", "collides")
    class Collides:
        def __init__(self, index: object) -> None:
            self.index = index

    with pytest.raises(RegistryError, match="実行時に注入されます"):
        registry.build("injection_test", {"type": "collides", "index": "cfg"}, index="runtime")

"""Phase 0 の完了条件: 設定YAMLでパイプラインが通り、trace が出ること。"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragforlocalllm.core.cache import NullCache
from ragforlocalllm.core.config import load_config
from ragforlocalllm.core.indexing import build_index
from ragforlocalllm.core.pipeline import QueryPipeline

CONFIG = Path("configs/smoke.yaml")


@pytest.fixture(scope="module")
def pipeline_and_index(tmp_path_factory: pytest.TempPathFactory):
    cfg = load_config(CONFIG, search_dir=Path("configs"))
    root = tmp_path_factory.mktemp("indexes")
    built = build_index(cfg, cache=NullCache(), root=root)
    pipeline = QueryPipeline.from_config(cfg, embedder=built.embedder, index=built.index)
    return pipeline, built


def test_index_is_built(pipeline_and_index) -> None:
    _, built = pipeline_and_index
    assert built.stats["n_chunks"] > 0
    assert (built.directory / "index_meta.json").exists()
    assert (built.directory / "signature.json").exists()


def test_index_is_reused_on_second_build(tmp_path: Path) -> None:
    cfg = load_config(CONFIG, search_dir=Path("configs"))
    first = build_index(cfg, cache=NullCache(), root=tmp_path)
    second = build_index(cfg, cache=NullCache(), root=tmp_path)
    assert first.stats["reused"] is False
    assert second.stats["reused"] is True
    assert first.signature == second.signature


def test_query_produces_answer_and_trace(pipeline_and_index) -> None:
    pipeline, _ = pipeline_and_index
    state = pipeline.run("情報セキュリティ基本方針を承認するのは誰か。")

    assert state.answer is not None
    assert state.prompt is not None
    assert state.contexts, "コンテキストが空"

    stages = [t.stage for t in state.trace]
    assert stages[0] == "query_transform"
    assert "retriever" in stages
    assert "generator" in stages
    assert stages[-1].startswith("post_generation")
    assert all(t.duration_ms >= 0 for t in state.trace)
    assert state.total_duration_ms > 0


def test_extractive_generator_finds_the_answer(pipeline_and_index) -> None:
    """LLMなしの下限ベースラインでも、明示的な記述は拾えること。"""
    pipeline, _ = pipeline_and_index
    state = pipeline.run("情報セキュリティ基本方針を承認するのは誰か。")
    assert state.answer is not None
    assert "経営者" in state.answer.text


def test_citations_resolve_to_chunk_ids(pipeline_and_index) -> None:
    pipeline, _ = pipeline_and_index
    state = pipeline.run("パスワードは何文字以上必要か。")
    assert state.answer is not None
    assert state.prompt is not None
    for chunk_id in state.answer.citations:
        assert chunk_id in state.prompt.context_chunk_ids


def test_context_token_budget_is_respected(pipeline_and_index) -> None:
    pipeline, _ = pipeline_and_index
    state = pipeline.run("バックアップは何世代分保管するか。")
    assert state.prompt is not None
    assert state.prompt.token_estimate is not None
    assert state.prompt.token_estimate_method == "char_heuristic"
    # コンテキストは予算内に収まっている（プロンプト全体は指示文を含むため少し超える）
    assert len(state.prompt.context_chunk_ids) <= 3

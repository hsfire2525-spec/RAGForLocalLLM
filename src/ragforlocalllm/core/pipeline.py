"""段の合成と trace 記録。

各段は「QueryState → QueryState」の変換として実行し、所要時間と
RSS を trace に残す。どの段で情報が落ちたのか、どの段が時間を
食っているのかを事後に追えるようにするのが目的
（docs/design/design.md §4.1、原則6）。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ragforlocalllm.core import registry
from ragforlocalllm.core.config import ExperimentConfig
from ragforlocalllm.core.env import rss_mb
from ragforlocalllm.core.protocols import (
    Generator,
    PostGenerationStep,
    PostRetrievalStep,
    PromptBuilder,
    QueryTransform,
    Retriever,
)
from ragforlocalllm.core.types import QueryState, StageTrace


@contextmanager
def traced(state: QueryState, stage: str, impl: str) -> Iterator[dict[str, Any]]:
    """段の実行を計測し、trace に1件追加する。

    yield される dict に観測値を入れると ``StageTrace.info`` に入る。
    例外が発生した場合も trace を残す（どこで落ちたかを見るため）。
    """
    info: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        yield info
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state.trace.append(
            StageTrace(
                stage=stage,
                impl=impl,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                rss_mb=rss_mb(),
                info=info,
            )
        )


@dataclass(frozen=True)
class QueryPipeline:
    """設定から組み立てたクエリ側パイプライン。"""

    query_transform: QueryTransform
    retriever: Retriever
    post_retrieval: tuple[PostRetrievalStep, ...]
    prompt_builder: PromptBuilder
    generator: Generator
    post_generation: tuple[PostGenerationStep, ...]
    top_k: int
    impl_names: dict[str, Any]

    @classmethod
    def from_config(cls, config: ExperimentConfig, *, embedder: Any, index: Any) -> QueryPipeline:
        """設定からパイプラインを組み立てる。

        Retriever は構築済みの埋め込み器とインデックスに依存するため、
        設定に書けない依存として注入する。
        """
        query = config.query
        return cls(
            query_transform=registry.build("query_transform", query.query_transform.as_spec()),
            retriever=registry.build(
                "retriever", query.retriever.as_spec(), embedder=embedder, index=index
            ),
            post_retrieval=tuple(
                registry.build("post_retrieval", spec.as_spec()) for spec in query.post_retrieval
            ),
            prompt_builder=registry.build("prompt", query.prompt.as_spec()),
            generator=registry.build("generator", query.generator.as_spec()),
            post_generation=tuple(
                registry.build("post_generation", spec.as_spec()) for spec in query.post_generation
            ),
            top_k=int(getattr(query.retriever, "top_k", 5) or 5),
            impl_names={
                "query_transform": query.query_transform.type,
                "retriever": query.retriever.type,
                "post_retrieval": [s.type for s in query.post_retrieval],
                "prompt": query.prompt.type,
                "generator": query.generator.type,
                "post_generation": [s.type for s in query.post_generation],
            },
        )

    def run(self, question: str) -> QueryState:
        state = QueryState.new(question)
        names = self.impl_names

        with traced(state, "query_transform", names["query_transform"]) as info:
            state.queries = self.query_transform.transform(question)
            info["n_queries"] = len(state.queries)

        with traced(state, "retriever", names["retriever"]) as info:
            state.retrieved = self.retriever.retrieve(state.queries, self.top_k)
            state.contexts = list(state.retrieved)
            info["n_retrieved"] = len(state.retrieved)
            info["top_k"] = self.top_k

        for i, step in enumerate(self.post_retrieval):
            with traced(state, f"post_retrieval[{i}]", names["post_retrieval"][i]) as info:
                before = len(state.contexts)
                state = step.process(state)
                info["n_in"] = before
                info["n_out"] = len(state.contexts)

        with traced(state, "prompt", names["prompt"]) as info:
            state.prompt = self.prompt_builder.build(state)
            info["n_contexts"] = len(state.prompt.context_chunk_ids)
            info["token_estimate"] = state.prompt.token_estimate
            info["token_estimate_method"] = state.prompt.token_estimate_method
            if state.prompt.n_dropped or state.prompt.n_truncated:
                info["budget_dropped"] = state.prompt.n_dropped
                info["budget_truncated"] = state.prompt.n_truncated

        with traced(state, "generator", names["generator"]) as info:
            assert state.prompt is not None
            state.answer = self.generator.generate(state.prompt)
            info["chars"] = len(state.answer.text)
            if state.answer.usage is not None:
                info["completion_tokens"] = state.answer.usage.completion_tokens

        for i, step in enumerate(self.post_generation):
            with traced(state, f"post_generation[{i}]", names["post_generation"][i]) as info:
                state = step.process(state)
                if state.answer is not None:
                    info["abstained"] = state.answer.abstained
                    info["n_citations"] = len(state.answer.citations)

        return state

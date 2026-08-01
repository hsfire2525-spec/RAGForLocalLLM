"""段の実装。

このモジュールを import した時点で、すべての実装が registry に
登録される。新しい実装を追加したら、下の import に1行足すこと
（設定ファイルから名前で解決できるようにするため）。
"""

from __future__ import annotations

# ruff: noqa: F401 - 副作用（レジストリ登録）のための import
from ragforlocalllm.stages.chunker import fixed as _chunker_fixed
from ragforlocalllm.stages.embedder import hashing as _embedder_hashing
from ragforlocalllm.stages.embedder import (
    sentence_transformers as _embedder_sentence_transformers,
)
from ragforlocalllm.stages.generator import extractive as _generator_extractive
from ragforlocalllm.stages.generator import openai_compat as _generator_openai_compat
from ragforlocalllm.stages.indexer import faiss_index as _indexer_faiss
from ragforlocalllm.stages.indexer import numpy_flat as _indexer_numpy_flat
from ragforlocalllm.stages.loader import pdf as _loader_pdf
from ragforlocalllm.stages.loader import text as _loader_text
from ragforlocalllm.stages.post_generation import checks as _post_generation_checks
from ragforlocalllm.stages.post_retrieval import basic as _post_retrieval_basic
from ragforlocalllm.stages.prompt import template as _prompt_template
from ragforlocalllm.stages.query_transform import rules as _query_transform_rules
from ragforlocalllm.stages.retriever import dense as _retriever_dense

__all__: list[str] = []

"""インデックス構築と読み込み。

インデックス構築（オフライン）とクエリ実行（オンライン）を分離し、
クエリ側だけを変える実験でインデックスを再利用できるようにする。
再利用の可否は ``index_signature``（index設定 + コーパスのSHA-256）で
判定する（docs/design/design.md §3.1、§4.4）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ragforlocalllm.core import registry
from ragforlocalllm.core.cache import Cache, content_key
from ragforlocalllm.core.config import ExperimentConfig
from ragforlocalllm.core.env import corpus_signature
from ragforlocalllm.core.types import Chunk, Document

DEFAULT_INDEX_ROOT = Path(".cache/indexes")
SIGNATURE_FILE = "signature.json"


@dataclass
class BuiltIndex:
    index: Any
    embedder: Any
    signature: str
    directory: Path
    stats: dict[str, Any]


def index_directory(signature: str, root: Path | str = DEFAULT_INDEX_ROOT) -> Path:
    return Path(root) / signature


def build_index(
    config: ExperimentConfig,
    *,
    cache: Cache | None = None,
    root: Path | str = DEFAULT_INDEX_ROOT,
    force: bool = False,
) -> BuiltIndex:
    """コーパスからインデックスを構築する。既存の成果物があれば再利用する。"""
    corpus = Path(config.corpus)
    if not corpus.exists():
        raise FileNotFoundError(
            f"コーパスがありません: {corpus}\n"
            "  `python scripts/fetch_corpus.py` で取得してください"
            "（コーパスはリポジトリにコミットされません）"
        )

    corpus_sha = corpus_signature(corpus)
    signature = config.index_signature(corpus_sha)
    directory = index_directory(signature, root)

    embedder = registry.build("embedder", config.index.embedder.as_spec())
    indexer = registry.build("indexer", config.index.indexer.as_spec())

    if directory.exists() and not force:
        index = indexer.load(directory)
        return BuiltIndex(
            index=index,
            embedder=embedder,
            signature=signature,
            directory=directory,
            stats={"reused": True, "n_chunks": len(index), "corpus_sha256": corpus_sha},
        )

    loader = registry.build("loader", config.index.loader.as_spec())
    chunker = registry.build("chunker", config.index.chunker.as_spec())

    documents: list[Document] = loader.load(corpus)
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunker.split(doc))
    if not chunks:
        raise ValueError(f"{corpus} からチャンクが1件も生成されませんでした")

    vectors = _embed_passages(embedder, chunks, config, cache)
    index = indexer.build(chunks, vectors)

    directory.mkdir(parents=True, exist_ok=True)
    index.save(directory)
    (directory / SIGNATURE_FILE).write_text(
        json.dumps(
            {
                "signature": signature,
                "corpus": str(corpus),
                "corpus_sha256": corpus_sha,
                "index_config": config.index.model_dump(mode="json"),
                "n_documents": len(documents),
                "n_chunks": len(chunks),
                "dim": int(vectors.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return BuiltIndex(
        index=index,
        embedder=embedder,
        signature=signature,
        directory=directory,
        stats={
            "reused": False,
            "n_documents": len(documents),
            "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]),
            "corpus_sha256": corpus_sha,
        },
    )


def _embed_passages(
    embedder: Any, chunks: list[Chunk], config: ExperimentConfig, cache: Cache | None
) -> np.ndarray:
    """チャンクを埋め込む。キャッシュがあればコーパス単位で再利用する。

    埋め込みは実験の中で最も繰り返される重い処理であり、ここを
    キャッシュしないと試行回数を確保できない。
    """
    embedder_spec = config.index.embedder.model_dump(mode="json")
    if cache is not None:
        key = content_key("passages", embedder_spec, [c.text for c in chunks])
        cached = cache.get_array("embeddings", key)
        if cached is not None and cached.shape[0] == len(chunks):
            return cached

    vectors: np.ndarray = embedder.embed_passages([c.text for c in chunks])
    if cache is not None:
        cache.put_array(
            "embeddings", content_key("passages", embedder_spec, [c.text for c in chunks]), vectors
        )
    return vectors

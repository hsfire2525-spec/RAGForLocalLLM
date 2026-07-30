"""密ベクトル検索。

複数クエリ（QueryTransform が展開した場合）の結果は、チャンクごとに
最大スコアで統合する（RRF によるハイブリッド統合は Phase 3）。
"""

from __future__ import annotations

from ragforlocalllm.core.protocols import Embedder
from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import ScoredChunk


@register("retriever", "dense")
class DenseRetriever:
    """埋め込み器とインデックスは実行時に注入される。"""

    def __init__(self, embedder: Embedder, index: object, top_k: int = 5) -> None:
        self.embedder = embedder
        self.index = index
        self.top_k = top_k

    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]:
        if not queries:
            return []
        k = top_k or self.top_k
        vectors = self.embedder.embed_queries(queries)
        # 複数クエリの場合、統合前に各クエリで多めに取る
        per_query_k = k if len(queries) == 1 else min(k * 2, k + 10)
        results = self.index.search(vectors, per_query_k)  # type: ignore[attr-defined]

        best: dict[str, float] = {}
        for hits in results:
            for chunk_id, score in hits:
                if chunk_id not in best or score > best[chunk_id]:
                    best[chunk_id] = score

        ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[ScoredChunk] = []
        for chunk_id, score in ordered:
            chunk = self.index.get(chunk_id)  # type: ignore[attr-defined]
            if chunk is None:
                continue
            out.append(ScoredChunk(chunk=chunk, score=score, provenance={"dense": score}))
        return out

"""ハイブリッド検索。複数の検索器の結果を RRF で統合する。

**スコアを直接足してはいけない。** dense のコサイン類似度は実測で
上位が 0.83〜0.89 に密集し、BM25 のスコアは 0〜数十の非有界な値を取る
（docs/design/design.md §9 Phase 2）。尺度も分布も違うものを加重和すると、
重みの調整が「どちらを使うか」ではなく「スケールを合わせる作業」になり、
コーパスが変わるたびに壊れる。

**RRF（Reciprocal Rank Fusion）は順位だけを使う。** スコアの正規化が
要らず、パラメータは実質 ``rrf_k`` ひとつで済む。

    score(c) = Σ_r  weight_r / (rrf_k + rank_r(c))

``rrf_k`` は上位の効きを鈍らせる定数で、慣例的に 60。小さくすると
1位が強く効き、大きくすると順位差が均される。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ragforlocalllm.core import registry
from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import ScoredChunk

DEFAULT_RETRIEVERS: tuple[dict[str, Any], ...] = (
    {"type": "dense", "top_k": 20},
    {"type": "sparse", "top_k": 20},
)


@register("retriever", "hybrid")
class HybridRetriever:
    """複数の検索器を RRF で統合する。

    Parameters
    ----------
    retrievers:
        統合する検索器の設定リスト。**各要素の ``top_k`` は最終的な
        件数ではなく候補数**で、最終件数より十分大きくする。候補が
        少ないと統合の材料が足りず、片方の検索器の順位がそのまま出る。
    weights:
        検索器ごとの重み。省略時は等価。
    rrf_k:
        RRF の定数。小さいほど上位を重視する。
    """

    def __init__(
        self,
        embedder: object,
        index: object,
        top_k: int = 5,
        retrievers: Sequence[Mapping[str, Any]] = DEFAULT_RETRIEVERS,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid には少なくとも1つの retrievers が必要です")
        if rrf_k <= 0:
            raise ValueError("rrf_k は正の整数である必要があります")
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.specs = [dict(spec) for spec in retrievers]
        self.children = [
            registry.build("retriever", spec, embedder=embedder, index=index) for spec in self.specs
        ]
        if weights is not None and len(weights) != len(self.children):
            raise ValueError("weights の数が retrievers の数と一致しません")
        self.weights = list(weights) if weights is not None else [1.0] * len(self.children)

    # ------------------------------------------------------------------

    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]:
        if not queries:
            return []
        k = top_k or self.top_k

        fused: dict[str, float] = {}
        chunks: dict[str, ScoredChunk] = {}
        provenance: dict[str, dict[str, float]] = {}

        for spec, child, weight in zip(self.specs, self.children, self.weights, strict=True):
            name = str(spec.get("type"))
            # 子の候補数はその設定に従う（最終件数で切らない）
            results = child.retrieve(queries, int(spec.get("top_k") or k))
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk.chunk_id
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (self.rrf_k + rank)
                chunks.setdefault(chunk_id, item)
                # 各検索器の元スコアと順位を残す。どちらが効いたかを
                # 事後に分析できないと、ハイブリッドの是非を判断できない。
                entry = provenance.setdefault(chunk_id, {})
                entry[name] = item.score
                entry[f"{name}_rank"] = float(rank)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            ScoredChunk(
                chunk=chunks[chunk_id].chunk,
                score=score,
                provenance={**provenance.get(chunk_id, {}), "rrf": score},
            )
            for chunk_id, score in ordered
        ]

    def describe(self) -> dict[str, Any]:
        children = []
        for child in self.children:
            describe = getattr(child, "describe", None)
            children.append(
                describe() if callable(describe) else {"retriever": type(child).__name__}
            )
        return {
            "retriever": "hybrid",
            "fusion": "rrf",
            "rrf_k": self.rrf_k,
            "weights": self.weights,
            "children": children,
        }

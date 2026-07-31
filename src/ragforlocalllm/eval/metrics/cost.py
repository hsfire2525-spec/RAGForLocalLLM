"""コスト指標。環境間比較に必須（docs/design/design.md §6.3）。

平均ではなく **p50 / p95** を見る。RAGのレイテンシは埋め込みキャッシュの
ヒット・ミスや生成長で裾が重くなり、平均が実感と乖離するため。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ragforlocalllm.core.types import QueryState


def percentile(values: Sequence[float], q: float) -> float:
    """線形補間なしの素朴なパーセンタイル（質問数が数十件のため十分）。"""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


@dataclass
class CostAccumulator:
    """ランを通したコストの集計。"""

    total_ms: list[float] = field(default_factory=list)
    stage_ms: dict[str, list[float]] = field(default_factory=dict)
    completion_tokens: list[int] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    context_tokens: list[int] = field(default_factory=list)
    n_dropped: list[int] = field(default_factory=list)
    peak_rss_mb: float = 0.0

    def add(self, state: QueryState) -> None:
        self.total_ms.append(state.total_duration_ms)
        for entry in state.trace:
            self.stage_ms.setdefault(entry.stage, []).append(entry.duration_ms)
            if entry.rss_mb is not None:
                self.peak_rss_mb = max(self.peak_rss_mb, entry.rss_mb)
        if state.prompt is not None:
            if state.prompt.token_estimate is not None:
                self.context_tokens.append(state.prompt.token_estimate)
            self.n_dropped.append(state.prompt.n_dropped)
        if state.answer is not None and state.answer.usage is not None:
            if state.answer.usage.completion_tokens is not None:
                self.completion_tokens.append(state.answer.usage.completion_tokens)
            if state.answer.usage.prompt_tokens is not None:
                self.prompt_tokens.append(state.answer.usage.prompt_tokens)

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "latency_ms_p50": round(percentile(self.total_ms, 0.50), 1),
            "latency_ms_p95": round(percentile(self.total_ms, 0.95), 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "stage_ms_p50": {
                stage: round(percentile(values, 0.50), 1)
                for stage, values in sorted(self.stage_ms.items())
            },
        }
        if self.completion_tokens:
            payload["completion_tokens_p50"] = percentile(
                [float(t) for t in self.completion_tokens], 0.50
            )
        if self.prompt_tokens:
            payload["prompt_tokens_p50"] = percentile([float(t) for t in self.prompt_tokens], 0.50)
        if self.context_tokens:
            payload["context_tokens_p50"] = percentile(
                [float(t) for t in self.context_tokens], 0.50
            )
        if self.n_dropped:
            # 予算に収まらず落としたチャンク。検索が正解を取れていても
            # ここで落ちていれば回答できないため、精度低下の切り分けに使う。
            payload["chunks_dropped_total"] = sum(self.n_dropped)
        return payload

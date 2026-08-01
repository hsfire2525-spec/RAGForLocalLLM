"""評価メトリクス。すべて非LLM・決定的で、外部APIを必要としない。

外部judgeが使えないため、この層がすべての構成比較の主軸になる
（docs/design/design.md §6.3、§6.4）。
"""

from __future__ import annotations

from ragforlocalllm.eval.metrics.cost import CostAccumulator, percentile
from ragforlocalllm.eval.metrics.generation import (
    AnswerJudgment,
    CitationJudgment,
    Outcome,
    OutcomeRates,
    aggregate_outcomes,
    judge_answer,
    judge_citations,
)
from ragforlocalllm.eval.metrics.retrieval import (
    context_hit,
    context_precision,
    evidence_recall_at_k,
    hit_at_k,
    ndcg_at_k,
    reciprocal_rank,
)
from ragforlocalllm.eval.metrics.stats import (
    Interval,
    bootstrap_mean,
    bootstrap_paired_diff,
    is_significant,
    mean,
)

__all__ = [
    "AnswerJudgment",
    "CitationJudgment",
    "CostAccumulator",
    "Interval",
    "Outcome",
    "OutcomeRates",
    "aggregate_outcomes",
    "bootstrap_mean",
    "bootstrap_paired_diff",
    "context_hit",
    "context_precision",
    "evidence_recall_at_k",
    "hit_at_k",
    "is_significant",
    "judge_answer",
    "judge_citations",
    "mean",
    "ndcg_at_k",
    "percentile",
    "reciprocal_rank",
]

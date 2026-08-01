"""評価の実行。gold データセットをパイプラインに通し、ランレコードを残す。

集計で気をつけている点:

1. **検索メトリクスは「解決できた質問」だけで測る。** gold の引用が
   どのチャンクにも解決できなかった質問を recall 0 として混ぜると、
   検索性能の低さと Loader/Chunker の情報損失が区別できなくなる。
   解決できなかった件数は別途 ``resolution`` に出す
2. **正答/誤答/棄権を常に併記する。** 単一の精度指標では棄権の効果が
   見えない（docs/design/design.md §6.3）
3. **質問ごとの系列を残す。** 比較レポートで対応のあるブートストラップを
   使うために、集計値だけでなく質問単位の値が要る
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragforlocalllm.core.cache import Cache
from ragforlocalllm.core.config import ExperimentConfig
from ragforlocalllm.core.env import collect_env
from ragforlocalllm.core.indexing import build_index
from ragforlocalllm.core.pipeline import QueryPipeline
from ragforlocalllm.core.types import QueryState
from ragforlocalllm.eval.dataset import GoldDataset, GoldItem
from ragforlocalllm.eval.metrics import (
    AnswerJudgment,
    CitationJudgment,
    CostAccumulator,
    aggregate_outcomes,
    bootstrap_mean,
    context_hit,
    context_precision,
    evidence_recall_at_k,
    hit_at_k,
    judge_answer,
    judge_citations,
    mean,
    ndcg_at_k,
    reciprocal_rank,
)
from ragforlocalllm.eval.record import RUNS_ROOT, RunRecord, create_run
from ragforlocalllm.eval.resolve import GoldResolution, ResolutionReport, Resolver

DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass
class ItemResult:
    """gold 1件の評価結果。"""

    item: GoldItem
    state: QueryState
    judgment: AnswerJudgment
    citations: CitationJudgment
    resolution: GoldResolution | None
    retrieval: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        answer = self.state.answer
        return _clean(
            {
                "qid": self.item.qid,
                "question": self.item.question,
                "question_type": self.item.question_type,
                "answer_type": self.item.answer_type,
                "answerable": self.item.answerable,
                "tags": self.item.tags,
                "gold_answer": self.item.answer,
                "answer": None if answer is None else answer.text,
                "raw_answer": None if answer is None else answer.raw_text,
                "abstained": bool(answer and answer.abstained),
                "outcome": self.judgment.outcome,
                "exact_match": self.judgment.exact_match,
                "contains": self.judgment.contains,
                "char_f1": round(self.judgment.char_f1, 4),
                "set_f1": (
                    None if self.judgment.set_f1 is None else round(self.judgment.set_f1, 4)
                ),
                "needs_human_review": self.judgment.needs_human_review,
                "citations": [] if answer is None else answer.citations,
                "citation_all_exist": self.citations.all_exist,
                "citation_supported": self.citations.supported,
                "n_hallucinated_citations": self.citations.n_hallucinated,
                "gold_chunk_ids": (
                    sorted(self.resolution.chunk_ids) if self.resolution is not None else []
                ),
                "context_chunk_ids": (
                    [] if self.state.prompt is None else self.state.prompt.context_chunk_ids
                ),
                "retrieved": [
                    {"chunk_id": s.chunk.chunk_id, "score": round(s.score, 6)}
                    for s in self.state.retrieved
                ],
                "retrieval": {k: round(v, 4) for k, v in self.retrieval.items()},
                "latency_ms": round(self.state.total_duration_ms, 1),
                "trace": [t.model_dump() for t in self.state.trace],
            }
        )


@dataclass
class EvaluationResult:
    record: RunRecord
    metrics: dict[str, Any]
    results: list[ItemResult]


def run_evaluation(
    config: ExperimentConfig,
    gold: GoldDataset,
    *,
    cache: Cache | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    limit: int | None = None,
    root: Path | str = RUNS_ROOT,
    env_label: str | None = None,
    on_item: Callable[[int, GoldItem], None] | None = None,
) -> EvaluationResult:
    """gold 全件を評価し、``runs/`` にランレコードを書き出す。"""
    built = build_index(config, cache=cache)
    chunks = getattr(built.index, "chunks", None)
    if chunks is None:
        raise TypeError(
            f"{type(built.index).__name__} は chunks を公開していません。"
            "gold引用の解決にはチャンク本文が必要です。"
        )

    resolver = Resolver(chunks)
    report = resolver.resolve_dataset(gold)
    pipeline = QueryPipeline.from_config(config, embedder=built.embedder, index=built.index)

    items = list(gold)[: limit if limit is not None else None]
    results: list[ItemResult] = []
    cost = CostAccumulator()

    for i, item in enumerate(items, start=1):
        if on_item is not None:
            on_item(i, item)
        state = pipeline.run(item.question)
        cost.add(state)
        results.append(_evaluate_item(item, state, report, k_values))

    metrics = _aggregate(results, report, cost, k_values, gold)

    record = create_run(config.name, config.config_hash(), root=root)
    record.write_config(config.model_dump(mode="json"))
    record.write_env(_environment(config, built, env_label))
    record.write_predictions([r.as_row() for r in results])
    record.write_metrics(metrics)
    return EvaluationResult(record=record, metrics=metrics, results=results)


# ----------------------------------------------------------------------


def _evaluate_item(
    item: GoldItem,
    state: QueryState,
    report: ResolutionReport,
    k_values: Sequence[int],
) -> ItemResult:
    answer = state.answer
    resolution = report.resolutions.get(item.qid)
    gold_ids = resolution.chunk_ids if resolution is not None else frozenset()

    judgment = judge_answer(item, answer) if answer is not None else _no_answer_judgment()
    citation_judgment = (
        judge_citations(
            answer,
            [] if state.prompt is None else state.prompt.context_chunk_ids,
            gold_ids,
        )
        if answer is not None
        else CitationJudgment(False, False, False)
    )

    retrieval: dict[str, float] = {}
    # 回答不能な質問には根拠が無い。検索メトリクスの対象外。
    if resolution is not None and resolution.measurable:
        retrieved_ids = [s.chunk.chunk_id for s in state.retrieved]
        context_ids = [] if state.prompt is None else state.prompt.context_chunk_ids
        for k in k_values:
            retrieval[f"hit@{k}"] = hit_at_k(resolution, retrieved_ids, k)
            retrieval[f"recall@{k}"] = evidence_recall_at_k(resolution, retrieved_ids, k)
            retrieval[f"ndcg@{k}"] = ndcg_at_k(resolution, retrieved_ids, k)
        retrieval["mrr"] = reciprocal_rank(resolution, retrieved_ids)
        retrieval["context_precision"] = context_precision(resolution, context_ids)
        # 検索できた根拠がプロンプトまで生き残ったか。hit@k との差が予算の損失。
        retrieval["context_hit"] = context_hit(resolution, context_ids)

    return ItemResult(item, state, judgment, citation_judgment, resolution, retrieval)


def _no_answer_judgment() -> AnswerJudgment:
    """Generator が回答を返さなかった場合。

    棄権ではないので誤答として数える。棄権に寄せると、生成が落ちた構成の
    誤答率が不当に低く見える。
    """
    return AnswerJudgment("incorrect", exact_match=False, contains=False, char_f1=0.0)


def _aggregate(
    results: Sequence[ItemResult],
    report: ResolutionReport,
    cost: CostAccumulator,
    k_values: Sequence[int],
    gold: GoldDataset,
) -> dict[str, Any]:
    judgments = [r.judgment for r in results]
    rates = aggregate_outcomes(judgments)
    n_unanswerable = sum(1 for r in results if not r.item.answerable)

    correctness = [1.0 if j.correct else 0.0 for j in judgments]
    generation: dict[str, Any] = {
        **rates.as_dict(n_unanswerable=n_unanswerable),
        "accuracy_ci": bootstrap_mean(correctness).as_dict(),
        "exact_match": _mean_of(j.exact_match for j in judgments),
        "contains": _mean_of(j.contains for j in judgments),
        "char_f1": round(mean([j.char_f1 for j in judgments]), 4),
    }
    set_f1_values = [j.set_f1 for j in judgments if j.set_f1 is not None]
    if set_f1_values:
        generation["set_f1"] = round(mean(set_f1_values), 4)

    cited = [r.citations for r in results if r.citations.cited]
    citations: dict[str, Any] = {
        "cited_rate": _mean_of(r.citations.cited for r in results),
        # 引用したもののうち、すべて実在した割合。存在しないIDの引用は
        # コンテキストを見ずに形式だけ真似ている兆候であり、正誤とは
        # 別軸で追う必要がある。
        "all_exist_rate": round(mean([1.0 if c.all_exist else 0.0 for c in cited]), 4)
        if cited
        else None,
        "supported_rate": round(mean([1.0 if c.supported else 0.0 for c in cited]), 4)
        if cited
        else None,
        "hallucinated_total": sum(r.citations.n_hallucinated for r in results),
    }

    measured = [r for r in results if r.retrieval]
    retrieval: dict[str, Any] = {"n_measured": len(measured)}
    if measured:
        keys = [f"{prefix}@{k}" for k in k_values for prefix in ("hit", "recall", "ndcg")]
        for key in [*keys, "mrr", "context_hit", "context_precision"]:
            values = [r.retrieval[key] for r in measured if key in r.retrieval]
            retrieval[key] = round(mean([v for v in values if v == v]), 4) if values else None
        primary = f"hit@{max(k_values)}"
        retrieval[f"{primary}_ci"] = bootstrap_mean(
            [r.retrieval[primary] for r in measured]
        ).as_dict()

    return _clean(
        {
            "n_items": len(results),
            "n_answerable": len(results) - n_unanswerable,
            "n_unanswerable": n_unanswerable,
            "gold_summary": gold.summary(),
            "resolution": report.summary(),
            "generation": generation,
            "retrieval": retrieval,
            "citations": citations,
            "cost": cost.summary(),
            "by_question_type": _stratify(results, lambda r: r.item.question_type),
            "by_answer_type": _stratify(results, lambda r: r.item.answer_type),
        }
    )


def _stratify(
    results: Sequence[ItemResult], key: Callable[[ItemResult], str]
) -> dict[str, dict[str, Any]]:
    """層別集計。

    「どの質問タイプで効いたか」が手法選択の判断材料になる。
    層ごとの件数は少ないため信頼区間は出さず、**n を必ず併記して**
    小標本であることが分かるようにする。
    """
    groups: dict[str, list[ItemResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)

    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        rates = aggregate_outcomes([r.judgment for r in group])
        payload: dict[str, Any] = {
            "n": len(group),
            "accuracy": round(rates.accuracy, 4),
            "error_rate": round(rates.error_rate, 4),
            "abstention_rate": round(rates.abstention_rate, 4),
            "char_f1": round(mean([r.judgment.char_f1 for r in group]), 4),
        }
        measured = [r for r in group if r.retrieval]
        if measured:
            hits = [v for r in measured for k, v in r.retrieval.items() if k.startswith("hit@")]
            if hits:
                payload["hit_mean"] = round(mean(hits), 4)
        out[name] = payload
    return out


def _environment(config: ExperimentConfig, built: Any, label: str | None) -> dict[str, Any]:
    env = collect_env(label=label, corpus_path=Path(config.corpus))
    env["index"] = {
        "signature": built.signature,
        "directory": str(built.directory),
        **{k: v for k, v in built.stats.items() if k != "corpus_sha256"},
    }
    env["config_hash"] = config.config_hash()
    # 生成器の実体（LM Studio 側のモデルID等）。同じ設定名でも
    # サーバ側で別のモデルがロードされていれば別の実験になる。
    from ragforlocalllm.core import registry  # 循環 import を避けるため遅延

    generator = registry.build("generator", config.query.generator.as_spec())
    describe = getattr(generator, "describe", None)
    if callable(describe):
        env["generator"] = describe()
    return env


def _mean_of(flags: Any) -> float:
    values = [1.0 if f else 0.0 for f in flags]
    return round(mean(values), 4) if values else 0.0


def _clean(payload: dict[str, Any]) -> dict[str, Any]:
    """NaN を None に置き換えた dict を返す。

    ``float('nan')`` は JSON として不正であり、そのまま書くと
    ``metrics.json`` を他のツールから読めなくなる。
    """
    return {key: _clean_value(value) for key, value in payload.items()}


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value

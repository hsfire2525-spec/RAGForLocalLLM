"""複数ランの比較。**信頼区間を必ず併記する。**

質問数が30〜50件の規模では、正答率の数ポイントの差はほとんどの場合
ノイズである。点推定だけを並べた表は「効いた/効かない」を誤読させる
（docs/design/design.md §6.5）。

比較の型は2つある。

- **各ランの絶対値** … 平均のブートストラップ区間を付ける
- **ベースラインとの差** … **対応のある**ブートストラップを使う。
  同じ gold を使っているので質問ごとの難易度差を相殺でき、
  独立標本として扱うより狭い区間で判定できる

差の区間が0を跨いでいれば「差があるとは言えない」と明示する。
これを出さないと、実験ログに improvements が積み上がっていく。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragforlocalllm.eval.metrics.stats import (
    Interval,
    bootstrap_mean,
    bootstrap_paired_diff,
    is_significant,
)
from ragforlocalllm.eval.record import RunRecord

CORRECT_OUTCOMES = frozenset({"correct", "correct_abstention"})

_LOWER_IS_BETTER = frozenset({"error_rate", "latency_ms"})
_NEUTRAL = frozenset({"abstention_rate"})


def polarity(metric: str) -> int:
    """指標の向き。1=大きいほど良い、-1=小さいほど良い、0=文脈依存。

    **棄権率を「小さいほど良い」と決めつけてはいけない。** 棄権は
    誤答を避けるための機能であり、回答不能な質問が多い gold では
    高いほうが正しい。逆に取りこぼしが多ければ下げたい。
    どちらかは gold の構成と目的次第なので、方向だけ示して
    良し悪しの判断は人に委ねる。
    """
    if metric in _NEUTRAL:
        return 0
    if metric in _LOWER_IS_BETTER:
        return -1
    return 1


@dataclass
class RunSeries:
    """1ランの質問単位の系列。対応のある比較に使う。"""

    record: RunRecord
    by_qid: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def short_name(self) -> str:
        """日時プレフィックスを落とした表示用の名前（``smoke-dd11c93611bb``）。

        比較表は指標 × ラン数だけ行が増えるので、フルネームだと横幅に
        収まらず、肝心の信頼区間が省略されてしまう。
        """
        parts = self.record.name.split("-", 2)
        return parts[2] if len(parts) == 3 else self.record.name

    @property
    def qids(self) -> set[str]:
        return set(self.by_qid)

    def series(self, metric: str, qids: Sequence[str]) -> list[float]:
        return [_metric_value(self.by_qid.get(qid), metric) for qid in qids]


def load_series(record: RunRecord) -> RunSeries:
    rows = record.read_predictions()
    return RunSeries(record=record, by_qid={row["qid"]: row for row in rows if "qid" in row})


def _metric_value(row: dict[str, Any] | None, metric: str) -> float:
    if row is None:
        return float("nan")
    if metric == "accuracy":
        return 1.0 if row.get("outcome") in CORRECT_OUTCOMES else 0.0
    if metric == "error_rate":
        return 1.0 if row.get("outcome") == "incorrect" else 0.0
    if metric == "abstention_rate":
        return 1.0 if row.get("abstained") else 0.0
    if metric == "char_f1":
        value = row.get("char_f1")
        return float(value) if value is not None else float("nan")
    if metric == "latency_ms":
        value = row.get("latency_ms")
        return float(value) if value is not None else float("nan")
    retrieval = row.get("retrieval") or {}
    value = retrieval.get(metric)
    return float(value) if value is not None else float("nan")


@dataclass(frozen=True)
class MetricRow:
    """1ラン・1指標の比較行。"""

    run: str
    metric: str
    interval: Interval
    diff: Interval | None = None
    """ベースラインとの差。ベースライン自身では None。"""

    @property
    def significant(self) -> bool:
        return self.diff is not None and is_significant(self.diff)

    @property
    def verdict(self) -> str:
        if self.diff is None:
            return "基準"
        if not self.significant:
            return "有意差なし"
        direction = polarity(self.metric)
        if direction == 0:
            # 良し悪しが文脈依存の指標。方向だけ示して判断は人に委ねる。
            return "増加" if self.diff.point > 0 else "減少"
        improved = (self.diff.point > 0) == (direction > 0)
        return "改善" if improved else "悪化"


DEFAULT_METRICS = ("accuracy", "error_rate", "abstention_rate", "char_f1")


def compare(
    records: Sequence[RunRecord],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    baseline: int = 0,
) -> list[MetricRow]:
    """先頭（または ``baseline`` 番目）のランを基準に比較する。

    共通する qid だけで比較する。gold を更新した前後のランを並べたとき、
    質問集合が違うまま比べると差が意味を持たなくなるため。
    """
    if not records:
        return []
    series = [load_series(r) for r in records]
    shared = set.intersection(*(s.qids for s in series)) if series else set()
    qids = sorted(shared)
    if not qids:
        raise ValueError("比較対象のランに共通する qid がありません")

    base = series[baseline]
    rows: list[MetricRow] = []
    for metric in metrics:
        base_values = base.series(metric, qids)
        for i, run in enumerate(series):
            values = run.series(metric, qids)
            rows.append(
                MetricRow(
                    run=run.short_name,
                    metric=metric,
                    interval=bootstrap_mean(values),
                    diff=None if i == baseline else bootstrap_paired_diff(values, base_values),
                )
            )
    return rows


def shared_qid_count(records: Sequence[RunRecord]) -> int:
    if not records:
        return 0
    series = [load_series(r) for r in records]
    return len(set.intersection(*(s.qids for s in series)))

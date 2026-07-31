"""信頼区間と有意差判定。

**質問数が30〜50件しかない。** この規模では数ポイントの差はほぼ確実に
ノイズであり、区間を併記しないと「効いた」と誤読する。比較レポートでは
常に信頼区間を出す（docs/design/design.md §6.5）。

外れ値のある小標本でも仮定を置かずに済むブートストラップを使う。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20260731


@dataclass(frozen=True)
class Interval:
    """点推定と信頼区間。"""

    point: float
    low: float
    high: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def as_dict(self) -> dict[str, float]:
        return {
            "point": round(self.point, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
            "ci_level": self.level,
        }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_mean(
    values: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """平均のブートストラップ信頼区間（パーセンタイル法）。

    ``seed`` を固定しているのは、同じ入力から同じ区間が出ないと
    実験ログの再現性が崩れるため。
    """
    clean = [v for v in values if v == v]  # NaN を除く
    if not clean:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    point = mean(clean)
    if len(clean) == 1:
        return Interval(point, point, point, level)

    rng = random.Random(seed)
    n = len(clean)
    means = []
    for _ in range(resamples):
        sample = [clean[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - level) / 2.0
    low = means[min(int(alpha * resamples), resamples - 1)]
    high = means[min(int((1.0 - alpha) * resamples), resamples - 1)]
    return Interval(point, low, high, level)


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """同一質問集合での差 (a - b) の信頼区間。

    **対応のあるブートストラップを使う。** 2つの構成を同じ gold で
    評価しているため、質問ごとの難易度差を相殺できる。独立標本として
    扱うと区間が無用に広がり、実在する差を見逃す。
    """
    if len(a) != len(b):
        raise ValueError("対応のある比較には同じ長さの系列が必要です")
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x == x and y == y]
    if not pairs:
        return Interval(float("nan"), float("nan"), float("nan"), level)

    diffs = [x - y for x, y in pairs]
    return bootstrap_mean(diffs, level=level, resamples=resamples, seed=seed)


def is_significant(interval: Interval) -> bool:
    """差の信頼区間が0を跨がないか。"""
    if interval.low != interval.low:  # NaN
        return False
    return interval.low > 0.0 or interval.high < 0.0

"""信頼区間と有意差判定。

**質問数が30〜50件しかない。** この規模では数ポイントの差はほぼ確実に
ノイズであり、区間を併記しないと「効いた」と誤読する。比較レポートでは
常に信頼区間を出す（docs/design/design.md §6.5）。

外れ値のある小標本でも仮定を置かずに済むブートストラップを使う。
"""

from __future__ import annotations

import math
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


def _z_for(level: float) -> float:
    """よく使う信頼水準の正規分位点。scipy を持ち込むほどではない。"""
    return {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(round(level, 2), 1.9600)


def wilson_interval(successes: int, n: int, *, level: float = 0.95) -> Interval:
    """二値データの Wilson スコア信頼区間。

    **全問正解のときブートストラップは壊れる。** 標本に分散が無いため
    どのリサンプルも同じ平均になり、区間が [1.0, 1.0] という
    「絶対に 1.0」という主張になってしまう。42問中42問正解でも、
    真の正答率が 0.92 である可能性は十分にある。

    Wilson 区間は境界（0 や 1）でも縮退せず、n が小さいときの
    非対称性も正しく扱う。二値の指標ではこちらを使う。
    """
    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    z = _z_for(level)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, center - half), min(1.0, center + half), level)


def bootstrap_mean(
    values: Sequence[float],
    *,
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    proportion: bool | None = None,
) -> Interval:
    """平均の信頼区間。

    二値（0/1）の系列は Wilson 区間、それ以外はブートストラップ。
    正答率や hit@k は二値であり、**全問正解・全問不正解のときに
    ブートストラップが縮退する**ため、そこだけ別扱いにしている。

    ``proportion`` を明示すると自動判定を上書きできる。**差の系列には
    使ってはいけない。** 差が偶然すべて 0 でも「割合 0」ではなく
    「差が無い」であり、Wilson の非対称な区間は意味を持たない。

    ``seed`` を固定しているのは、同じ入力から同じ区間が出ないと
    実験ログの再現性が崩れるため。
    """
    clean = [v for v in values if v == v]  # NaN を除く
    if not clean:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    point = mean(clean)
    if len(clean) == 1:
        return Interval(point, point, point, level)
    is_proportion = proportion if proportion is not None else all(v in (0.0, 1.0) for v in clean)
    if is_proportion:
        return wilson_interval(int(sum(clean)), len(clean), level=level)

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
    # 差は割合ではない。全件が 0 でも Wilson を使ってはいけない。
    return bootstrap_mean(diffs, level=level, resamples=resamples, seed=seed, proportion=False)


def is_significant(interval: Interval) -> bool:
    """差の信頼区間が0を跨がないか。"""
    if interval.low != interval.low:  # NaN
        return False
    return interval.low > 0.0 or interval.high < 0.0

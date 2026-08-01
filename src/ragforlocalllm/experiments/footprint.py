"""コンポーネントごとのメモリ実測。

**逐次ロード・オフロード機構は先に作らない。** 「LLMのみ iGPU、
リランカー・埋め込み・NLI はCPU」の配分で環境1（8GB共有）が成立する
見込みが高く、実測前に機構を作るのは投機的な複雑化になる
（docs/design/design.md §10.5）。

代わりにこれを測って判断材料にする。各コンポーネントを順に構築し、
都度 RSS の増分とピークを記録する。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

from ragforlocalllm.core import registry
from ragforlocalllm.core.config import ExperimentConfig
from ragforlocalllm.core.env import detect_gpu, rss_mb


@dataclass(frozen=True)
class ComponentFootprint:
    kind: str
    impl: str
    rss_before_mb: float | None
    rss_after_build_mb: float | None
    rss_after_warmup_mb: float | None
    built: bool
    warmed: bool = False
    error: str | None = None

    @property
    def build_delta_mb(self) -> float | None:
        if self.rss_before_mb is None or self.rss_after_build_mb is None:
            return None
        return round(self.rss_after_build_mb - self.rss_before_mb, 1)

    @property
    def delta_mb(self) -> float | None:
        """構築からウォームアップ完了までの総増分。**これが常駐量。**"""
        end = self.rss_after_warmup_mb or self.rss_after_build_mb
        if self.rss_before_mb is None or end is None:
            return None
        return round(end - self.rss_before_mb, 1)


def measure(config: ExperimentConfig) -> dict[str, Any]:
    """設定に現れるコンポーネントを順に構築し、RSS の増分を測る。

    **構築しただけでは測れない。** 埋め込み器やリランカーはモデルを
    遅延読み込みするため（グローバルなシングルトンを持たない方針の帰結）、
    構築直後の RSS はほぼゼロになる。実際に使わせて読み込ませてから測る。

    Retriever は埋め込み器とインデックスの注入が要るため対象外。
    ここで見たいのはモデルを持つコンポーネントの常駐量であり、
    それらはすべて単体で構築できる。
    """
    specs: list[tuple[str, dict[str, Any]]] = [
        ("embedder", config.index.embedder.as_spec()),
        ("generator", config.query.generator.as_spec()),
        *(("post_retrieval", spec.as_spec()) for spec in config.query.post_retrieval),
        *(("post_generation", spec.as_spec()) for spec in config.query.post_generation),
    ]

    measurements: list[ComponentFootprint] = []
    held: list[Any] = []  # 参照を保持し、同時常駐時の総量を測る
    for kind, spec in specs:
        gc.collect()
        before = rss_mb()
        impl = str(spec.get("type"))
        try:
            component = registry.build(kind, spec)
        except Exception as exc:
            measurements.append(
                ComponentFootprint(kind, impl, before, None, None, False, error=str(exc))
            )
            continue
        held.append(component)
        after_build = rss_mb()
        warmed = _warmup(component)
        gc.collect()
        measurements.append(
            ComponentFootprint(kind, impl, before, after_build, rss_mb(), True, warmed=warmed)
        )

    return _report(measurements)


def _warmup(component: Any) -> bool:
    """モデルの遅延読み込みを強制する。

    LM Studio 経由の生成器のようにモデルを別プロセスに持つものは
    ウォームアップ対象にしない（ネットワーク越しの呼び出しになり、
    RSS にも現れないため）。
    """
    warmup = getattr(component, "warmup", None)
    if callable(warmup):
        warmup()
        return True
    embed = getattr(component, "embed_queries", None)
    if callable(embed):
        embed(["ウォームアップ"])
        return True
    # dim の参照だけでモデルを読み込む実装もある
    if hasattr(type(component), "dim"):
        _ = component.dim
        return True
    return False


def _report(measurements: list[ComponentFootprint]) -> dict[str, Any]:
    return {
        "components": [
            {
                "kind": m.kind,
                "impl": m.impl,
                "delta_mb": m.delta_mb,
                "build_delta_mb": m.build_delta_mb,
                "rss_after_mb": m.rss_after_warmup_mb or m.rss_after_build_mb,
                "built": m.built,
                "warmed": m.warmed,
                "error": m.error,
            }
            for m in measurements
        ],
        "resident_total_rss_mb": rss_mb(),
        "gpu": detect_gpu(),
        "note": (
            "増分はモデル読み込み後（ウォームアップ済み）の常駐量。"
            "LM Studio 側のモデルは別プロセスなので含まれない"
            "（生成器の実体は `rag env -c <設定>` で確認する）。"
        ),
    }

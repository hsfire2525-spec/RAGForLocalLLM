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
    rss_after_mb: float | None
    built: bool
    error: str | None = None

    @property
    def delta_mb(self) -> float | None:
        if self.rss_before_mb is None or self.rss_after_mb is None:
            return None
        return round(self.rss_after_mb - self.rss_before_mb, 1)


def measure(config: ExperimentConfig) -> dict[str, Any]:
    """設定に現れるコンポーネントを順に構築し、RSS の増分を測る。

    Retriever は埋め込み器とインデックスの注入が要るため対象外。
    ここで見たいのは**モデルを持つコンポーネント**の常駐量であり、
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
        try:
            component = registry.build(kind, spec)
        except Exception as exc:
            measurements.append(
                ComponentFootprint(kind, str(spec.get("type")), before, None, False, str(exc))
            )
            continue
        held.append(component)
        measurements.append(ComponentFootprint(kind, str(spec.get("type")), before, rss_mb(), True))

    return {
        "components": [
            {
                "kind": m.kind,
                "impl": m.impl,
                "delta_mb": m.delta_mb,
                "rss_after_mb": m.rss_after_mb,
                "built": m.built,
                "error": m.error,
            }
            for m in measurements
        ],
        "resident_total_rss_mb": rss_mb(),
        "gpu": detect_gpu(),
        "note": (
            "RSS はプロセス全体の常駐量。LM Studio 側のモデルは別プロセスなので"
            "含まれない（生成器の実体は `rag env -c <設定>` で確認する）。"
        ),
    }

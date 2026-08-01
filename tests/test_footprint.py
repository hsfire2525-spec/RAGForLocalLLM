"""コンポーネント別メモリ実測の検証。

**構築しただけでは測れない。** 埋め込み器やリランカーはモデルを遅延
読み込みするため（グローバルなシングルトンを持たない方針の帰結）、
構築直後の RSS はほぼゼロになる。実際に `rag footprint -c baseline` を
動かして初めて露見した欠陥なので、回帰として固定する。
"""

from __future__ import annotations

from pathlib import Path

from ragforlocalllm.core.config import load_config
from ragforlocalllm.experiments.footprint import ComponentFootprint, _warmup, measure


class LazyEmbedder:
    """モデルを遅延読み込みする埋め込み器の模擬。"""

    def __init__(self) -> None:
        self.loaded = False

    @property
    def dim(self) -> int:
        self.loaded = True
        return 768

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.loaded = True
        return [[0.0] * 768 for _ in texts]


class ExplicitWarmup:
    def __init__(self) -> None:
        self.loaded = False

    def warmup(self) -> None:
        self.loaded = True


class Trivial:
    """モデルを持たない段（並べ替え・重複除去など）。"""


# ----------------------------------------------------------------------


def test_warmup_prefers_an_explicit_hook() -> None:
    component = ExplicitWarmup()
    assert _warmup(component)
    assert component.loaded


def test_warmup_forces_lazy_model_load() -> None:
    """**これが無いと埋め込み器の常駐量が 0MB と表示される。**"""
    component = LazyEmbedder()
    assert _warmup(component)
    assert component.loaded


def test_warmup_reports_false_for_components_without_models() -> None:
    assert not _warmup(Trivial())


# ----------------------------------------------------------------------


def test_delta_covers_build_and_warmup() -> None:
    """常駐量は「構築 + モデル読み込み」の合計で見る。"""
    footprint = ComponentFootprint(
        kind="embedder",
        impl="sentence_transformers",
        rss_before_mb=50.0,
        rss_after_build_mb=53.0,
        rss_after_warmup_mb=1513.0,
        built=True,
        warmed=True,
    )
    assert footprint.build_delta_mb == 3.0
    assert footprint.delta_mb == 1463.0


def test_delta_falls_back_to_build_when_not_warmed() -> None:
    footprint = ComponentFootprint(
        kind="post_retrieval",
        impl="dedupe",
        rss_before_mb=50.0,
        rss_after_build_mb=50.0,
        rss_after_warmup_mb=None,
        built=True,
    )
    assert footprint.delta_mb == 0.0


def test_failed_component_has_no_delta() -> None:
    footprint = ComponentFootprint(
        kind="embedder",
        impl="missing",
        rss_before_mb=50.0,
        rss_after_build_mb=None,
        rss_after_warmup_mb=None,
        built=False,
        error="boom",
    )
    assert footprint.delta_mb is None
    assert footprint.build_delta_mb is None


def test_measure_runs_over_a_config_without_extra_deps() -> None:
    """追加依存の要らない設定で一通り動くこと。"""
    import ragforlocalllm.stages  # noqa: F401 - レジストリ登録

    config = load_config(Path("configs/smoke.yaml"), search_dir=Path("configs"))
    result = measure(config)

    kinds = [c["kind"] for c in result["components"]]
    assert "embedder" in kinds
    assert all(c["built"] for c in result["components"])
    assert result["resident_total_rss_mb"] > 0
    # 埋め込み器はウォームアップされる（モデルを持ちうる段のため）
    embedder = next(c for c in result["components"] if c["kind"] == "embedder")
    assert embedder["warmed"]

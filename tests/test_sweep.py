"""スイープの検証。

**手回しのシェルループを仕組みにしたもの。** 予算の調整で5点比較したときの
問題（設定が残らない・失敗に気付けない・再インデックスが読めない）を
潰すのが目的なので、そこを中心に固める。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragforlocalllm.core.config import ConfigError
from ragforlocalllm.experiments.sweep import (
    SweepConfig,
    describe_plan,
    expand,
    load_sweep,
    set_path,
    validate,
)

CONFIGS = Path("configs")


def sweep(**kwargs: object) -> SweepConfig:
    payload: dict[str, object] = {"base": "hybrid_budget"}
    payload.update(kwargs)
    return SweepConfig.model_validate(payload)


# ----------------------------------------------------------------------
# パスへの差し込み
# ----------------------------------------------------------------------


def test_set_path_updates_a_nested_value() -> None:
    payload = {"query": {"prompt": {"context_token_budget": 1536}}}
    set_path(payload, "query.prompt.context_token_budget", 2560)
    assert payload["query"]["prompt"]["context_token_budget"] == 2560


def test_missing_path_is_an_error() -> None:
    """**黙って作らない。**

    綴りを間違えた軸が「新しいキー」として通ると、何も変えていないのに
    変えたつもりの比較をすることになる。
    """
    with pytest.raises(ConfigError, match="存在しません"):
        set_path({"query": {}}, "query.prompt.budget", 1)
    with pytest.raises(ConfigError, match="存在しません"):
        set_path({"query": {"prompt": {}}}, "query.nonexistent.budget", 1)


def test_changing_type_drops_the_previous_arguments() -> None:
    """**実装を差し替えたら前の実装の引数は捨てる。**

    {type: hybrid, rrf_k: 60, retrievers: [...]} の type を dense にすると、
    rrf_k と retrievers は dense の引数ではないためレジストリが拒否する。
    extends の差分継承と同じ扱いにしている。
    """
    payload = {"retriever": {"type": "hybrid", "top_k": 5, "rrf_k": 60, "retrievers": [{}]}}
    set_path(payload, "retriever.type", "dense")
    assert payload["retriever"] == {"type": "dense"}


def test_setting_the_same_type_keeps_the_arguments() -> None:
    payload = {"retriever": {"type": "hybrid", "rrf_k": 60}}
    set_path(payload, "retriever.type", "hybrid")
    assert payload["retriever"] == {"type": "hybrid", "rrf_k": 60}


# ----------------------------------------------------------------------
# 変種の展開
# ----------------------------------------------------------------------


def test_ablation_changes_one_axis_at_a_time() -> None:
    variants = expand(
        sweep(axes={"query.retriever.rrf_k": [10, 120], "query.retriever.top_k": [3]}),
        config_dir=CONFIGS,
    )
    assert variants[0].overrides == {}  # 先頭は基準
    assert all(len(v.overrides) <= 1 for v in variants)
    assert len(variants) == 4


def test_grid_takes_the_product() -> None:
    variants = expand(
        sweep(
            mode="grid",
            axes={"query.retriever.rrf_k": [10, 120], "query.retriever.top_k": [3, 8]},
        ),
        config_dir=CONFIGS,
    )
    # 基準 + 4通り。ただし基準と同一になる組合せは畳まれる。
    assert len(variants) == 5
    assert any(len(v.overrides) == 2 for v in variants)


def test_variants_identical_to_the_base_are_dropped() -> None:
    """**基準値を軸に並べても重複したランを走らせない。**

    見通しのため各軸に基準値を含めるのは自然だが、素朴に展開すると
    基準と同じランが軸の数だけ増える。実測では10件中3件が重複だった。
    """
    base_only = expand(sweep(axes={}), config_dir=CONFIGS)
    assert len(base_only) == 1

    # rrf_k=60 は hybrid_budget の基準値なので畳まれる
    variants = expand(sweep(axes={"query.retriever.rrf_k": [60, 120]}), config_dir=CONFIGS)
    assert [v.overrides for v in variants] == [{}, {"query.retriever.rrf_k": 120}]


def test_every_variant_is_a_valid_config() -> None:
    """**実行前に全部検証する。**

    30分のスイープの28分目で設定エラーが出るのが最悪。
    """
    variants = expand(sweep(axes={"query.retriever.type": ["dense", "sparse"]}), config_dir=CONFIGS)
    for variant in variants:
        config = validate(variant)
        assert config.name == variant.name


def test_variant_names_are_distinct_and_readable() -> None:
    variants = expand(
        sweep(axes={"query.prompt.context_token_budget": [1024, 4096]}), config_dir=CONFIGS
    )
    names = [v.name for v in variants]
    assert len(set(names)) == len(names)
    assert any("context_token_budget-1024" in n for n in names)


def test_label_summarises_the_difference_from_the_base() -> None:
    variants = expand(sweep(axes={"query.retriever.rrf_k": [10]}), config_dir=CONFIGS)
    assert variants[0].label == "(基準)"
    assert variants[1].label == "rrf_k=10"


def test_unknown_axis_fails_at_expansion() -> None:
    with pytest.raises(ConfigError, match="存在しません"):
        expand(sweep(axes={"query.retriever.typo_k": [1]}), config_dir=CONFIGS)


# ----------------------------------------------------------------------
# 計画の提示
# ----------------------------------------------------------------------


def test_plan_flags_axes_that_force_reindexing() -> None:
    """index 側を触る軸は変種ごとに再埋め込みが走る。先に知らせる。"""
    config = sweep(axes={"index.chunker.chunk_size": [256, 512]})
    plan = describe_plan(config, expand(config, config_dir=CONFIGS))
    assert plan["reindex_axes"] == ["index.chunker.chunk_size"]


def test_plan_has_no_reindex_axes_for_query_only_sweeps() -> None:
    config = sweep(axes={"query.retriever.rrf_k": [10]})
    assert describe_plan(config, expand(config, config_dir=CONFIGS))["reindex_axes"] == []


# ----------------------------------------------------------------------
# 読み込み
# ----------------------------------------------------------------------


def test_shipped_sweeps_load_and_expand() -> None:
    for path in sorted(Path("configs/sweeps").glob("*.yaml")):
        config = load_sweep(path)
        variants = expand(config, config_dir=CONFIGS)
        assert len(variants) > 1, path
        for variant in variants:
            validate(variant)


def test_missing_sweep_file_is_reported() -> None:
    with pytest.raises(ConfigError, match="ありません"):
        load_sweep("configs/sweeps/nope.yaml")


def test_unknown_key_in_sweep_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("base: hybrid_budget\nmodes: ablation\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sweep(path)

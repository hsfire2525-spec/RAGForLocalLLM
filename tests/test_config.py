from __future__ import annotations

from pathlib import Path

import pytest

from ragforlocalllm.core.config import ConfigError, deep_merge, load_config, load_raw_config


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


BASE = """
name: base
corpus: data/corpus/sample
index:
  loader: {type: text}
  chunker: {type: fixed, chunk_size: 300}
  embedder: {type: hashing, dim: 64}
  indexer: {type: numpy_flat}
query:
  retriever: {type: dense, top_k: 5}
  post_retrieval:
    - {type: dedupe}
  prompt: {type: template, path: prompts/basic_ja.jinja}
  generator: {type: extractive}
"""


def test_deep_merge_replaces_lists_and_merges_dicts() -> None:
    base = {"a": {"x": 1, "y": 2}, "items": [1, 2, 3]}
    override = {"a": {"y": 9}, "items": [7]}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 9}, "items": [7]}


def test_deep_merge_replaces_component_when_type_changes() -> None:
    """実装が変われば親の引数は別の実装のものであり、引き継いではならない。"""
    base = {"generator": {"type": "extractive", "max_sentences": 2}}
    override = {"generator": {"type": "openai_compat", "model": "m"}}
    merged = deep_merge(base, override)
    assert merged["generator"] == {"type": "openai_compat", "model": "m"}
    assert "max_sentences" not in merged["generator"]


def test_deep_merge_keeps_merging_when_type_is_unchanged() -> None:
    base = {"generator": {"type": "openai_compat", "model": "m", "seed": 42}}
    override = {"generator": {"type": "openai_compat", "temperature": 0.5}}
    merged = deep_merge(base, override)
    assert merged["generator"] == {
        "type": "openai_compat",
        "model": "m",
        "seed": 42,
        "temperature": 0.5,
    }


def test_extends_switching_component_type_drops_stale_args(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", BASE)
    write(
        tmp_path / "child.yaml",
        """
extends: base
name: child
query:
  generator: {type: openai_compat, model: some-model}
""",
    )
    cfg = load_config(tmp_path / "child.yaml", search_dir=tmp_path)
    extra = cfg.query.generator.model_extra or {}
    assert cfg.query.generator.type == "openai_compat"
    assert extra == {"model": "some-model"}


def test_extends_merges_and_replaces_stage_list(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", BASE)
    write(
        tmp_path / "child.yaml",
        """
extends: base
name: child
query:
  retriever: {type: dense, top_k: 10}
  post_retrieval:
    - {type: top_k, k: 3}
    - {type: reorder_lost_in_middle}
""",
    )
    cfg = load_config(tmp_path / "child.yaml", search_dir=tmp_path)

    assert cfg.name == "child"
    assert cfg.query.retriever.model_extra is not None
    assert cfg.query.retriever.model_extra["top_k"] == 10
    # 段のリストは置換される（追記ではない）。順序自体が実験軸のため。
    assert [s.type for s in cfg.query.post_retrieval] == ["top_k", "reorder_lost_in_middle"]
    # 継承元の index はそのまま
    assert cfg.index.chunker.type == "fixed"


def test_extends_cycle_is_detected(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", "extends: b\nname: a\n")
    write(tmp_path / "b.yaml", "extends: a\nname: b\n")
    with pytest.raises(ConfigError, match="循環"):
        load_raw_config(tmp_path / "a.yaml", search_dir=tmp_path)


def test_missing_file_reports_searched_paths(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="見つかりません"):
        load_raw_config("nope.yaml", search_dir=tmp_path)


def test_index_signature_ignores_query_changes(tmp_path: Path) -> None:
    """クエリ側だけ変えた実験でインデックスが再利用できること。"""
    write(tmp_path / "base.yaml", BASE)
    write(
        tmp_path / "child.yaml",
        "extends: base\nname: child\nquery:\n  retriever: {type: dense, top_k: 20}\n",
    )
    a = load_config(tmp_path / "base.yaml", search_dir=tmp_path)
    b = load_config(tmp_path / "child.yaml", search_dir=tmp_path)

    assert a.index_signature("abc") == b.index_signature("abc")
    assert a.config_hash() != b.config_hash()


def test_config_hash_ignores_name_only(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", BASE)
    write(tmp_path / "renamed.yaml", "extends: base\nname: renamed\n")
    a = load_config(tmp_path / "base.yaml", search_dir=tmp_path)
    b = load_config(tmp_path / "renamed.yaml", search_dir=tmp_path)
    assert a.config_hash() == b.config_hash()


def test_index_signature_changes_with_corpus() -> None:
    cfg = load_config(Path("configs/smoke.yaml"), search_dir=Path("configs"))
    assert cfg.index_signature("aaa") != cfg.index_signature("bbb")


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    write(tmp_path / "bad.yaml", BASE + "\nunknown_section: 1\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "bad.yaml", search_dir=tmp_path)

"""実験設定の読み込みと同一性ハッシュ。

1実験 = YAML 1枚。``extends`` による差分継承を持つ（設定の重複は
そのまま実験の信頼性を損なうため、初期から入れる）。
設計方針は docs/design/design.md §4.4 を参照。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConfigError(Exception):
    """設定の読み込み・検証に失敗した。"""


class ComponentSpec(BaseModel):
    """``{"type": <登録名>, **引数}`` 形式のコンポーネント指定。

    引数はコンポーネントごとに異なるため extra を許可し、
    実際の検証は registry.build がコンストラクタのシグネチャに対して行う。
    """

    model_config = ConfigDict(extra="allow")

    type: str

    def as_spec(self) -> dict[str, Any]:
        return self.model_dump()


class PromptSpec(ComponentSpec):
    context_token_budget: int | None = None
    """コンテキストに割り当てる上限トークン数。

    4B級・4bit量子化モデルでは名目コンテキスト長に達する前に品質が
    劣化する。top_k の件数ではなくトークン予算で制御するほうが、
    チャンクサイズを変える実験と整合する。
    """
    overflow_policy: Literal["drop_lowest", "truncate_each"] = "drop_lowest"


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loader: ComponentSpec
    chunker: ComponentSpec
    embedder: ComponentSpec
    indexer: ComponentSpec


class QueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_transform: ComponentSpec = Field(default_factory=lambda: ComponentSpec(type="identity"))
    retriever: ComponentSpec
    post_retrieval: list[ComponentSpec] = Field(default_factory=list)
    prompt: PromptSpec
    generator: ComponentSpec
    post_generation: list[ComponentSpec] = Field(default_factory=list)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: Path | None = None
    metrics: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    corpus: Path
    """コーパスのパス。リポジトリにはコミットされない（data/corpus/README.md 参照）。"""
    index: IndexConfig
    query: QueryConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)

    # ------------------------------------------------------------------
    # 同一性ハッシュ
    # ------------------------------------------------------------------

    def index_signature(self, corpus_sha256: str) -> str:
        """インデックス成果物の同一性。

        クエリ側だけを変える実験でインデックスを再利用できるよう、
        index セクションとコーパスのハッシュのみから決める。
        """
        return _hash({"index": self.index.model_dump(mode="json"), "corpus": corpus_sha256})

    def config_hash(self) -> str:
        """設定全体の同一性。同一設定の再実行検出に使う。

        ``name`` は実験のラベルであり挙動に影響しないため除外する。
        """
        payload = self.model_dump(mode="json")
        payload.pop("name", None)
        return _hash(payload)


def _hash(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------
# 読み込みと extends 解決
# ----------------------------------------------------------------------


def load_config(path: str | Path, *, search_dir: Path | None = None) -> ExperimentConfig:
    """YAML を読み込み、``extends`` を解決して検証済みの設定を返す。"""
    raw = load_raw_config(path, search_dir=search_dir)
    try:
        return ExperimentConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{path}: 設定の検証に失敗しました\n{exc}") from exc


def load_raw_config(
    path: str | Path,
    *,
    search_dir: Path | None = None,
    _seen: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """``extends`` を解決した生の dict を返す（検証前）。"""
    resolved = _resolve_path(path, search_dir)
    if resolved in _seen:
        chain = " -> ".join(p.name for p in (*_seen, resolved))
        raise ConfigError(f"extends が循環しています: {chain}")

    with resolved.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{resolved}: トップレベルはマッピングである必要があります")

    parent_ref = data.pop("extends", None)
    if parent_ref is None:
        return data

    parent = load_raw_config(
        parent_ref,
        search_dir=search_dir or resolved.parent,
        _seen=(*_seen, resolved),
    )
    return deep_merge(parent, data)


def _resolve_path(path: str | Path, search_dir: Path | None) -> Path:
    candidate = Path(path)
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".yaml")
    if candidate.is_absolute() and candidate.exists():
        return candidate

    roots = [Path.cwd()]
    if search_dir is not None:
        roots.insert(0, search_dir)
    for root in roots:
        found = root / candidate
        if found.exists():
            return found.resolve()

    tried = ", ".join(str(r / candidate) for r in roots)
    raise ConfigError(f"設定ファイルが見つかりません: {path}（探索: {tried}）")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """dict は再帰的にマージし、それ以外（リストを含む）は置換する。

    ただし2つの例外がある。

    1. **リストは置換**。``post_retrieval`` のような段のリストは順序
       自体が実験軸であり、追記マージだと親の設定に依存して順序が
       読めなくなる。差分設定では常に全体を書く。
    2. **``type`` が変わったコンポーネント指定は置換**。実装が変われば
       残りのキーは別の実装の引数であり、引き継ぐと意味がない。
       たとえば親の ``{type: extractive, max_sentences: 2}`` を
       ``{type: openai_compat, model: ...}`` で上書きしたとき、
       ``max_sentences`` を残してはならない。
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            if _changes_component_type(current, value):
                merged[key] = dict(value)
            else:
                merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _changes_component_type(base: Mapping[str, Any], override: Mapping[str, Any]) -> bool:
    return "type" in base and "type" in override and base["type"] != override["type"]

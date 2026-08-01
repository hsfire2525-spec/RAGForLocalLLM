"""スイープ。ベースラインから軸を振って比較する。

**既定はアブレーション**（1軸ずつ変える）。「何が効いたか」の解釈が容易な
ためで、格子探索は組合せ数が爆発するうえ、どの要因が効いたのか分離できない
（docs/design/design.md §6.6）。

実際に必要になった経緯: 予算の調整（§9 Phase 3-2）で
`context_token_budget` × `chunk_size` × `top_k` を5点比較したが、
シェルのループで設定ファイルを生成する手回しだった。同じことを繰り返すなら
仕組みにしたほうがよい。手回しには3つの問題がある。

1. **設定がどこにも残らない。** 何を比較したのか後から追えない
2. **失敗しても気付きにくい。** 途中で落ちたランが黙って欠ける
3. **インデックスの再利用が読めない。** `index` 側を触る軸は再構築が要る

ここではそれぞれに対処する。スイープ設定はファイルとして残り、失敗した
変種は記録して続行し、インデックスの再構築が要る軸は事前に警告する。
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ragforlocalllm.core.config import ConfigError, ExperimentConfig, load_raw_config

SweepMode = Literal["ablation", "grid"]

# これらを含む軸を振るとインデックスの再構築（＝再埋め込み）が要る
INDEX_AXIS_PREFIX = "index."


class SweepConfig(BaseModel):
    """スイープの定義。"""

    model_config = ConfigDict(extra="forbid")

    base: str
    """基準にする設定名（``configs/<base>.yaml``）。"""
    mode: SweepMode = "ablation"
    axes: dict[str, list[Any]] = Field(default_factory=dict)
    """ドット区切りのパス → 試す値の一覧。

    例: ``query.prompt.context_token_budget: [1536, 2560, 4096]``
    """
    name: str | None = None

    def touches_index(self) -> list[str]:
        return sorted(a for a in self.axes if a.startswith(INDEX_AXIS_PREFIX))


@dataclass(frozen=True)
class Variant:
    """スイープが生成する1つの構成。"""

    name: str
    overrides: dict[str, Any]
    raw: dict[str, Any]

    @property
    def label(self) -> str:
        if not self.overrides:
            return "(基準)"
        return ", ".join(f"{k.rsplit('.', 1)[-1]}={_short(v)}" for k, v in self.overrides.items())


@dataclass
class SweepResult:
    variants: list[Variant] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


def load_sweep(path: str | Path) -> SweepConfig:
    source = Path(path)
    if not source.exists():
        raise ConfigError(f"スイープ設定がありません: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: トップレベルはマッピングである必要があります")
    try:
        return SweepConfig.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"{source}: スイープ設定の検証に失敗しました\n{exc}") from exc


def set_path(payload: dict[str, Any], dotted: str, value: Any) -> None:
    """``a.b.c`` 形式のパスに値を差し込む。

    **途中の階層が無い場合はエラーにする。** 黙って作ると、綴りを間違えた
    軸が「新しいキー」として通ってしまい、何も変えていないのに変えたつもりの
    比較をすることになる。
    """
    keys = dotted.split(".")
    cursor: Any = payload
    for i, key in enumerate(keys[:-1]):
        if not isinstance(cursor, dict) or key not in cursor:
            path = ".".join(keys[: i + 1])
            raise ConfigError(f"軸のパスが基準設定に存在しません: {dotted}（{path} が無い）")
        cursor = cursor[key]
    leaf = keys[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        raise ConfigError(f"軸のパスが基準設定に存在しません: {dotted}")

    # **実装を差し替えたら、前の実装の引数は捨てる。**
    # 例: {type: hybrid, rrf_k: 60, retrievers: [...]} の type を dense に
    # 変えると、rrf_k と retrievers は dense の引数ではないためレジストリが
    # 拒否する。extends の差分継承でも同じ扱いにしている（core/config.py）。
    if leaf == "type" and cursor.get(leaf) != value:
        for key in [k for k in cursor if k != "type"]:
            del cursor[key]
    cursor[leaf] = value


def expand(sweep: SweepConfig, *, config_dir: Path = Path("configs")) -> list[Variant]:
    """スイープ設定から実行する構成の一覧を作る。"""
    base_raw = load_raw_config(f"{sweep.base}.yaml", search_dir=config_dir)
    base_name = str(base_raw.get("name", sweep.base))

    variants: list[Variant] = [Variant(name=base_name, overrides={}, raw=copy.deepcopy(base_raw))]
    # **結果が同じ構成は1度しか走らせない。** アブレーションでは各軸に
    # 基準値そのものを並べることが多く（比較の見通しのため）、素朴に
    # 展開すると基準と同一のランが軸の数だけ増える。実測では 10 件中 3 件が
    # 基準の重複だった。
    seen: set[str] = {_signature(base_raw)}

    for combo in _combinations(sweep):
        raw = copy.deepcopy(base_raw)
        for dotted, value in combo.items():
            set_path(raw, dotted, copy.deepcopy(value))
        signature = _signature(raw)
        if signature in seen:
            continue
        seen.add(signature)
        name = f"{base_name}__{_slug(combo)}"
        raw["name"] = name
        variants.append(Variant(name=name, overrides=dict(combo), raw=raw))
    return variants


def _combinations(sweep: SweepConfig) -> list[dict[str, Any]]:
    if sweep.mode == "ablation":
        # 基準から1軸ずつ変える。効いた要因の分離が容易。
        return [{axis: value} for axis, values in sweep.axes.items() for value in values]

    combos: list[dict[str, Any]] = [{}]
    for axis, values in sweep.axes.items():
        combos = [{**c, axis: v} for c in combos for v in values]
    return combos


def validate(variant: Variant) -> ExperimentConfig:
    """変種が設定として妥当か確かめる。**実行前に全部見る。**

    30分のスイープの28分目で設定エラーが出るのが最悪なので、
    起動時に全変種を検証する。
    """
    try:
        return ExperimentConfig.model_validate(variant.raw)
    except Exception as exc:
        raise ConfigError(f"{variant.name}: 設定として妥当ではありません\n{exc}") from exc


def _signature(raw: Mapping[str, Any]) -> str:
    """構成の同一性。``name`` は実験のラベルなので無視する。"""
    payload = {k: v for k, v in raw.items() if k != "name"}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=repr)


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= 24 else text[:21] + "…"


def _slug(combo: Mapping[str, Any]) -> str:
    parts = []
    for dotted, value in sorted(combo.items()):
        leaf = dotted.rsplit(".", 1)[-1]
        raw = value if isinstance(value, str | int | float) else _digest(value)
        cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(raw)).strip("-")
        parts.append(f"{leaf}-{cleaned}"[:40])
    return "_".join(parts)


def _digest(value: Any) -> str:
    import hashlib

    payload = repr(value).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=3).hexdigest()


def describe_plan(sweep: SweepConfig, variants: Sequence[Variant]) -> dict[str, Any]:
    """実行前に人へ示す計画。"""
    return {
        "base": sweep.base,
        "mode": sweep.mode,
        "n_variants": len(variants),
        "axes": {axis: len(values) for axis, values in sweep.axes.items()},
        "reindex_axes": sweep.touches_index(),
    }

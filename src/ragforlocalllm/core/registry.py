"""コンポーネントのレジストリ。

``{type: recursive_ja, chunk_size: 512}`` のような設定から実装を解決する。
実験リポジトリでは候補実装が増えるため、未知の名前・引数に対して
「候補の提示付きで即座に落ちる」ことを重視する。
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

_REGISTRY: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


class RegistryError(Exception):
    """レジストリの解決・生成に失敗した。"""


def register(kind: str, name: str) -> Callable[[T], T]:
    """コンポーネントを ``kind``/``name`` で登録するデコレータ。

    例::

        @register("chunker", "fixed")
        class FixedChunker:
            def __init__(self, chunk_size: int = 512, overlap: int = 64): ...
    """

    def decorator(cls: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        existing = bucket.get(name)
        if existing is not None and existing is not cls:
            raise RegistryError(
                f"{kind}/{name} は既に {existing.__module__}.{existing.__qualname__} "
                f"で登録済みです（{cls.__module__}.{cls.__qualname__} と衝突）"
            )
        bucket[name] = cls
        return cls

    return decorator


def available(kind: str) -> list[str]:
    """登録済みの名前一覧（ソート済み）。"""
    return sorted(_REGISTRY.get(kind, {}))


def kinds() -> list[str]:
    return sorted(_REGISTRY)


def lookup(kind: str, name: str) -> type:
    bucket = _REGISTRY.get(kind)
    if bucket is None:
        raise RegistryError(f"未知の種別 {kind!r} です。登録済み: {', '.join(kinds()) or '(なし)'}")
    cls = bucket.get(name)
    if cls is None:
        raise RegistryError(_unknown_name_message(kind, name, bucket))
    return cls


def build(kind: str, spec: Mapping[str, Any], **injected: Any) -> Any:
    """設定 dict からインスタンスを生成する。

    ``spec`` は ``{"type": <登録名>, **コンストラクタ引数}``。

    ``injected`` は設定に書けない依存（構築済みの埋め込み器や
    インデックスなど）を渡すためのもの。設定側で同名のキーが
    指定されていた場合は衝突としてエラーにする。

    **注入は「使うものだけ」渡す。** コンポーネントはコンストラクタに
    書いた依存だけを受け取り、要らないものは無視される。これが無いと、
    合成する側（hybrid 検索器など）が子ごとに必要な依存を知っていなければ
    ならず、依存の一覧が2箇所に分散する。BM25 は埋め込み器を必要としないが、
    hybrid は dense と BM25 の両方を同じ呼び出しで構築する。
    """
    if "type" not in spec:
        raise RegistryError(f"{kind} の設定に 'type' がありません: {dict(spec)!r}")

    name = spec["type"]
    if not isinstance(name, str):
        raise RegistryError(f"{kind} の 'type' は文字列である必要があります: {name!r}")

    cls = lookup(kind, name)
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    conflicts = sorted(set(kwargs) & set(injected))
    if conflicts:
        raise RegistryError(
            f"{kind}/{name}: 設定で指定できないキーが含まれています: {', '.join(conflicts)}"
            "（これらは実行時に注入されます）"
        )
    kwargs.update(_accepted_injections(cls, injected))
    _check_kwargs(kind, name, cls, kwargs)
    try:
        return cls(**kwargs)
    except TypeError as exc:  # シグネチャ検査で拾えない場合の保険
        raise RegistryError(f"{kind}/{name} の生成に失敗しました: {exc}") from exc


def _accepted_injections(cls: type, injected: Mapping[str, Any]) -> dict[str, Any]:
    """コンストラクタが受け取ると宣言している依存だけを返す。

    設定キーと違い、注入は**渡す側が一律に用意する**もの。受け取らない
    コンポーネントにまで渡すとエラーになり、合成が成立しない。
    誤字は「必須引数が足りない」として別途 TypeError になる。
    """
    if not injected:
        return {}
    try:
        signature = inspect.signature(cls)
    except (ValueError, TypeError):  # C拡張など
        return dict(injected)
    if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(injected)
    return {k: v for k, v in injected.items() if k in signature.parameters}


def _check_kwargs(kind: str, name: str, cls: type, kwargs: Mapping[str, Any]) -> None:
    """未知の引数・不足している必須引数を、設定ファイル利用者向けの語彙で報告する。"""
    try:
        # クラス自体に対して取ると self を除いたコンストラクタ引数が得られる
        signature = inspect.signature(cls)
    except (ValueError, TypeError):  # C拡張など
        return

    params = {
        pname: p
        for pname, p in signature.parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    accepts_extra = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())

    if not accepts_extra:
        unknown = sorted(set(kwargs) - set(params))
        if unknown:
            suggestions = []
            for key in unknown:
                close = difflib.get_close_matches(key, params, n=1)
                suggestions.append(f"{key}" + (f"（{close[0]} の誤りでは?）" if close else ""))
            raise RegistryError(
                f"{kind}/{name} は次の設定キーを受け付けません: {', '.join(suggestions)}\n"
                f"  受け付けるキー: {', '.join(params) or '(なし)'}"
            )

    missing = sorted(
        pname for pname, p in params.items() if p.default is p.empty and pname not in kwargs
    )
    if missing:
        raise RegistryError(f"{kind}/{name} に必須の設定キーがありません: {', '.join(missing)}")


def _unknown_name_message(kind: str, name: str, bucket: Mapping[str, type]) -> str:
    close = difflib.get_close_matches(name, bucket, n=3)
    hint = f"\n  近い候補: {', '.join(close)}" if close else ""
    return (
        f"未知の {kind} 実装 {name!r} です。"
        f"\n  登録済み: {', '.join(sorted(bucket)) or '(なし)'}{hint}"
    )


def clear_for_tests() -> None:
    """テスト専用。レジストリを空にする。"""
    _REGISTRY.clear()

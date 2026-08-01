"""アクセス制御と監査ログ。

**社内共有で機密資料を配る以上、ここは後付けできない。** 誰が何を尋ねたかを
残せない状態で資料を検索可能にすると、情報が出た経路を追えなくなる。

このモジュールが提供するのは**共有トークンによる認証と全問い合わせの記録**まで。
以下は意図的に実装していない。組織側の決定が要るためで、技術で勝手に
決めてよい範囲を超える。

- **個人の識別**（誰がトークンを使ったか）… SSO / LDAP との接続が必要
- **資料ごとのアクセス権**… 「誰がどの資料を見てよいか」の方針が先
- **通信の暗号化**… TLS 終端はリバースプロキシ側の仕事

トークンは全員で共有する1つの合言葉であり、**個人を特定しない**。
監査ログに残るのは「いつ・どのIPから・何を尋ねたか」までである。
これで足りない要件なら、SSO を挟むまで社内公開してはいけない。
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOKEN_ENV = "RAG_SERVER_TOKEN"
DEFAULT_AUDIT_LOG = Path("data/private/audit/access.jsonl")
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


@dataclass(frozen=True)
class SecurityPolicy:
    """起動時に確定する公開範囲と認証の設定。"""

    host: str
    token: str | None
    audit_log: Path

    @property
    def is_loopback(self) -> bool:
        return self.host in LOOPBACK

    @property
    def requires_token(self) -> bool:
        return self.token is not None

    def check(self, presented: str | None) -> bool:
        """トークンを検証する。**比較は定数時間で行う。**

        素朴な ``==`` は先頭から一致した長さで所要時間が変わるため、
        トークンを1文字ずつ推測できてしまう。
        """
        if self.token is None:
            return True
        if not presented:
            return False
        return hmac.compare_digest(presented, self.token)

    def warnings(self) -> list[str]:
        """起動時に出す警告。**黙って危険な状態で立ち上げない。**"""
        out: list[str] = []
        if not self.is_loopback and not self.requires_token:
            out.append(
                f"認証なしで {self.host} に公開しようとしています。"
                f"{TOKEN_ENV} を設定するか、host を 127.0.0.1 にしてください。"
            )
        if not self.is_loopback:
            out.append(
                "通信は暗号化されません。LAN上でも問い合わせ内容と資料の断片は"
                "平文で流れます。TLS終端するリバースプロキシの背後に置いてください。"
            )
            out.append(
                "共有トークンは個人を特定しません。監査ログに残るのはIPアドレスまでです。"
                "個人単位の追跡が必要なら SSO を挟むまで社内公開しないでください。"
            )
        return out


def resolve_policy(
    host: str, *, token: str | None = None, audit_log: Path | None = None
) -> SecurityPolicy:
    """環境変数を含めて実効的な設定を決める。"""
    effective = token if token is not None else os.environ.get(TOKEN_ENV)
    return SecurityPolicy(
        host=host,
        token=effective or None,
        audit_log=audit_log or DEFAULT_AUDIT_LOG,
    )


def generate_token() -> str:
    """共有トークンを生成する。"""
    return secrets.token_urlsafe(32)


class AuditLog:
    """問い合わせの記録。

    **回答本文は残さない。** 監査に必要なのは「誰がいつ何を尋ねたか」と
    「どの資料を参照したか」であり、回答をそのまま蓄積すると、
    ログ自体が二次的な機密の集積になる。参照した chunk_id は残すので、
    必要なら後から追跡できる。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def record(
        self,
        *,
        client: str,
        question: str,
        config: str,
        chunk_ids: list[str],
        abstained: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "client": client,
            "config": config,
            "question": question,
            "context_chunk_ids": chunk_ids,
            "abstained": abstained,
            "latency_ms": round(latency_ms, 1),
        }
        if error:
            entry["error"] = error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

"""社内共有サーバの検証。

**機密資料を複数人に配る前提**なので、認証・監査・危険な設定の拒否を
中心に固める。回答の中身より、誰が入れるか・何が残るかのほうが重要。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragforlocalllm.server.security import (
    AuditLog,
    SecurityPolicy,
    generate_token,
    resolve_policy,
)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ragforlocalllm.server.app import create_app


def policy(
    tmp_path: Path, *, host: str = "127.0.0.1", token: str | None = "s3cret"
) -> SecurityPolicy:
    return SecurityPolicy(host=host, token=token, audit_log=tmp_path / "access.jsonl")


@pytest.fixture
def client(tmp_path: Path) -> Any:
    return TestClient(create_app(policy(tmp_path), config_dir=Path("configs")))


# ----------------------------------------------------------------------
# 認証
# ----------------------------------------------------------------------


def test_api_requires_a_token(client: Any) -> None:
    assert client.get("/api/configs").status_code == 401
    assert client.post("/api/ask", json={"question": "x"}).status_code == 401


def test_wrong_token_is_rejected(client: Any) -> None:
    res = client.get("/api/configs", headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 401


def test_valid_token_is_accepted(client: Any) -> None:
    res = client.get("/api/configs", headers={"Authorization": "Bearer s3cret"})
    assert res.status_code == 200
    assert "baseline" in res.json()["configs"]


def test_health_needs_no_token_and_leaks_nothing(client: Any) -> None:
    """疎通確認用。機密を返さないので認証不要にしている。"""
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "auth_required": True}


def test_token_comparison_is_constant_time() -> None:
    """**素朴な == は先頭一致の長さで所要時間が変わり、推測を許す。**"""
    import inspect

    source = inspect.getsource(SecurityPolicy.check)
    assert "compare_digest" in source


def test_generated_tokens_are_unpredictable() -> None:
    tokens = {generate_token() for _ in range(20)}
    assert len(tokens) == 20
    assert all(len(t) >= 32 for t in tokens)


# ----------------------------------------------------------------------
# 危険な設定の検出
# ----------------------------------------------------------------------


def test_public_bind_without_token_is_flagged(tmp_path: Path) -> None:
    """**黙って危険な状態で立ち上げない。**"""
    warnings = policy(tmp_path, host="0.0.0.0", token=None).warnings()
    assert any("認証なし" in w for w in warnings)


def test_public_bind_warns_about_plaintext_and_identity(tmp_path: Path) -> None:
    warnings = policy(tmp_path, host="0.0.0.0", token="t").warnings()
    assert any("暗号化されません" in w for w in warnings)
    # 共有トークンは個人を特定しない。これを黙っていると監査要件を誤解させる。
    assert any("個人を特定しません" in w for w in warnings)


def test_loopback_is_quiet(tmp_path: Path) -> None:
    assert policy(tmp_path, host="127.0.0.1", token=None).warnings() == []


def test_token_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SERVER_TOKEN", "from-env")
    assert resolve_policy("127.0.0.1").token == "from-env"


def test_explicit_token_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SERVER_TOKEN", "from-env")
    assert resolve_policy("127.0.0.1", token="explicit").token == "explicit"


# ----------------------------------------------------------------------
# 監査ログ
# ----------------------------------------------------------------------


def test_audit_records_who_asked_what(tmp_path: Path) -> None:
    import json

    log = AuditLog(tmp_path / "a.jsonl")
    log.record(
        client="10.0.0.5",
        question="就業規則の休職期間は",
        config="private_docs",
        chunk_ids=["doc#c0007"],
        abstained=False,
        latency_ms=1234.5,
    )
    entry = json.loads((tmp_path / "a.jsonl").read_text(encoding="utf-8"))
    assert entry["client"] == "10.0.0.5"
    assert entry["question"] == "就業規則の休職期間は"
    assert entry["context_chunk_ids"] == ["doc#c0007"]
    assert entry["at"]


def test_audit_does_not_store_the_answer(tmp_path: Path) -> None:
    """**回答本文は残さない。**

    監査に要るのは「誰がいつ何を尋ね、どこを参照したか」。回答を
    蓄積するとログ自体が二次的な機密の集積になる。
    """
    import inspect

    source = inspect.getsource(AuditLog.record)
    assert "answer" not in source


def test_audit_appends(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(3):
        log.record(
            client="c", question=f"q{i}", config="x", chunk_ids=[], abstained=False, latency_ms=1.0
        )
    assert len((tmp_path / "a.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 3


# ----------------------------------------------------------------------
# 入力の検証
# ----------------------------------------------------------------------


def test_path_traversal_in_config_is_rejected_at_input(client: Any) -> None:
    """設定名はファイル名になる。入力側で `../` を弾く（多層防御の外側）。"""
    res = client.post(
        "/api/ask",
        json={"question": "x", "config": "../../etc/passwd"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert res.status_code == 422


def test_unknown_config_is_rejected(client: Any) -> None:
    """名前として妥当でも、実在しない設定は拒否する（多層防御の内側）。"""
    res = client.post(
        "/api/ask",
        json={"question": "x", "config": "does_not_exist"},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert res.status_code == 404


def test_empty_and_oversized_questions_are_rejected(client: Any) -> None:
    auth = {"Authorization": "Bearer s3cret"}
    assert client.post("/api/ask", json={"question": ""}, headers=auth).status_code == 422
    assert client.post("/api/ask", json={"question": "あ" * 5000}, headers=auth).status_code == 422


def test_ui_is_served(client: Any) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "社内文書検索" in res.text


def test_ui_has_no_external_references() -> None:
    """**閉域で動かす前提。** 外部参照があると動かないか、情報が漏れる。"""
    html = (
        Path(__file__).resolve().parent.parent / "src/ragforlocalllm/server/static/index.html"
    ).read_text(encoding="utf-8")
    for marker in ("http://", "https://", "//cdn", "integrity="):
        assert marker not in html, marker

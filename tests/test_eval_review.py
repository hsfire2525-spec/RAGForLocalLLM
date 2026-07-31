from __future__ import annotations

from pathlib import Path

import pytest

from ragforlocalllm.eval.review import (
    JudgmentStore,
    agreement,
    answer_key,
    stratified_sample,
)


def row(qid: str, **kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "qid": qid,
        "question": "q",
        "answer": "経営者",
        "outcome": "correct",
        "question_type": "responsibility",
    }
    payload.update(kwargs)
    return payload


# ----------------------------------------------------------------------
# 過去判定の再利用
# ----------------------------------------------------------------------


def test_judgment_is_reused_for_the_same_answer(tmp_path: Path) -> None:
    """**これがないと人手コストが実験回数に比例して増える。**"""
    store = JudgmentStore(tmp_path / "j.jsonl")
    store.record("q1", "経営者です。", "correct")

    # 表記が揺れても正規化して同一視する
    assert store.get("q1", "経営者です") is not None
    assert store.get("q1", "経営者 です。") is not None
    # 回答が変われば再判定が必要
    assert store.get("q1", "従業員です。") is None
    # 質問が違えば別物
    assert store.get("q2", "経営者です。") is None


def test_store_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    JudgmentStore(path).record("q1", "経営者", "correct", comment="妥当")

    reloaded = JudgmentStore(path)
    judgment = reloaded.get("q1", "経営者")
    assert judgment is not None
    assert judgment.verdict == "correct"
    assert judgment.comment == "妥当"


def test_answer_key_handles_missing_answer() -> None:
    assert answer_key("q1", None) == answer_key("q1", "")


# ----------------------------------------------------------------------
# 層別サンプリング
# ----------------------------------------------------------------------


def test_stratified_sample_covers_every_stratum() -> None:
    """単純無作為だと少数の質問タイプが丸ごと抜け落ちる。"""
    rows = (
        [row(f"a{i}", question_type="numeric") for i in range(20)]
        + [row("b1", question_type="table")]
        + [row("c1", question_type="unanswerable")]
    )
    sample = stratified_sample(rows, 6)
    types = {str(r["question_type"]) for r in sample}
    assert types == {"numeric", "table", "unanswerable"}


def test_stratified_sample_is_deterministic() -> None:
    rows = [row(f"q{i}", question_type="numeric") for i in range(20)]
    assert [r["qid"] for r in stratified_sample(rows, 5)] == [
        r["qid"] for r in stratified_sample(rows, 5)
    ]


def test_stratified_sample_handles_small_pools() -> None:
    rows = [row("q1"), row("q2")]
    assert len(stratified_sample(rows, 10)) == 2
    assert stratified_sample(rows, 0) == []


# ----------------------------------------------------------------------
# 自動採点との一致率
# ----------------------------------------------------------------------


def test_agreement_maps_the_four_values(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path / "j.jsonl")
    rows = [
        row("q1", answer="経営者", outcome="correct"),
        row("q2", answer="従業員", outcome="correct"),
        row("q3", answer="分かりません", outcome="correct_abstention"),
    ]
    store.record("q1", "経営者", "correct")
    store.record("q2", "従業員", "incorrect")  # 自動は正答としたが人手は誤答
    store.record("q3", "分かりません", "valid_abstention")

    result = agreement(rows, store)
    assert result["n_reviewed"] == 3
    assert result["agreement_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert [d["qid"] for d in result["disagreements"]] == ["q2"]


def test_agreement_ignores_unreviewed_rows(tmp_path: Path) -> None:
    store = JudgmentStore(tmp_path / "j.jsonl")
    result = agreement([row("q1")], store)
    assert result["n_reviewed"] == 0
    assert result["agreement_rate"] is None

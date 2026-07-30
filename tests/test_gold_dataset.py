from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragforlocalllm.eval.dataset import Evidence, GoldItem, load_gold

SAMPLE = Path("data/gold/sample_qa.jsonl")


def test_sample_dataset_loads() -> None:
    gold = load_gold(SAMPLE)
    assert len(gold) == 8
    assert gold.answerable_count == 6


def test_sample_dataset_has_enough_unanswerable() -> None:
    """棄権性能を測るため answerable=false を10〜20%含める方針の検証。"""
    gold = load_gold(SAMPLE)
    assert gold.unanswerable_ratio >= 0.10


def test_sample_dataset_includes_machine_checkable_types() -> None:
    """外部judgeが使えないため、機械採点可能な型が中心であること。"""
    gold = load_gold(SAMPLE)
    checkable = {"short", "numeric", "list"}
    assert all(item.answer_type in checkable for item in gold)


def test_evidence_requires_an_anchor() -> None:
    with pytest.raises(ValidationError):
        Evidence()


def test_evidence_accepts_quote_only() -> None:
    ev = Evidence(quote="経営者が承認する")
    assert ev.quote == "経営者が承認する"


def test_answerable_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence が必要"):
        GoldItem(qid="x", question="q", answer="a", answerable=True)


def test_unanswerable_rejects_evidence() -> None:
    with pytest.raises(ValidationError, match="evidence は指定できません"):
        GoldItem(
            qid="x",
            question="q",
            answer="分かりません",
            answerable=False,
            evidence=[Evidence(quote="something")],
        )


def test_list_answer_splits_into_items() -> None:
    item = GoldItem(
        qid="x",
        question="q",
        answer="年1回以上、入社後30日以内",
        answer_type="list",
        evidence=[Evidence(quote="year")],
    )
    assert item.answer_items == ["年1回以上", "入社後30日以内"]


def test_duplicate_qid_is_rejected(tmp_path: Path) -> None:
    row = {
        "qid": "dup",
        "question": "q",
        "answer": "a",
        "evidence": [{"quote": "z"}],
    }
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重複"):
        load_gold(path)


def test_comment_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "commented.jsonl"
    path.write_text(
        "# コメント\n\n"
        + json.dumps({"qid": "a", "question": "q", "answer": "a", "evidence": [{"quote": "z"}]})
        + "\n",
        encoding="utf-8",
    )
    assert len(load_gold(path)) == 1

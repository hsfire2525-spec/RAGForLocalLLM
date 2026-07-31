"""評価ランナーとランレコードの結合検証。

追加依存もLLMも要らない ``configs/smoke.yaml``（合成コーパス）で
一通り流す。**完了条件「ダミー回答器に対して評価が回り、数値と
信頼区間が出る」**（docs/design/design.md §8 Phase 1）に対応する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragforlocalllm import stages  # noqa: F401 - レジストリ登録のため
from ragforlocalllm.core.cache import NullCache
from ragforlocalllm.core.config import load_config
from ragforlocalllm.eval.dataset import load_gold
from ragforlocalllm.eval.record import RunRecord, create_run, find_runs, resolve_run
from ragforlocalllm.eval.runner import EvaluationResult, run_evaluation
from ragforlocalllm.experiments.report import compare, load_series

CONFIG = Path("configs/smoke.yaml")
GOLD = Path("data/gold/sample_qa.jsonl")


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> EvaluationResult:
    root = tmp_path_factory.mktemp("runs")
    config = load_config(CONFIG, search_dir=Path("configs"))
    with NullCache() as cache:
        return run_evaluation(config, load_gold(GOLD), cache=cache, root=root)


# ----------------------------------------------------------------------


def test_evaluation_produces_the_expected_record_files(evaluated: EvaluationResult) -> None:
    directory = evaluated.record.directory
    for filename in ("config.resolved.yaml", "env.json", "predictions.jsonl", "metrics.json"):
        assert (directory / filename).exists(), filename


def test_metrics_report_correct_error_and_abstention(evaluated: EvaluationResult) -> None:
    generation = evaluated.metrics["generation"]
    for key in ("accuracy", "error_rate", "abstention_rate", "abstention_precision"):
        assert key in generation
    ci = generation["accuracy_ci"]
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_metrics_are_valid_json_without_nan(evaluated: EvaluationResult) -> None:
    """NaN は JSON として不正。書けても他のツールから読めなくなる。"""
    text = (evaluated.record.directory / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    json.loads(text)


def test_environment_records_the_corpus_hash(evaluated: EvaluationResult) -> None:
    """コーパスをコミットできない以上、ハッシュが無いと再現不能。"""
    env = evaluated.record.read_env()
    assert env["corpus"]["sha256"]
    assert env["index"]["signature"]
    assert env["config_hash"]


def test_retrieval_metrics_exclude_unanswerable_questions(evaluated: EvaluationResult) -> None:
    metrics = evaluated.metrics
    assert metrics["retrieval"]["n_measured"] == metrics["resolution"]["n_measurable"]
    assert metrics["retrieval"]["n_measured"] < metrics["n_items"]


def test_predictions_keep_per_question_detail(evaluated: EvaluationResult) -> None:
    rows = evaluated.record.read_predictions()
    assert len(rows) == evaluated.metrics["n_items"]
    first = rows[0]
    for key in ("qid", "outcome", "char_f1", "gold_chunk_ids", "context_chunk_ids", "trace"):
        assert key in first


def test_stratified_summary_includes_counts(evaluated: EvaluationResult) -> None:
    """層ごとの n が無いと、小標本であることが読み取れない。"""
    by_type = evaluated.metrics["by_question_type"]
    assert by_type
    assert all("n" in stats for stats in by_type.values())


# ----------------------------------------------------------------------
# ランレコード
# ----------------------------------------------------------------------


def test_run_directory_includes_the_config_hash(tmp_path: Path) -> None:
    """同じ名前で中身が違う実験を後から区別できるようにする。"""
    record = create_run("baseline", "abc123", root=tmp_path)
    assert record.name.endswith("-baseline-abc123")


def test_find_runs_ignores_incomplete_directories(tmp_path: Path) -> None:
    (tmp_path / "20260731-000000-partial-aaa").mkdir()
    complete = create_run("done", "bbb", root=tmp_path)
    complete.write_metrics({"n_items": 1})
    assert [r.name for r in find_runs(tmp_path)] == [complete.name]


def test_resolve_run_by_prefix(tmp_path: Path) -> None:
    record = create_run("baseline", "abc123", root=tmp_path)
    record.write_metrics({"n_items": 1})
    assert resolve_run(record.name[:17], root=tmp_path).name == record.name

    with pytest.raises(FileNotFoundError):
        resolve_run("nope", root=tmp_path)


def test_resolve_run_rejects_ambiguous_prefix(tmp_path: Path) -> None:
    for suffix in ("aaa", "bbb"):
        record = RunRecord(tmp_path / f"20260731-000000-run-{suffix}")
        record.write_metrics({"n_items": 1})
    with pytest.raises(ValueError, match="曖昧"):
        resolve_run("20260731", root=tmp_path)


def test_reviews_are_appended(tmp_path: Path) -> None:
    record = create_run("r", "h", root=tmp_path)
    record.append_review({"qid": "q1", "verdict": "correct"})
    record.append_review({"qid": "q2", "verdict": "incorrect"})
    assert [r["qid"] for r in record.read_reviews()] == ["q1", "q2"]


# ----------------------------------------------------------------------
# 比較
# ----------------------------------------------------------------------


def test_comparing_a_run_with_itself_shows_no_difference(evaluated: EvaluationResult) -> None:
    record = evaluated.record
    rows = compare([record, record], metrics=("accuracy",))
    baseline, other = rows
    assert baseline.verdict == "基準"
    assert other.diff is not None
    assert other.diff.point == pytest.approx(0.0)
    assert other.verdict == "有意差なし"


def test_comparison_requires_shared_questions(evaluated: EvaluationResult, tmp_path: Path) -> None:
    """gold を入れ替えた前後のランを並べても差は意味を持たない。"""
    other = create_run("other", "zzz", root=tmp_path)
    other.write_metrics({"n_items": 1})
    other.write_predictions([{"qid": "別の質問", "outcome": "correct"}])
    with pytest.raises(ValueError, match="共通する qid"):
        compare([evaluated.record, other])


def test_series_maps_outcomes_to_binary_correctness(evaluated: EvaluationResult) -> None:
    series = load_series(evaluated.record)
    qids = sorted(series.qids)
    values = series.series("accuracy", qids)
    assert set(values) <= {0.0, 1.0}
    assert len(values) == len(qids)

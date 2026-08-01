from __future__ import annotations

import pytest

from ragforlocalllm.core.types import Answer, Chunk
from ragforlocalllm.eval.dataset import Evidence, GoldItem
from ragforlocalllm.eval.metrics import (
    aggregate_outcomes,
    bootstrap_mean,
    bootstrap_paired_diff,
    context_precision,
    evidence_recall_at_k,
    hit_at_k,
    is_significant,
    judge_answer,
    judge_citations,
    ndcg_at_k,
    reciprocal_rank,
)
from ragforlocalllm.eval.resolve import Resolver


def gold(**kwargs: object) -> GoldItem:
    payload: dict[str, object] = {
        "qid": "q1",
        "question": "承認するのは誰か。",
        "answer": "経営者",
        "evidence": [Evidence(page=1)],
    }
    payload.update(kwargs)
    return GoldItem.model_validate(payload)


def answer(text: str, *, abstained: bool = False, citations: list[str] | None = None) -> Answer:
    return Answer(text=text, abstained=abstained, citations=citations or [])


# ----------------------------------------------------------------------
# 4値分類
# ----------------------------------------------------------------------


def test_correct_answer_with_explanation_counts_as_correct() -> None:
    judgment = judge_answer(gold(), answer("承認するのは経営者です。"))
    assert judgment.outcome == "correct"
    assert not judgment.exact_match  # EM だけでは取りこぼす
    assert judgment.contains


def test_wrong_answer_is_incorrect() -> None:
    assert judge_answer(gold(), answer("従業員です。")).outcome == "incorrect"


def test_abstaining_on_answerable_question_is_unjustified() -> None:
    judgment = judge_answer(gold(), answer("分かりません", abstained=True))
    assert judgment.outcome == "unjustified_abstention"
    assert not judgment.correct


def test_abstaining_on_unanswerable_question_is_correct() -> None:
    item = gold(answerable=False, evidence=[], answer="分かりません")
    judgment = judge_answer(item, answer("分かりません", abstained=True))
    assert judgment.outcome == "correct_abstention"
    assert judgment.correct


def test_answering_an_unanswerable_question_is_incorrect() -> None:
    """**ここを別枠にすると、何にでも答える構成の誤答率が低く見える。**"""
    item = gold(answerable=False, evidence=[], answer="分かりません")
    assert judge_answer(item, answer("経営者です。")).outcome == "incorrect"


def test_numeric_answers_do_not_match_on_substrings() -> None:
    """「5」が「15項目」に含まれてしまう事故を防ぐ。"""
    item = gold(answer="5", answer_type="numeric")
    assert judge_answer(item, answer("全部で15項目です。")).outcome == "incorrect"
    assert judge_answer(item, answer("全部で5項目です。")).outcome == "correct"


def test_list_answers_use_set_f1() -> None:
    item = gold(answer="経営者、従業員、委託先", answer_type="list")
    assert judge_answer(item, answer("経営者、委託先、従業員")).outcome == "correct"
    assert judge_answer(item, answer("経営者")).outcome == "incorrect"


def test_long_answers_are_flagged_for_human_review() -> None:
    item = gold(answer="経営者が承認し周知する", answer_type="long")
    assert judge_answer(item, answer("経営者が承認し周知する")).needs_human_review


# ----------------------------------------------------------------------
# 集計
# ----------------------------------------------------------------------


def test_rates_report_correct_error_and_abstention_together() -> None:
    judgments = [
        judge_answer(gold(), answer("経営者")),
        judge_answer(gold(), answer("従業員")),
        judge_answer(gold(), answer("分かりません", abstained=True)),
        judge_answer(
            gold(answerable=False, evidence=[], answer="分かりません"),
            answer("分かりません", abstained=True),
        ),
    ]
    rates = aggregate_outcomes(judgments)

    assert rates.accuracy == pytest.approx(0.5)  # correct + correct_abstention
    assert rates.error_rate == pytest.approx(0.25)
    assert rates.abstention_rate == pytest.approx(0.5)
    assert rates.abstention_precision == pytest.approx(0.5)
    assert rates.abstention_recall(n_unanswerable=1) == pytest.approx(1.0)


def test_rates_dict_includes_recall_only_when_denominator_given() -> None:
    rates = aggregate_outcomes([judge_answer(gold(), answer("経営者"))])
    assert "abstention_recall" not in rates.as_dict()
    assert "abstention_recall" in rates.as_dict(n_unanswerable=0)


# ----------------------------------------------------------------------
# 引用
# ----------------------------------------------------------------------


def test_citation_to_a_nonexistent_chunk_is_flagged() -> None:
    """存在しないIDを引用していれば、コンテキストを見ずに真似ている。"""
    judgment = judge_citations(answer("経営者", citations=["c9"]), ["c1", "c2"], frozenset({"c1"}))
    assert judgment.cited
    assert not judgment.all_exist
    assert judgment.n_hallucinated == 1
    assert not judgment.supported


def test_citation_pointing_at_gold_evidence_is_supported() -> None:
    judgment = judge_citations(answer("経営者", citations=["c1"]), ["c1", "c2"], frozenset({"c1"}))
    assert judgment.all_exist
    assert judgment.supported


# ----------------------------------------------------------------------
# 検索
# ----------------------------------------------------------------------


def resolution_for(evidences: list[Evidence], chunks: list[Chunk]) -> object:
    item = GoldItem(qid="q1", question="q", answer="a", evidence=evidences)
    return Resolver(chunks).resolve_item(item)


def make_chunks(*specs: tuple[str, str, int]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=cid,
            doc_id="doc",
            text=text,
            metadata={"page": page, "page_start": page, "page_end": page},
        )
        for cid, text, page in specs
    ]


def test_multi_hop_recall_counts_each_evidence() -> None:
    """1つの Evidence 内は選択肢、Evidence 間はすべて必要。"""
    chunks = make_chunks(("c1", "根拠A", 1), ("c2", "根拠B", 2))
    resolution = resolution_for(
        [Evidence(page=1, quote="根拠A"), Evidence(page=2, quote="根拠B")], chunks
    )

    assert evidence_recall_at_k(resolution, ["c1"], 5) == pytest.approx(0.5)
    assert evidence_recall_at_k(resolution, ["c1", "c2"], 5) == pytest.approx(1.0)
    # hit は「1つでも取れたか」なので片方でも 1.0
    assert hit_at_k(resolution, ["c1"], 5) == pytest.approx(1.0)


def test_reciprocal_rank_uses_the_first_gold_chunk() -> None:
    chunks = make_chunks(("c1", "根拠A", 1), ("c2", "無関係", 2))
    resolution = resolution_for([Evidence(page=1, quote="根拠A")], chunks)
    assert reciprocal_rank(resolution, ["c2", "c1"]) == pytest.approx(0.5)
    assert reciprocal_rank(resolution, ["c2"]) == pytest.approx(0.0)


def test_ndcg_ideal_uses_evidence_count_not_gold_set_size() -> None:
    """ページ指定で解決した質問が構造的に低く出ないようにする。"""
    chunks = make_chunks(("c1", "本文A", 5), ("c2", "本文B", 5), ("c3", "別ページ", 9))
    # ページアンカーなので c1 と c2 の両方が正解集合に入る
    resolution = resolution_for([Evidence(page=5)], chunks)
    assert ndcg_at_k(resolution, ["c1", "c3"], 5) == pytest.approx(1.0)


def test_context_precision_measures_noise_in_the_prompt() -> None:
    chunks = make_chunks(("c1", "根拠A", 1), ("c2", "無関係", 2))
    resolution = resolution_for([Evidence(page=1, quote="根拠A")], chunks)
    assert context_precision(resolution, ["c1", "c2"]) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# 統計
# ----------------------------------------------------------------------


def test_bootstrap_is_deterministic() -> None:
    """区間が実行ごとに動くと実験ログの再現性が崩れる。"""
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert bootstrap_mean(values).as_dict() == bootstrap_mean(values).as_dict()


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    interval = bootstrap_mean([1.0, 0.0, 1.0, 1.0, 0.0])
    assert interval.low <= interval.point <= interval.high


def test_small_samples_give_wide_intervals() -> None:
    """30〜50問では数ポイントの差はノイズ。区間がそれを示す。"""
    interval = bootstrap_mean([1.0, 0.0, 1.0, 0.0])
    assert interval.high - interval.low > 0.5


def test_paired_diff_detects_a_consistent_improvement() -> None:
    """全問で改善していれば、小標本でも有意差として検出できる。"""
    a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert is_significant(bootstrap_paired_diff(a, b))


def test_paired_diff_of_noise_is_not_significant() -> None:
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    assert not is_significant(bootstrap_paired_diff(a, b))


def test_paired_diff_requires_aligned_series() -> None:
    with pytest.raises(ValueError, match="同じ長さ"):
        bootstrap_paired_diff([1.0, 0.0], [1.0])


# ----------------------------------------------------------------------
# 冗長な回答の採点（実測で見つかった採点漏れの回帰）
# ----------------------------------------------------------------------


def test_enumeration_answered_inside_a_sentence_is_correct() -> None:
    """**モデルは列挙を裸の単語では返さない。**

    要素の完全一致を要求すると、内容として正しい回答が 0 点になる。
    実測では列挙型の正答率が 0.20 まで落ちた。
    """
    item = gold(
        answer="実施している、一部実施している、実施していない、わからない", answer_type="list"
    )
    pred = (
        "「実施している 4点」「一部実施している 2点」「実施していない 0点」「わからない -１点」 [1]"
    )
    assert judge_answer(item, answer(pred)).outcome == "correct"


def test_verbose_but_correct_long_answer_is_accepted() -> None:
    """char F1 は冗長さを強く罰する。窓で見れば内容の一致が拾える。"""
    item = gold(
        answer="脅威の起こりやすさと脆弱性のつけ込みやすさの2つの数値から算出する",
        answer_type="long",
    )
    pred = (
        "「被害発生可能性」は、「脅威の起こりやすさ」と「脆弱性のつけ込みやすさ」の2つの数値から"
        "算出されます [3]。これは、脅威が脆弱性を利用して、どの程度被害をもたらす可能性があるかを"
        "示すものです。"
    )
    assert judge_answer(item, answer(pred)).outcome == "correct"


def test_phrase_short_answer_tolerates_rewording() -> None:
    """句の短答は助詞や語尾の違いだけで EM も包含も落ちる。"""
    item = gold(answer="委託先がどのような情報セキュリティ対策を行っているか考慮する")
    pred = (
        "業務の一部を外部に委託し重要な情報を委託先に提供する場合、委託先がどのような"
        "情報セキュリティ対策を行っているかを考慮する必要がある [1]。"
    )
    assert judge_answer(item, answer(pred)).outcome == "correct"


def test_single_word_short_answer_stays_strict() -> None:
    """**単語の短答に char F1 を使ってはいけない。**

    「経営」は「経営者」に対して char F1 0.8 になり、誤って正答になる。
    gold の長さで指標を切り替えている。
    """
    item = gold(answer="経営者")
    assert judge_answer(item, answer("従業")).outcome == "incorrect"


def test_plausible_but_wrong_answers_are_still_rejected() -> None:
    """緩めた採点が「何でも正答」になっていないことの確認。"""
    cases = [
        (gold(answer="5", answer_type="numeric"), "設問は30項目です。"),
        (
            gold(answer="知っているもの、持っているもの、本人自身に関するもの", answer_type="list"),
            "3要素は、パスワード、合言葉、秘密の質問です。",
        ),
        (
            gold(answer="重要度と被害発生可能性の2つの数値の掛け算で算定する", answer_type="long"),
            "リスク値は、資産価値と復旧コストの足し算で算定します。",
        ),
        (
            gold(answer="委託先がどのような情報セキュリティ対策を行っているか考慮する"),
            "外注先の対策状況は発注側の責任外なので、確認する必要はありません。",
        ),
    ]
    for item, pred in cases:
        assert judge_answer(item, answer(pred)).outcome == "incorrect", pred

from __future__ import annotations

import pytest

from ragforlocalllm.eval.normalize import (
    best_char_f1,
    char_f1,
    contains_answer,
    exact_match,
    normalize_answer,
    normalize_for_match,
    set_f1,
    split_items,
)


def test_whitespace_is_removed_for_matching() -> None:
    """PDF抽出では字間調整で空白が入る。残すと引用が解決できない。"""
    assert (
        normalize_for_match("リスク値 ＝ 重要度 × 被害発生可能性")
        == "リスク値=重要度×被害発生可能性"
    )


def test_line_breaks_are_removed_for_matching() -> None:
    """チャンク本文は行で折り返されている。引用文はそれを跨ぐ。"""
    assert normalize_for_match("経営者が\n承認し") == "経営者が承認し"


def test_fullwidth_and_halfwidth_are_folded() -> None:
    assert normalize_for_match("ＥＤＲ１") == normalize_for_match("EDR1")


def test_symbols_are_kept_for_matching_but_dropped_for_scoring() -> None:
    """記号は引用の識別に効くので解決では残し、採点では落とす。"""
    assert "=" in normalize_for_match("リスク値＝重要度")
    assert normalize_answer("経営者。") == normalize_answer("経営者")


def test_exact_match_uses_aliases() -> None:
    assert exact_match("社長", ["経営者", "社長"])
    assert not exact_match("従業員", ["経営者", "社長"])


def test_exact_match_rejects_empty_prediction() -> None:
    assert not exact_match("", ["経営者"])
    assert not exact_match("。", ["経営者"])


def test_contains_catches_short_answers_with_explanation() -> None:
    """4B級モデルは短答に説明を付ける。EMだけでは正しい回答を取りこぼす。"""
    assert contains_answer("承認するのは経営者です。", ["経営者"])
    assert not contains_answer("承認するのは従業員です。", ["経営者"])


def test_char_f1_is_symmetric_and_bounded() -> None:
    assert char_f1("経営者", "経営者") == pytest.approx(1.0)
    assert char_f1("経営者", "従業員") == pytest.approx(0.0)
    assert 0.0 < char_f1("経営者が承認", "経営者") < 1.0


def test_char_f1_handles_empty_strings() -> None:
    assert char_f1("", "") == pytest.approx(1.0)
    assert char_f1("経営者", "") == pytest.approx(0.0)


def test_best_char_f1_takes_the_best_alias() -> None:
    assert best_char_f1("社長", ["経営者", "社長"]) == pytest.approx(1.0)


def test_set_f1_ignores_order_and_notation() -> None:
    assert set_f1(["経営者", "従業員"], ["従業員", "経営者"]) == pytest.approx(1.0)
    assert set_f1(["経営者"], ["経営者", "従業員"]) == pytest.approx(2 / 3)


def test_split_items_keeps_decimal_numbers_intact() -> None:
    """小数点を区切りと誤認すると数値回答が壊れる。"""
    assert split_items("3.5") == ["3.5"]
    assert split_items("経営者、従業員") == ["経営者", "従業員"]
    assert split_items("経営者\n従業員") == ["経営者", "従業員"]

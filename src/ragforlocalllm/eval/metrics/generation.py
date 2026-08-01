"""生成段のメトリクス。

**単一の精度指標では棄権の効果が見えない。** 「分かりません」と答える
構成は、正答率だけを見れば低いが、誤答率で見れば優秀でありうる。
実用上の目的は「誤答を避けつつ正答を増やす」ことなので、
**正答 / 誤答 / 棄権を常に併記する**（docs/design/design.md §6.3）。

そのため回答は4値で分類する:

| 分類 | 意味 |
| --- | --- |
| ``correct`` | 回答可能な質問に正answered |
| ``incorrect`` | 回答可能な質問に誤答、または回答不能な質問に答えてしまった |
| ``correct_abstention`` | 回答不能な質問に正しく棄権した |
| ``unjustified_abstention`` | 回答可能な質問に棄権した（取りこぼし） |

``incorrect`` に「回答不能なのに答えた」を含めるのが要点で、これを
別枠にすると、何にでも答える構成の誤答率が低く見えてしまう。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ragforlocalllm.core.types import Answer
from ragforlocalllm.eval.dataset import GoldItem
from ragforlocalllm.eval.normalize import (
    best_char_f1,
    best_windowed_char_f1,
    contains_answer,
    exact_match,
    normalize_answer,
    set_f1,
    split_items,
)

Outcome = Literal["correct", "incorrect", "correct_abstention", "unjustified_abstention"]

_DIGITS = re.compile(r"[0-9]+(?:\.[0-9]+)?")

PHRASE_MIN_CHARS = 10
"""短答型を「単語」と「句」に分ける境界（正規化後の文字数）。

これ以上の長さの gold には char F1 の判定を併用する。これ未満では
文字の偶然の重なりで誤って正答になるため、EM と包含だけで見る。
"""


@dataclass(frozen=True)
class AnswerJudgment:
    """1件の回答に対する自動採点結果。

    採点の内訳（EM / 包含 / F1）も残す。**自動指標が人手判定と
    どこで食い違うか**を後から調べられないと、指標を信頼できる範囲が
    分からない（docs/design/design.md §6.4）。
    """

    outcome: Outcome
    exact_match: bool
    contains: bool
    char_f1: float
    set_f1: float | None = None
    needs_human_review: bool = False
    """自由記述型など、自動採点の信頼度が低いもの。抽出検査の層別に使う。"""

    @property
    def correct(self) -> bool:
        return self.outcome in ("correct", "correct_abstention")

    @property
    def abstained(self) -> bool:
        return self.outcome in ("correct_abstention", "unjustified_abstention")


def judge_answer(
    item: GoldItem,
    answer: Answer,
    *,
    char_f1_threshold: float = 0.6,
    set_f1_threshold: float = 0.6,
    accept_contains: bool = True,
) -> AnswerJudgment:
    """gold と回答を突き合わせて4値に分類する。

    Parameters
    ----------
    accept_contains:
        受理回答を部分文字列として含めば正解とみなす。4B級モデルは
        「経営者です。」のように短答へ説明を付けるため、EM だけで測ると
        内容が正しい回答を大量に取りこぼす。
    """
    em = exact_match(answer.text, item.accepted_answers)
    contains = contains_answer(answer.text, item.accepted_answers)
    cf1 = best_char_f1(answer.text, item.accepted_answers)
    sf1 = (
        set_f1(split_items(answer.text), item.answer_items) if item.answer_type == "list" else None
    )

    if not item.answerable:
        outcome: Outcome = "correct_abstention" if answer.abstained else "incorrect"
        return AnswerJudgment(outcome, em, contains, cf1, sf1)

    if answer.abstained:
        return AnswerJudgment("unjustified_abstention", em, contains, cf1, sf1)

    correct = _is_correct(
        item,
        answer.text,
        em=em,
        contains=contains,
        char_f1_value=cf1,
        windowed=best_windowed_char_f1(answer.text, item.accepted_answers),
        set_f1_value=sf1,
        char_f1_threshold=char_f1_threshold,
        set_f1_threshold=set_f1_threshold,
        accept_contains=accept_contains,
    )
    return AnswerJudgment(
        "correct" if correct else "incorrect",
        em,
        contains,
        cf1,
        sf1,
        needs_human_review=item.answer_type == "long",
    )


def _is_correct(
    item: GoldItem,
    prediction: str,
    *,
    em: bool,
    contains: bool,
    char_f1_value: float,
    windowed: float,
    set_f1_value: float | None,
    char_f1_threshold: float,
    set_f1_threshold: float,
    accept_contains: bool,
) -> bool:
    if item.answer_type == "list":
        return (set_f1_value or 0.0) >= set_f1_threshold
    if item.answer_type == "long":
        # 言い回しが違っても内容が一致していれば正答とする。包含は
        # 「gold の表現をそのまま含む」場合を拾い、窓付き char F1 は
        # 冗長な回答の中に埋もれた正解を拾う。
        return contains or windowed >= char_f1_threshold
    if item.answer_type == "numeric":
        # 「5」が「15項目」に含まれてしまうため、数値型では包含判定を
        # 使わず、数値トークンの一致で見る。
        return em or _numeric_match(prediction, item.accepted_answers)
    if em or (accept_contains and contains):
        return True
    # **短答型には2種類ある。** 単語（「経営者」）と句（「利用者認証を…
    # 行う技術」）で、適切な指標が違う。単語に char F1 を使うと「経営」が
    # 「経営者」に対して 0.8 になり誤って正答になる。句では逆に、助詞や
    # 語順の違いだけで EM も包含も落ちる（実測で definition の正答率が
    # 0.60 まで下がった）。gold の長さで切り分ける。
    return len(normalize_answer(item.answer)) >= PHRASE_MIN_CHARS and (
        windowed >= char_f1_threshold
    )


def _numeric_match(prediction: str, accepted: Sequence[str]) -> bool:
    """予測に含まれる数値トークンの中に、受理回答の数値が現れるか。"""
    predicted = set(_DIGITS.findall(normalize_answer(prediction)))
    if not predicted:
        return False
    for candidate in accepted:
        wanted = _DIGITS.findall(normalize_answer(candidate))
        if wanted and set(wanted) <= predicted:
            return True
    return False


# ----------------------------------------------------------------------
# 引用の検証
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CitationJudgment:
    cited: bool
    """引用を1つでも出したか。"""
    all_exist: bool
    """引用したIDがすべてプロンプトに実在したか（捏造の検出）。"""
    supported: bool
    """引用先の少なくとも1つが gold の根拠だったか。"""
    n_citations: int = 0
    n_hallucinated: int = 0


def judge_citations(
    answer: Answer,
    context_chunk_ids: Sequence[str],
    gold_chunk_ids: frozenset[str],
) -> CitationJudgment:
    """引用IDの実在性と、引用先が根拠を含むかを検証する。

    存在しないIDを引用していれば、モデルはコンテキストを見ずに
    形式だけ真似ている。**正答していても信用できない**ため、
    正誤とは別軸で必ず記録する。
    """
    citations = list(answer.citations)
    available = set(context_chunk_ids)
    hallucinated = [c for c in citations if c not in available]
    return CitationJudgment(
        cited=bool(citations),
        all_exist=bool(citations) and not hallucinated,
        supported=bool(gold_chunk_ids) and any(c in gold_chunk_ids for c in citations),
        n_citations=len(citations),
        n_hallucinated=len(hallucinated),
    )


# ----------------------------------------------------------------------
# 集計
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeRates:
    """正答 / 誤答 / 棄権の3値と、棄権の適合率・再現率。"""

    n: int
    correct: int
    incorrect: int
    correct_abstention: int
    unjustified_abstention: int

    @property
    def accuracy(self) -> float:
        """正答率。回答不能な質問への正しい棄権も正答として数える。"""
        return _ratio(self.correct + self.correct_abstention, self.n)

    @property
    def error_rate(self) -> float:
        """誤答率。**主指標はこれを一定以下に抑えたうえでの正答率**。"""
        return _ratio(self.incorrect, self.n)

    @property
    def abstention_rate(self) -> float:
        return _ratio(self.correct_abstention + self.unjustified_abstention, self.n)

    @property
    def abstention_precision(self) -> float:
        """棄権したうち、本当に回答不能だった割合。"""
        return _ratio(
            self.correct_abstention, self.correct_abstention + self.unjustified_abstention
        )

    def abstention_recall(self, n_unanswerable: int) -> float:
        """回答不能な質問のうち、棄権できた割合。

        分母は gold 側の情報なので引数で受け取る。``incorrect`` には
        「回答可能なのに誤答」も混ざっており、この型だけでは分けられない。
        """
        return _ratio(self.correct_abstention, n_unanswerable)

    def as_dict(self, *, n_unanswerable: int | None = None) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "n": self.n,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "correct_abstention": self.correct_abstention,
            "unjustified_abstention": self.unjustified_abstention,
            "accuracy": round(self.accuracy, 4),
            "error_rate": round(self.error_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "abstention_precision": round(self.abstention_precision, 4),
        }
        if n_unanswerable is not None:
            payload["abstention_recall"] = round(self.abstention_recall(n_unanswerable), 4)
        return payload


def aggregate_outcomes(judgments: Sequence[AnswerJudgment]) -> OutcomeRates:
    counts = {"correct": 0, "incorrect": 0, "correct_abstention": 0, "unjustified_abstention": 0}
    for judgment in judgments:
        counts[judgment.outcome] += 1
    return OutcomeRates(n=len(judgments), **counts)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

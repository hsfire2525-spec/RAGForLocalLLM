"""LLM非依存の下限ベースラインの挙動を固定する。

このベースラインは**棄権できることが必須**。棄権しないベースラインは
誤答率が常に最大になり、比較対象として機能しない。
"""

from __future__ import annotations

from ragforlocalllm.core.types import ChatMessage, Prompt
from ragforlocalllm.stages.generator.extractive import ExtractiveGenerator

CONTEXT = """# 参考文書

[1] （第2章 体制）

## 第2章 体制

### 2.1 情報セキュリティ基本方針の承認

情報セキュリティ基本方針は、経営者が承認する。
承認された基本方針は、社内ポータルに掲示する。

[2] （第3章 技術的対策）

### 3.1 パスワード

パスワードは12文字以上とする。

# 質問

{question}
"""


def _prompt(question: str) -> Prompt:
    return Prompt(
        messages=[ChatMessage(role="user", content=CONTEXT.format(question=question))],
        context_chunk_ids=["doc#c0001", "doc#c0002"],
        token_estimate=100,
    )


def test_finds_the_answering_sentence() -> None:
    answer = ExtractiveGenerator(max_sentences=1).generate(
        _prompt("情報セキュリティ基本方針を承認するのは誰か。")
    )
    assert answer.abstained is False
    assert "経営者が承認する" in answer.text
    assert answer.citations == ["1"]


def test_markdown_headings_are_not_returned_as_answers() -> None:
    """見出しは構造情報であり回答本体ではない。

    見出しは質問と語が重なりやすいため、除外しないと回答が見出しで埋まる。
    """
    answer = ExtractiveGenerator(max_sentences=2).generate(
        _prompt("情報セキュリティ基本方針を承認するのは誰か。")
    )
    assert "##" not in answer.text
    assert "2.1" not in answer.text


def test_abstains_when_nothing_is_relevant() -> None:
    answer = ExtractiveGenerator().generate(_prompt("四半期の売上高はいくらか。"))
    assert answer.abstained is True
    assert answer.text == "分かりません"
    assert answer.citations == []


def test_min_score_controls_abstention() -> None:
    question = "パスワードは何文字以上必要か。"
    lenient = ExtractiveGenerator(min_score=0.0).generate(_prompt(question))
    strict = ExtractiveGenerator(min_score=0.99).generate(_prompt(question))
    assert lenient.abstained is False
    assert strict.abstained is True


def test_dice_score_does_not_favour_long_sentences() -> None:
    """正規化なしの重なり数だと、無関係でも長い文が勝ってしまう。"""
    long_irrelevant = "あ" * 200
    prompt = Prompt(
        messages=[
            ChatMessage(
                role="user",
                content=(
                    "# 参考文書\n\n[1]\n\n"
                    f"{long_irrelevant}。\nパスワードは12文字以上とする。\n\n# 質問\n\n"
                    "パスワードは何文字以上必要か。"
                ),
            )
        ],
        context_chunk_ids=["doc#c1"],
    )
    answer = ExtractiveGenerator(max_sentences=1).generate(prompt)
    assert "12文字以上" in answer.text


def test_invalid_min_score_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="min_score"):
        ExtractiveGenerator(min_score=1.5)


def test_usage_records_prompt_token_estimate() -> None:
    answer = ExtractiveGenerator().generate(_prompt("パスワードは何文字以上必要か。"))
    assert answer.usage is not None
    assert answer.usage.prompt_tokens == 100

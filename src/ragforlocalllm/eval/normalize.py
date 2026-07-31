"""評価で使う文字列正規化。

引用の解決と回答の採点で**同じ正規化を共有する**。片方だけを変えると、
「引用は解決できるのに回答が不正解になる」ような説明のつかない差が出る。

日本語での注意点:

- **空白を落とす。** PDFから抽出した本文は行の折り返しや字間調整で空白が
  入る（「リスク値 ＝ 重要度 × 被害発生可能性」）。空白を残すと引用が
  解決できない
- **NFKC で全角・半角を畳む。** 「EDR」と「ＥＤＲ」、「1」と「１」は
  同じものとして扱う
- **単語分割に依存しない。** 形態素解析器を変えると数値が動くため、
  主指標は文字レベルで測る（docs/design/design.md §6.3）
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WHITESPACE = re.compile(r"\s+")

# 回答の採点でのみ落とす記号。引用の解決では落とさない
# （記号まで落とすと、別の箇所を誤って引用元と判定しうる）。
_PUNCTUATION = re.compile(
    r"[。、．，・：；！？"
    r"「」『』（）〔〕［］｛｝〈〉《》【】"
    r"\"'`()\[\]{}<>:;!?,.…‥~〜ー–—\-_/\\|＋+*=＝]"
)


def normalize_for_match(text: str) -> str:
    """引用の解決に使う正規化。NFKC + 空白除去。

    記号は残す。「リスク値＝重要度×被害発生可能性」のような式では
    記号自体が識別に効くため。
    """
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text))


def normalize_answer(text: str) -> str:
    """回答の採点に使う正規化。NFKC + 空白除去 + 記号除去 + 小文字化。

    「経営者。」と「経営者」、「EDR」と「ＥＤＲ」を同一視する。
    """
    normalized = normalize_for_match(text)
    return _PUNCTUATION.sub("", normalized).casefold()


def exact_match(prediction: str, accepted: list[str]) -> bool:
    """正規化後に受理回答のいずれかと完全一致するか。"""
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return any(pred == normalize_answer(a) for a in accepted if a.strip())


def contains_answer(prediction: str, accepted: list[str]) -> bool:
    """受理回答のいずれかを部分文字列として含むか。

    4B級モデルは「経営者です。」「承認するのは経営者です」のように
    短答に説明を付ける。EM だけで測ると、内容が正しい回答を大量に
    取りこぼす。EM とは別指標として併記する。
    """
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return any(normalize_answer(a) in pred for a in accepted if a.strip())


def char_f1(prediction: str, reference: str) -> float:
    """文字レベルF1。

    日本語では単語F1が形態素解析器に依存し、解析器を変えると数値が
    動く。比較の土台にできないため、文字レベルを主指標にする。
    """
    pred = Counter(normalize_answer(prediction))
    ref = Counter(normalize_answer(reference))
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def best_char_f1(prediction: str, accepted: list[str]) -> float:
    """受理回答のうち最も高い char F1。"""
    candidates = [a for a in accepted if a.strip()]
    if not candidates:
        return 0.0
    return max(char_f1(prediction, a) for a in candidates)


def set_f1(predicted_items: list[str], reference_items: list[str]) -> float:
    """列挙型回答の集合F1。要素は正規化して比較する。"""
    pred = {normalize_answer(i) for i in predicted_items if normalize_answer(i)}
    ref = {normalize_answer(i) for i in reference_items if normalize_answer(i)}
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    common = len(pred & ref)
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


_LIST_SEPARATORS = re.compile(r"[、,，\n・]|(?<![0-9])\.(?![0-9])")


def split_items(text: str) -> list[str]:
    """回答文を列挙の要素に分解する。

    小数点（3.5）を区切りと誤認しないよう、数字に挟まれた ``.`` は
    区切りにしない。
    """
    return [part.strip() for part in _LIST_SEPARATORS.split(text) if part.strip()]

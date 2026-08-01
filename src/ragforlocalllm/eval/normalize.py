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


def windowed_char_f1(prediction: str, reference: str) -> float:
    """予測の中で最も gold に一致する部分の char F1。

    **char F1 は冗長さを強く罰する。** 4B級モデルは根拠や補足を添えて
    答えるため、内容が正しくても予測が gold の3〜4倍の長さになり、
    適合率が落ちて不正解と判定される。実測では以下がすべて誤答扱いだった:

        gold: 脅威の起こりやすさと脆弱性のつけ込みやすさの2つの数値から算出する
        pred: 「脅威の起こりやすさ」と「脆弱性のつけ込みやすさ」の2つの数値から
              算出されます。これは、脅威が脆弱性を利用して…

    そこで gold と同じ長さの窓を予測上で滑らせ、最も一致する位置で測る。
    「gold の内容が、gold と同じ密度でどこかに現れるか」を見ていることになる。
    冗長さの上限を別途パラメータで持つより、概念が1つで済む。
    """
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0
    if len(pred) <= len(ref):
        return char_f1(prediction, reference)

    ref_counts = Counter(ref)
    width = len(ref)
    step = max(1, width // 10)
    best = 0.0
    for start in range(0, len(pred) - width + 1, step):
        window = Counter(pred[start : start + width])
        common = sum((window & ref_counts).values())
        if common:
            # 窓幅 = gold 長なので適合率と再現率は同じ分母になる
            best = max(best, common / width)
    return best


def best_windowed_char_f1(prediction: str, accepted: list[str]) -> float:
    candidates = [a for a in accepted if a.strip()]
    if not candidates:
        return 0.0
    return max(windowed_char_f1(prediction, a) for a in candidates)


def set_f1(predicted_items: list[str], reference_items: list[str]) -> float:
    """列挙型回答の集合F1。要素は**包含**で対応付ける。

    完全一致で突き合わせてはいけない。モデルは列挙を裸の単語では返さず、
    文の中に埋め込んで答える:

        gold: 実施している、一部実施している、実施していない、わからない
        pred: 「実施している 4点」「一部実施している 2点」…

    要素の完全一致を要求すると、この**内容として正しい回答が 0 点**になる。
    実際に4B級モデルで測ったところ、列挙型の正答率が 0.20 まで落ち、
    自動採点が人手判定と大きく乖離した。

    そこで「gold の要素が予測のどこかに現れるか」で対応付ける。
    再現率は gold 側、適合率は予測側の要素がどれかの gold に対応したかで数える
    （冗長な列挙に歯止めをかけるため適合率も見る）。
    """
    pred = [normalize_answer(i) for i in predicted_items]
    ref = [normalize_answer(i) for i in reference_items]
    pred = [p for p in pred if p]
    ref = [r for r in ref if r]
    if not pred or not ref:
        return 1.0 if not pred and not ref else 0.0

    def linked(a: str, b: str) -> bool:
        return a in b or b in a

    matched_ref = sum(1 for r in ref if any(linked(r, p) for p in pred))
    matched_pred = sum(1 for p in pred if any(linked(r, p) for r in ref))
    if matched_ref == 0:
        return 0.0
    recall = matched_ref / len(ref)
    precision = matched_pred / len(pred)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_LIST_SEPARATORS = re.compile(r"[、,，\n・]|(?<![0-9])\.(?![0-9])")


def split_items(text: str) -> list[str]:
    """回答文を列挙の要素に分解する。

    小数点（3.5）を区切りと誤認しないよう、数字に挟まれた ``.`` は
    区切りにしない。
    """
    return [part.strip() for part in _LIST_SEPARATORS.split(text) if part.strip()]

"""検索段のメトリクス。

**正解集合の意味づけに注意が要る。** 解決器は1つの Evidence に対して
複数のチャンクIDを返す（オーバーラップした複数チャンクが同じ引用文を
含む、ページ指定で解決した、など）。これらは「どれか1つ取れれば良い
**選択肢**」であって、全部取るべきものではない。

一方、**Evidence が複数ある質問（multi-hop）では、すべての Evidence を
取る必要がある**。したがって:

- 1つの Evidence 内 … チャンクIDは選択肢（いずれか1つで充足）
- Evidence 間 … すべて充足して初めて満点

この2層を潰して素朴に ``|取得 ∩ 正解| / |正解|`` を計算すると、
ページ指定で解決した質問の正解集合だけが巨大になり、スコアが
検索性能ではなく gold の書き方を反映してしまう。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ragforlocalllm.eval.resolve import GoldResolution


def evidence_recall_at_k(resolution: GoldResolution, retrieved: Sequence[str], k: int) -> float:
    """上位k件で充足できた Evidence の割合。

    multi-hop では「2つの根拠のうち1つだけ取れた」を 0.5 として扱う。
    """
    evidences = [m for m in resolution.matches if m.resolved]
    if not evidences:
        return float("nan")
    top = set(retrieved[:k])
    covered = sum(1 for m in evidences if top & m.chunk_ids)
    return covered / len(evidences)


def hit_at_k(resolution: GoldResolution, retrieved: Sequence[str], k: int) -> float:
    """上位k件に根拠が1つでも入っていれば 1。"""
    gold = resolution.chunk_ids
    if not gold:
        return float("nan")
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def reciprocal_rank(resolution: GoldResolution, retrieved: Sequence[str]) -> float:
    """最初に現れた根拠チャンクの順位の逆数。"""
    gold = resolution.chunk_ids
    if not gold:
        return float("nan")
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(resolution: GoldResolution, retrieved: Sequence[str], k: int) -> float:
    """二値適合度の nDCG@k。

    理想順位は「充足できる Evidence の数」だけ先頭に並んだ状態とする。
    正解集合の大きさ（＝解決したチャンク数）を理想値に使うと、
    ページ指定で解決した質問が構造的に低く出てしまう。
    """
    gold = resolution.chunk_ids
    if not gold:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in gold
    )
    n_ideal = min(k, max(1, sum(1 for m in resolution.matches if m.resolved)))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return dcg / idcg if idcg else 0.0


def context_hit(resolution: GoldResolution, contexts: Sequence[str]) -> float:
    """根拠がプロンプトまで生き残ったか。

    **hit@k との差が、コンテキスト予算で捨てられた分。** 検索が正解を
    取れていても、予算に収まらず落ちていれば当然答えられない。
    この2つを分けて測らないと、「検索は当たっているのに答えられない」
    という誤った結論に至る。

    実測では 42問中38問で予算により計69チャンクが落ちており、
    ハイブリッド検索の改善が正答率に十分つながらない原因になっていた
    （docs/design/design.md §9 Phase 3-1）。
    """
    gold = resolution.chunk_ids
    if not gold:
        return float("nan")
    return 1.0 if gold & set(contexts) else 0.0


def context_precision(resolution: GoldResolution, contexts: Sequence[str]) -> float:
    """プロンプトに入れたコンテキストのうち、根拠だったものの割合。

    低いほど無関係な文脈でトークン予算を消費している。4B級モデルでは
    ノイズが直接精度を下げるため、recall と併せて見る必要がある。
    """
    if not contexts:
        return float("nan")
    gold = resolution.chunk_ids
    if not gold:
        return float("nan")
    return sum(1 for c in contexts if c in gold) / len(contexts)

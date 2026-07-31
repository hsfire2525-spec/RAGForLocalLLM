"""gold の根拠（ページ + 引用文）を実際のチャンク集合へ解決する。

**gold を chunk_id でアンカーしない**ため、この解決器が評価の要になる。
chunk_id は Chunker の設定に依存するので、チャンク戦略を変えるたびに
gold が無効になり、本リポジトリの主目的と両立しない
（docs/design/design.md §6.2）。

副産物として **gold quote resolvability rate（引用解決率）** が得られる。
引用文がどのチャンクにも見つからなければ、Loader か Chunker が情報を
壊している。しかもこの2つは切り分けられる:

- 文書本文（チャンクを連結したもの）にも無い → **Loader の抽出漏れ**
  （縦書きの混入で文が分断された、表のセルが落ちた、等）
- 本文にはあるがどのチャンクにも収まらない → **Chunker の境界分断**
  （チャンク境界が引用文の途中に来た）

前者は抽出器の比較に、後者はチャンクサイズ・オーバーラップの調整に
直結する。同じ「解決できない」でも打ち手が違うため、必ず区別して記録する。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from ragforlocalllm.core.types import Chunk
from ragforlocalllm.eval.dataset import Evidence, GoldDataset, GoldItem
from ragforlocalllm.eval.normalize import normalize_for_match

Anchor = Literal["quote", "page", "section_path"]
QuoteStatus = Literal["resolved", "split_across_chunks", "missing_from_corpus"]


@dataclass(frozen=True)
class EvidenceMatch:
    """1つの Evidence の解決結果。"""

    evidence: Evidence
    chunk_ids: frozenset[str]
    anchor: Anchor | None
    """どのアンカーで解決したか。解決できなければ None。"""
    quote_status: QuoteStatus | None = None
    """``quote`` を持つ Evidence のみ設定される。引用解決率の集計に使う。"""

    @property
    def resolved(self) -> bool:
        return bool(self.chunk_ids)


@dataclass(frozen=True)
class GoldResolution:
    """gold 1件の解決結果。"""

    qid: str
    matches: tuple[EvidenceMatch, ...] = ()

    @property
    def chunk_ids(self) -> frozenset[str]:
        """根拠となるチャンクIDの和集合。検索メトリクスの正解集合。"""
        if not self.matches:
            return frozenset()
        return frozenset().union(*(m.chunk_ids for m in self.matches))

    @property
    def measurable(self) -> bool:
        """検索メトリクスを計算できるか。

        根拠を1つも解決できなかった質問を「recall 0」として集計すると、
        検索性能の低さと gold の解決失敗が混ざってしまう。除外して数える。
        """
        return bool(self.chunk_ids)


@dataclass
class ResolutionReport:
    """データセット全体の解決結果と、その健全性。"""

    resolutions: dict[str, GoldResolution] = field(default_factory=dict)

    def __getitem__(self, qid: str) -> GoldResolution:
        return self.resolutions[qid]

    def quote_counts(self) -> dict[QuoteStatus, int]:
        counts: dict[QuoteStatus, int] = {
            "resolved": 0,
            "split_across_chunks": 0,
            "missing_from_corpus": 0,
        }
        for resolution in self.resolutions.values():
            for match in resolution.matches:
                if match.quote_status is not None:
                    counts[match.quote_status] += 1
        return counts

    @property
    def quote_resolvability(self) -> float | None:
        """引用文を持つ Evidence のうち、単一チャンクに解決できた割合。

        引用文を持つ Evidence が無ければ None（0.0 と区別する）。
        """
        counts = self.quote_counts()
        total = sum(counts.values())
        if total == 0:
            return None
        return counts["resolved"] / total

    def unmeasurable_qids(self) -> list[str]:
        return sorted(qid for qid, r in self.resolutions.items() if not r.measurable)

    def summary(self) -> dict[str, object]:
        counts = self.quote_counts()
        resolvability = self.quote_resolvability
        return {
            "n_items": len(self.resolutions),
            "n_measurable": sum(1 for r in self.resolutions.values() if r.measurable),
            "unmeasurable_qids": self.unmeasurable_qids(),
            "quote_resolvability": None if resolvability is None else round(resolvability, 4),
            "quote_status_counts": counts,
        }


class Resolver:
    """チャンク集合に対して gold の根拠を解決する。

    正規化済みテキストを一度だけ作って使い回す。データセット全件 ×
    チャンク数の総当たりになるため、ここを毎回正規化すると遅い。
    """

    def __init__(self, chunks: Sequence[Chunk], *, fallback_to_page: bool = True) -> None:
        self._chunks = list(chunks)
        self._normalized = [normalize_for_match(c.text) for c in self._chunks]
        self.fallback_to_page = fallback_to_page
        """引用文が解決できなかったとき、ページで代替するか。

        既定で有効。これがないと引用解決に失敗した質問の検索メトリクスが
        まるごと欠測になり、**Loader の問題が検索性能の問題に見えてしまう**。
        代替したことは ``quote_status`` に残るので、解決率の集計は汚れない。
        """
        # 文書ごとに連結した本文。チャンク境界での分断と、抽出漏れを
        # 切り分けるために使う。文書をまたいだ偽の一致を避けるため
        # doc_id ごとに分ける。
        self._by_doc: dict[str, str] = {}
        for chunk, text in zip(self._chunks, self._normalized, strict=True):
            self._by_doc[chunk.doc_id] = self._by_doc.get(chunk.doc_id, "") + text

    # ------------------------------------------------------------------

    def resolve_item(self, item: GoldItem) -> GoldResolution:
        matches = tuple(self._resolve_evidence(e) for e in item.evidence)
        return GoldResolution(qid=item.qid, matches=matches)

    def resolve_dataset(self, gold: GoldDataset) -> ResolutionReport:
        return ResolutionReport(
            resolutions={item.qid: self.resolve_item(item) for item in gold if item.answerable}
        )

    # ------------------------------------------------------------------

    def _resolve_evidence(self, evidence: Evidence) -> EvidenceMatch:
        if evidence.quote:
            return self._resolve_quote(evidence)
        if evidence.page is not None:
            ids = self._chunks_on_page(evidence.page)
            return EvidenceMatch(evidence, ids, "page" if ids else None)
        if evidence.section_path:
            ids = self._chunks_in_section(evidence.section_path)
            return EvidenceMatch(evidence, ids, "section_path" if ids else None)
        return EvidenceMatch(evidence, frozenset(), None)

    def _resolve_quote(self, evidence: Evidence) -> EvidenceMatch:
        needle = normalize_for_match(evidence.quote or "")
        hits = {
            chunk.chunk_id
            for chunk, text in zip(self._chunks, self._normalized, strict=True)
            if needle and needle in text
        }
        if hits:
            # ページも指定されていれば、そのページのチャンクに絞る。
            # 同じ文言が複数箇所にある場合の誤解決を防ぐ。
            if evidence.page is not None:
                on_page = hits & self._chunks_on_page(evidence.page)
                if on_page:
                    hits = on_page
            return EvidenceMatch(evidence, frozenset(hits), "quote", "resolved")

        status: QuoteStatus = (
            "split_across_chunks"
            if any(needle in body for body in self._by_doc.values())
            else "missing_from_corpus"
        )
        if self.fallback_to_page and evidence.page is not None:
            ids = self._chunks_on_page(evidence.page)
            if ids:
                return EvidenceMatch(evidence, ids, "page", status)
        return EvidenceMatch(evidence, frozenset(), None, status)

    def _chunks_on_page(self, page: int) -> frozenset[str]:
        return frozenset(c.chunk_id for c in self._chunks if _covers_page(c, page))

    def _chunks_in_section(self, section_path: str) -> frozenset[str]:
        needle = normalize_for_match(section_path)
        return frozenset(
            c.chunk_id
            for c in self._chunks
            if c.section_path and needle in normalize_for_match(c.section_path)
        )


def _covers_page(chunk: Chunk, page: int) -> bool:
    """チャンクが指定ページを含むか。

    節単位で切り出した Document は複数ページにまたがるため、
    ``page`` 単体ではなく ``page_start``〜``page_end`` の範囲で見る。
    """
    start = chunk.metadata.get("page_start", chunk.metadata.get("page"))
    end = chunk.metadata.get("page_end", start)
    if start is None:
        return False
    return int(start) <= page <= int(end if end is not None else start)

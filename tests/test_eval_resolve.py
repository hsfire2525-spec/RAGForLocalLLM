"""gold引用 → チャンクID の解決器。

**この解決器が Loader/Chunker の比較を成立させている。** gold を
chunk_id でアンカーしていたら、チャンク設定を変えるたびに gold を
作り直すことになる。
"""

from __future__ import annotations

from ragforlocalllm.core.types import Chunk
from ragforlocalllm.eval.dataset import Evidence, GoldDataset, GoldItem
from ragforlocalllm.eval.resolve import Resolver


def chunk(chunk_id: str, text: str, *, page: int | None = None, **meta: object) -> Chunk:
    metadata: dict[str, object] = dict(meta)
    if page is not None:
        metadata.setdefault("page", page)
        metadata.setdefault("page_start", page)
        metadata.setdefault("page_end", page)
    return Chunk(chunk_id=chunk_id, doc_id="doc", text=text, metadata=metadata)


def item(qid: str, evidence: list[Evidence]) -> GoldItem:
    return GoldItem(qid=qid, question="q", answer="a", evidence=evidence)


# ----------------------------------------------------------------------


def test_quote_resolves_across_line_breaks_and_spacing() -> None:
    """引用文はチャンク内で折り返されている。正規化しないと解決できない。"""
    chunks = [chunk("c1", "基本方針は\n経営者が 承認し、周知する。", page=23)]
    resolver = Resolver(chunks)
    resolution = resolver.resolve_item(item("q1", [Evidence(page=23, quote="経営者が承認し")]))

    assert resolution.chunk_ids == {"c1"}
    assert resolution.matches[0].anchor == "quote"
    assert resolution.matches[0].quote_status == "resolved"


def test_page_narrows_an_ambiguous_quote() -> None:
    """同じ文言が複数箇所にある場合、ページで絞る。"""
    chunks = [chunk("c1", "対策を実施する", page=10), chunk("c2", "対策を実施する", page=40)]
    resolver = Resolver(chunks)
    resolution = resolver.resolve_item(item("q1", [Evidence(page=40, quote="対策を実施する")]))
    assert resolution.chunk_ids == {"c2"}


def test_quote_split_across_chunks_is_distinguished_from_extraction_loss() -> None:
    """同じ「解決できない」でも打ち手が違うため必ず区別する。

    境界で切れた → チャンクサイズ・オーバーラップの調整。
    本文にも無い → Loader の抽出漏れ。
    """
    chunks = [chunk("c1", "リスク値＝重要", page=47), chunk("c2", "度×被害発生可能性", page=47)]
    resolver = Resolver(chunks, fallback_to_page=False)
    resolution = resolver.resolve_item(
        item("q1", [Evidence(page=47, quote="リスク値＝重要度×被害発生可能性")])
    )

    assert resolution.matches[0].quote_status == "split_across_chunks"
    assert not resolution.matches[0].resolved


def test_quote_missing_from_corpus_is_reported() -> None:
    chunks = [chunk("c1", "まったく別の本文", page=1)]
    resolver = Resolver(chunks, fallback_to_page=False)
    resolution = resolver.resolve_item(item("q1", [Evidence(page=1, quote="存在しない文言")]))
    assert resolution.matches[0].quote_status == "missing_from_corpus"


def test_quote_does_not_match_across_documents() -> None:
    """文書をまたいだ連結で偽の一致を作らない。"""
    chunks = [
        Chunk(chunk_id="a1", doc_id="docA", text="リスク値＝重要", metadata={"page": 1}),
        Chunk(chunk_id="b1", doc_id="docB", text="度×被害発生可能性", metadata={"page": 2}),
    ]
    resolver = Resolver(chunks, fallback_to_page=False)
    resolution = resolver.resolve_item(
        item("q1", [Evidence(page=1, quote="リスク値＝重要度×被害発生可能性")])
    )
    assert resolution.matches[0].quote_status == "missing_from_corpus"


def test_page_fallback_keeps_retrieval_measurable_but_records_the_failure() -> None:
    """引用解決の失敗を検索性能の低さに見せかけない。

    フォールバックしないと、その質問の検索メトリクスが欠測になり
    「Loader の問題」が「検索の問題」として集計される。
    """
    chunks = [chunk("c1", "リスク値＝重要", page=47), chunk("c2", "度×被害", page=47)]
    resolver = Resolver(chunks, fallback_to_page=True)
    resolution = resolver.resolve_item(
        item("q1", [Evidence(page=47, quote="リスク値＝重要度×被害")])
    )

    assert resolution.measurable  # 検索メトリクスは計算できる
    assert resolution.matches[0].anchor == "page"
    assert resolution.matches[0].quote_status == "split_across_chunks"  # 失敗は記録される


def test_page_range_covers_multi_page_sections() -> None:
    """節単位の Document は複数ページにまたがる。"""
    chunks = [chunk("c1", "本文", page_start=8, page_end=10)]
    resolver = Resolver(chunks)
    resolution = resolver.resolve_item(item("q1", [Evidence(page=9)]))
    assert resolution.chunk_ids == {"c1"}


def test_section_path_anchor() -> None:
    chunks = [chunk("c1", "本文", page=1, section_path="第1 部 経営者編 > 2 経営者が負う責任")]
    resolver = Resolver(chunks)
    resolution = resolver.resolve_item(item("q1", [Evidence(section_path="経営者が負う責任")]))
    assert resolution.chunk_ids == {"c1"}


# ----------------------------------------------------------------------


def test_report_counts_resolvability_and_excludes_unanswerable() -> None:
    chunks = [chunk("c1", "経営者が承認し", page=23)]
    gold = GoldDataset(
        items=[
            item("q1", [Evidence(page=23, quote="経営者が承認し")]),
            item("q2", [Evidence(page=23, quote="どこにも無い")]),
            GoldItem(qid="q3", question="q", answer="分かりません", answerable=False),
        ]
    )
    report = Resolver(chunks, fallback_to_page=False).resolve_dataset(gold)

    # 回答不能な質問には根拠が無い。解決の対象外。
    assert set(report.resolutions) == {"q1", "q2"}
    assert report.quote_resolvability == 0.5
    assert report.unmeasurable_qids() == ["q2"]
    assert report.quote_counts()["missing_from_corpus"] == 1


def test_resolvability_is_none_when_no_quotes_are_used() -> None:
    """引用文なしのアンカーも許すため、0.0 と「該当なし」を区別する。"""
    chunks = [chunk("c1", "本文", page=1)]
    gold = GoldDataset(items=[item("q1", [Evidence(page=1)])])
    report = Resolver(chunks).resolve_dataset(gold)
    assert report.quote_resolvability is None

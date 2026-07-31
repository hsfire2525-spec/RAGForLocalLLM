"""個別コンポーネントの単体テスト。

特に、精度に直結するが「動くが精度だけ落ちる」形で失敗しやすい
挙動（埋め込みのプレフィックス規約、日本語の分割境界、
lost-in-the-middle 並べ替え）を固定する。
"""

from __future__ import annotations

import numpy as np

from ragforlocalllm.core.tokens import CharHeuristicCounter
from ragforlocalllm.core.types import Chunk, Document, QueryState, ScoredChunk
from ragforlocalllm.stages.chunker.fixed import FixedChunker, RecursiveJapaneseChunker
from ragforlocalllm.stages.embedder.hashing import HashingEmbedder
from ragforlocalllm.stages.indexer.numpy_flat import NumpyFlatIndexer
from ragforlocalllm.stages.post_generation.checks import (
    AbstainWithoutCitation,
    ParseCitations,
)
from ragforlocalllm.stages.post_retrieval.basic import Dedupe, ReorderLostInMiddle
from ragforlocalllm.stages.query_transform.rules import (
    NormalizeTransform,
    SynonymExpandTransform,
)

DOC = Document(doc_id="d1", text="あいうえお。かきくけこ。さしすせそ。たちつてと。")


def _scored(chunk_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=chunk_id, doc_id="d1", text=text), score=score)


# ----------------------------------------------------------------------
# Chunker
# ----------------------------------------------------------------------


def test_fixed_chunker_respects_size_and_overlap() -> None:
    doc = Document(doc_id="d", text="あ" * 250)
    chunks = FixedChunker(chunk_size=100, overlap=20).split(doc)
    assert all(len(c.text) <= 100 for c in chunks)
    assert len(chunks) >= 3
    assert chunks[0].metadata["char_start"] == 0


def test_recursive_ja_chunker_keeps_sentence_terminators() -> None:
    """句点で切っても文末の「。」が失われないこと。"""
    chunks = RecursiveJapaneseChunker(chunk_size=12, overlap=0).split(DOC)
    assert chunks
    joined = "".join(c.text for c in chunks)
    assert joined.count("。") == DOC.text.count("。")


def test_recursive_ja_chunker_records_char_offsets() -> None:
    """評価アンカーの解決に使うため、元テキスト上の位置を持つこと。"""
    chunks = RecursiveJapaneseChunker(chunk_size=12, overlap=0).split(DOC)
    for chunk in chunks:
        start = chunk.metadata["char_start"]
        assert DOC.text[start : start + len(chunk.text)] == chunk.text


def test_empty_document_yields_no_chunks() -> None:
    empty = Document(doc_id="e", text="")
    assert RecursiveJapaneseChunker().split(empty) == []
    assert FixedChunker().split(empty) == []


def test_chunks_inherit_document_metadata() -> None:
    """**page / section_path が落ちると gold のアンカーが解決できない。**

    節は複数ページにまたがるため、チャンク単体では正確なページを
    特定できない。範囲（page_start〜page_end）をそのまま持たせ、
    解決器が範囲で判定する。
    """
    doc = Document(
        doc_id="d1",
        text="あ" * 900,
        metadata={
            "page": 23,
            "page_start": 23,
            "page_end": 25,
            "section_path": "第1 部 > 2 経営者が負う責任",
            "source": "x.pdf",
        },
    )
    chunkers = (
        FixedChunker(chunk_size=300, overlap=50),
        RecursiveJapaneseChunker(chunk_size=300, overlap=50),
    )
    for chunker in chunkers:
        chunks = chunker.split(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.page == 23
            assert chunk.section_path == "第1 部 > 2 経営者が負う責任"
            assert chunk.metadata["page_end"] == 25
            assert chunk.metadata["source"] == "x.pdf"
            # チャンク固有のキーは文書側の値に上書きされない
            assert chunk.metadata["n_chars"] == len(chunk.text)


# ----------------------------------------------------------------------
# Embedder / Index
# ----------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed_passages(["情報セキュリティ"])
    b = embedder.embed_passages(["情報セキュリティ"])
    assert np.allclose(a, b)
    assert a.shape == (1, 64)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_numpy_flat_index_ranks_exact_match_first() -> None:
    embedder = HashingEmbedder(dim=128)
    chunks = [
        Chunk(chunk_id="c1", doc_id="d", text="パスワードは12文字以上とする"),
        Chunk(chunk_id="c2", doc_id="d", text="バックアップは7世代分を保管する"),
    ]
    vectors = embedder.embed_passages([c.text for c in chunks])
    index = NumpyFlatIndexer().build(chunks, vectors)

    query = embedder.embed_queries(["パスワードは12文字以上とする"])
    hits = index.search(query, top_k=2)[0]
    assert hits[0][0] == "c1"


def test_numpy_flat_index_roundtrip(tmp_path) -> None:
    embedder = HashingEmbedder(dim=32)
    chunks = [Chunk(chunk_id="c1", doc_id="d", text="テスト")]
    vectors = embedder.embed_passages(["テスト"])
    indexer = NumpyFlatIndexer()
    indexer.build(chunks, vectors).save(tmp_path)

    loaded = indexer.load(tmp_path)
    assert len(loaded) == 1
    assert loaded.get("c1") is not None


# ----------------------------------------------------------------------
# QueryTransform
# ----------------------------------------------------------------------


def test_normalize_transform_applies_nfkc() -> None:
    assert NormalizeTransform().transform("ＰＡＳＳＷＯＲＤ　１２") == ["PASSWORD 12"]


def test_synonym_expand_adds_variants() -> None:
    transform = SynonymExpandTransform(synonyms={"標的型攻撃": ["スピアフィッシング"]})
    out = transform.transform("標的型攻撃の対策は")
    assert out[0] == "標的型攻撃の対策は"
    assert "スピアフィッシングの対策は" in out


def test_synonym_expand_respects_max_queries() -> None:
    transform = SynonymExpandTransform(synonyms={"A": ["B", "C", "D", "E", "F"]}, max_queries=3)
    assert len(transform.transform("Aの話")) <= 3


# ----------------------------------------------------------------------
# PostRetrieval
# ----------------------------------------------------------------------


def test_reorder_places_top_items_at_both_ends() -> None:
    state = QueryState.new("q")
    state.contexts = [_scored(f"c{i}", f"text{i}", score=1.0 - i * 0.1) for i in range(5)]
    reordered = ReorderLostInMiddle().process(state).contexts

    ids = [c.chunk.chunk_id for c in reordered]
    # 最上位は先頭、2番目は末尾に置かれる
    assert ids[0] == "c0"
    assert ids[-1] == "c1"
    assert len(ids) == 5


def test_dedupe_removes_near_duplicates_keeping_higher_score() -> None:
    state = QueryState.new("q")
    state.contexts = [
        _scored("low", "パスワードは12文字以上とする", 0.5),
        _scored("high", "パスワードは12文字以上とする", 0.9),
        _scored("other", "バックアップは7世代分を保管する", 0.7),
    ]
    kept = Dedupe(threshold=0.9).process(state).contexts
    ids = {c.chunk.chunk_id for c in kept}
    assert ids == {"high", "other"}


# ----------------------------------------------------------------------
# PostGeneration
# ----------------------------------------------------------------------


def test_parse_citations_maps_numbers_to_chunk_ids() -> None:
    from ragforlocalllm.core.types import Answer, ChatMessage, Prompt, StageTrace

    state = QueryState.new("q")
    state.prompt = Prompt(
        messages=[ChatMessage(role="user", content="x")],
        context_chunk_ids=["a#c1", "a#c2", "a#c3"],
    )
    state.answer = Answer(text="経営者です [1] [3]")
    state.trace.append(StageTrace(stage="generator", impl="x", duration_ms=1.0))

    ParseCitations().process(state)
    assert state.answer.citations == ["a#c1", "a#c3"]


def test_parse_citations_flags_out_of_range_reference() -> None:
    from ragforlocalllm.core.types import Answer, ChatMessage, Prompt, StageTrace

    state = QueryState.new("q")
    state.prompt = Prompt(
        messages=[ChatMessage(role="user", content="x")], context_chunk_ids=["a#c1"]
    )
    state.answer = Answer(text="根拠は [9] です")
    state.trace.append(StageTrace(stage="generator", impl="x", duration_ms=1.0))

    ParseCitations().process(state)
    assert state.answer.citations == []
    assert state.trace[-1].info["invalid_citations"] == ["9"]


def test_abstain_without_citation_replaces_answer() -> None:
    from ragforlocalllm.core.types import Answer, ChatMessage, Prompt

    state = QueryState.new("q")
    state.prompt = Prompt(messages=[ChatMessage(role="user", content="x")])
    state.answer = Answer(text="たぶん経営者です")

    AbstainWithoutCitation().process(state)
    assert state.answer.abstained is True
    assert state.answer.text == "分かりません"
    assert state.answer.raw_text == "たぶん経営者です"


# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------


def test_char_heuristic_counter_is_conservative_for_japanese() -> None:
    counter = CharHeuristicCounter()
    assert counter.method == "char_heuristic"
    japanese = counter.count("情報セキュリティ")  # 8文字
    ascii_text = counter.count("information security")  # 20文字
    # 日本語のほうが文字あたりのトークン数が多く見積もられる
    assert japanese / 8 > ascii_text / 20

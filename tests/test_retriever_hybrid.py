"""BM25・ハイブリッド検索の検証。

**動機は「定番だから」ではない。** ベースラインの実測で
「表4は何を示した表か」のような参照質問が dense 検索で1件も取れず、
table 型の正答率が 0.50 に落ちていた（docs/design/design.md §9 Phase 2）。
字面で拾える経路を足すことがここでの目的。
"""

from __future__ import annotations

import numpy as np
import pytest

from ragforlocalllm.core.registry import build
from ragforlocalllm.core.types import Chunk
from ragforlocalllm.stages.indexer.numpy_flat import NumpyFlatIndexer
from ragforlocalllm.stages.retriever.hybrid import HybridRetriever
from ragforlocalllm.stages.retriever.sparse import BM25Retriever
from ragforlocalllm.stages.retriever.tokenize import (
    CharNgramTokenizer,
    build_tokenizer,
)

CORPUS = [
    ("c1", "【表4】自社診断のための25項目。診断内容を一覧にしています。"),
    ("c2", "【表1】本ガイドラインの全体構成。第1部と第2部、付録から成ります。"),
    ("c3", "情報セキュリティ基本方針は経営者が承認し、従業員に周知します。"),
    ("c4", "多要素認証は、知っているもの、持っているもの、本人自身に関するものを用います。"),
]


def make_index() -> object:
    chunks = [
        Chunk(chunk_id=cid, doc_id="doc", text=text, metadata={"page": i + 1})
        for i, (cid, text) in enumerate(CORPUS)
    ]
    # 検索経路を分離するため、密ベクトルは意図的に無情報にする
    vectors = np.zeros((len(chunks), 4), dtype=np.float32)
    vectors[:, 0] = 1.0
    return NumpyFlatIndexer().build(chunks, vectors)


class ConstantEmbedder:
    """すべてのクエリを同じ向きに埋め込む。dense 側を無力化する。"""

    dim = 4

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.embed_queries(texts)


# ----------------------------------------------------------------------
# トークナイザ
# ----------------------------------------------------------------------


def test_char_ngram_keeps_reference_tokens_intact() -> None:
    """**「表4」が割れないことが要点。**

    形態素解析だと「表」「4」に分かれて弁別力を失うが、bi-gram なら
    「表4」がそのまま特徴になる。dense が苦手な参照質問はここで拾う。
    """
    tokens = CharNgramTokenizer().tokenize("表4")
    assert "表4" in tokens


def test_char_ngram_folds_width_and_case() -> None:
    assert CharNgramTokenizer().tokenize("ＥＤＲ") == CharNgramTokenizer().tokenize("edr")


def test_char_ngram_does_not_span_whitespace() -> None:
    """空白は語の境界。またいで n-gram を作ると偽の語ができる。"""
    assert "あい" not in CharNgramTokenizer().tokenize("あ い")


def test_char_ngram_keeps_short_segments() -> None:
    assert CharNgramTokenizer(n=2).tokenize("表") == ["表"]


def test_unknown_tokenizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="char_ngram"):
        build_tokenizer("mecab")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------


def test_bm25_finds_the_reference_question_dense_cannot() -> None:
    """dense が構造的に落とす質問を BM25 が拾えること。"""
    retriever = BM25Retriever(index=make_index(), top_k=2)
    hits = retriever.retrieve(["表4は何を示した表か。"], 2)
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].provenance["bm25"] > 0


def test_bm25_registered_and_needs_no_embedder() -> None:
    """**BM25 は埋め込み器を必要としない。**

    合成する側が子ごとに必要な依存を知らなくて済むよう、
    レジストリは受け取ると宣言された依存だけを渡す。
    """
    retriever = build(
        "retriever", {"type": "sparse"}, embedder=ConstantEmbedder(), index=make_index()
    )
    assert isinstance(retriever, BM25Retriever)


def test_bm25_records_the_tokenizer() -> None:
    """同じ BM25 でも分割方式が違えば別物。ランレコードに残す。"""
    info = BM25Retriever(index=make_index(), tokenizer="char_ngram").describe()
    assert info["tokenizer"] == "char_ngram"
    assert info["n"] == 2


def test_bm25_handles_empty_queries() -> None:
    assert BM25Retriever(index=make_index()).retrieve([], 5) == []


# ----------------------------------------------------------------------
# RRF による統合
# ----------------------------------------------------------------------


def hybrid(**kwargs: object) -> HybridRetriever:
    return HybridRetriever(embedder=ConstantEmbedder(), index=make_index(), **kwargs)  # type: ignore[arg-type]


def test_hybrid_recovers_what_dense_alone_misses() -> None:
    """dense を無情報にしても、BM25 経由で正解が上位に来ること。"""
    hits = hybrid(top_k=2).retrieve(["表4は何を示した表か。"], 2)
    assert hits[0].chunk.chunk_id == "c1"


def test_hybrid_keeps_each_retriever_score_and_rank() -> None:
    """どちらが効いたかを事後に分析できないと是非を判断できない。"""
    hits = hybrid(top_k=2).retrieve(["表4は何を示した表か。"], 2)
    provenance = hits[0].provenance
    assert "rrf" in provenance
    assert "sparse" in provenance
    assert "sparse_rank" in provenance


def test_rrf_uses_ranks_not_raw_scores() -> None:
    """**スコアを足してはいけない。**

    dense のコサインは上位が密集し、BM25 は非有界。尺度が違うものを
    加重和すると、重み調整がスケール合わせの作業になる。
    RRF は順位のみを使うので、スコアを何倍しても結果が変わらない。
    """

    class Scaled:
        def __init__(self, factor: float) -> None:
            self.factor = factor

        def retrieve(self, queries: list[str], top_k: int) -> list:
            base = BM25Retriever(index=make_index(), top_k=top_k).retrieve(queries, top_k)
            for item in base:
                object.__setattr__(item, "score", item.score * self.factor)
            return base

    query = ["表4は何を示した表か。"]
    plain = HybridRetriever(embedder=ConstantEmbedder(), index=make_index(), top_k=3)
    plain.children = [Scaled(1.0)]  # type: ignore[list-item]
    plain.specs = [{"type": "sparse", "top_k": 4}]
    plain.weights = [1.0]

    scaled = HybridRetriever(embedder=ConstantEmbedder(), index=make_index(), top_k=3)
    scaled.children = [Scaled(1000.0)]  # type: ignore[list-item]
    scaled.specs = [{"type": "sparse", "top_k": 4}]
    scaled.weights = [1.0]

    assert [h.chunk.chunk_id for h in plain.retrieve(query, 3)] == [
        h.chunk.chunk_id for h in scaled.retrieve(query, 3)
    ]


def test_weights_must_match_the_retrievers() -> None:
    with pytest.raises(ValueError, match="weights"):
        hybrid(weights=[1.0])


def test_empty_retrievers_are_rejected() -> None:
    with pytest.raises(ValueError, match="retrievers"):
        hybrid(retrievers=[])


def test_rrf_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        hybrid(rrf_k=0)


def test_hybrid_describes_its_children() -> None:
    info = hybrid().describe()
    assert info["fusion"] == "rrf"
    assert len(info["children"]) == 2

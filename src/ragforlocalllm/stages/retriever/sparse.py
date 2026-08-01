"""BM25 による語彙検索。

**dense 検索が構造的に苦手な質問がある。** ベースラインの実測では
「表4は何を示した表か」「付録6は何のサンプルか」のような参照質問で
根拠を1件も取れておらず、table 型の正答率が 0.50 に落ちていた
（docs/design/design.md §9 Phase 2）。質問文に意味的な手がかりが乏しく、
埋め込みが効かないためである。

こういう質問では**字面の一致がそのまま答え**になる。BM25 を併用する
動機はここにあり、単に「定番だから入れる」のではない。

インデックスは ``index.chunks`` から遅延構築する。密インデックスとは
別に永続化していないのは、BM25 の構築が数千チャンク規模では一瞬で終わり、
成果物を増やすほうがコストになるため。大規模化したら見直す。
"""

from __future__ import annotations

from typing import Any

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import ScoredChunk
from ragforlocalllm.stages.retriever.tokenize import TokenizerName, build_tokenizer

_EMPTY_DOC = "__empty__"


@register("retriever", "sparse")
class BM25Retriever:
    """BM25 による検索。インデックスは実行時に注入される。

    Parameters
    ----------
    tokenizer:
        ``char_ngram``（既定、追加依存なし）または ``sudachi``。
        **日本語では分割方式が検索性能を直接左右する**ため実験軸にする。
    k1 / b:
        BM25 のパラメータ。``b`` は文書長による正規化の強さ。
    """

    def __init__(
        self,
        index: object,
        top_k: int = 5,
        tokenizer: TokenizerName = "char_ngram",
        tokenizer_options: dict[str, Any] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.tokenizer = build_tokenizer(tokenizer, **(tokenizer_options or {}))
        self.k1 = k1
        self.b = b
        self._bm25: Any | None = None
        self._chunk_ids: list[str] = []

    # ------------------------------------------------------------------

    def _ensure_index(self) -> Any:
        if self._bm25 is not None:
            return self._bm25
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - 任意依存
            raise RuntimeError("rank-bm25 が必要です: uv sync --extra retrieval") from exc

        chunks = getattr(self.index, "chunks", None)
        if chunks is None:
            raise TypeError(
                f"{type(self.index).__name__} は chunks を公開していません。"
                "BM25 はチャンク本文を必要とします。"
            )
        self._chunk_ids = [c.chunk_id for c in chunks]
        # 空の文書があると rank_bm25 の平均文書長がゼロ除算になるため、
        # 実際のクエリと衝突しないプレースホルダを入れる。文字n-gramの
        # トークンは常に n 文字なので、これより長い語とは衝突しない。
        corpus = [self.tokenizer.tokenize(c.text) or [_EMPTY_DOC] for c in chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        return self._bm25

    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]:
        if not queries:
            return []
        bm25 = self._ensure_index()
        k = top_k or self.top_k

        # 複数クエリはチャンクごとの最大スコアで統合する（dense と同じ規約）
        best: dict[str, float] = {}
        for query in queries:
            tokens = self.tokenizer.tokenize(query)
            if not tokens:
                continue
            for chunk_id, score in zip(self._chunk_ids, bm25.get_scores(tokens), strict=True):
                value = float(score)
                if value > best.get(chunk_id, float("-inf")):
                    best[chunk_id] = value

        ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[ScoredChunk] = []
        for chunk_id, score in ordered:
            chunk = self.index.get(chunk_id)  # type: ignore[attr-defined]
            if chunk is None:
                continue
            out.append(ScoredChunk(chunk=chunk, score=score, provenance={"bm25": score}))
        return out

    def describe(self) -> dict[str, Any]:
        """**分割方式を必ず記録する。** 同じ BM25 でも別物になる。"""
        return {"retriever": "sparse", "k1": self.k1, "b": self.b, **self.tokenizer.describe()}

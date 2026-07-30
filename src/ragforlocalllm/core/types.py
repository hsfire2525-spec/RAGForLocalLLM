"""パイプラインを流れるデータ型。

段（stage）間の受け渡しはすべてここで定義した型で行う。
設計方針は docs/design/design.md §4.1 を参照。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class Document(BaseModel):
    """Loader が出力する文書1件。"""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    """source, title, n_pages など。"""


class Chunk(BaseModel):
    """Chunker が出力する検索単位。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    """page, section_path, is_table, char_start, char_end など。"""
    parent_id: str | None = None
    """親子分割・sentence-window で、生成に渡す親チャンクのID。"""

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if page is not None else None

    @property
    def section_path(self) -> str | None:
        path = self.metadata.get("section_path")
        return str(path) if path is not None else None

    @property
    def is_table(self) -> bool:
        return bool(self.metadata.get("is_table", False))


class ScoredChunk(BaseModel):
    """検索・リランク結果。

    provenance に各段のスコアを残すことで、hybrid 検索やリランクの
    寄与を事後に分析できる（例: {"dense": 0.81, "bm25": 4.2, "rerank": 0.95}）。
    """

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float
    provenance: dict[str, float] = Field(default_factory=dict)

    def with_score(self, score: float, *, source: str) -> ScoredChunk:
        """スコアを更新し、元のスコアを provenance に残した複製を返す。"""
        return ScoredChunk(
            chunk=self.chunk,
            score=score,
            provenance={**self.provenance, source: score},
        )


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


class Prompt(BaseModel):
    """Generator に渡すチャット形式のプロンプト。

    プロンプト文字列を自前で組まず、チャットテンプレートの適用は
    LM Studio 側に任せる（docs/design/design.md §3.2(9)）。
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]
    context_chunk_ids: list[str] = Field(default_factory=list)
    """実際にプロンプトへ含めたチャンクID（提示順）。引用検証に使う。"""
    token_estimate: int | None = None
    token_estimate_method: Literal["tokenizer", "char_heuristic"] | None = None
    """トークン数の算定方式。数値の比較可能性に影響するため必ず記録する。"""
    template: str | None = None
    """使用したテンプレートの識別子（パス等）。"""
    n_dropped: int = 0
    """コンテキスト予算に収まらず落としたチャンク数。

    予算がボトルネックになっているかの判断材料。検索が正解を
    取れていても、ここで落ちていれば回答できない。
    """
    n_truncated: int = 0
    """予算に合わせて途中で切ったチャンク数。"""


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Answer(BaseModel):
    """Generator / PostGeneration が出力する回答。"""

    model_config = ConfigDict(extra="forbid")

    text: str
    """後処理後の最終回答。棄権した場合は棄権文が入る。"""
    raw_text: str | None = None
    """モデルの生出力（後処理で書き換えた場合の元）。"""
    citations: list[str] = Field(default_factory=list)
    """回答が引用したチャンクID。引用整合性チェックで検証する。"""
    abstained: bool = False
    """根拠不足として棄権したか。誤答率の算出に使う。"""
    schema_valid: bool | None = None
    """構造化出力を要求した場合のスキーマ適合。要求していなければ None。"""
    model: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: float | None = None


class StageTrace(BaseModel):
    """1つの段の実行記録。

    どの段で情報が落ちたのか、どの段が時間を食っているのかを
    事後に追えるようにする（docs/design/design.md §4.1）。
    """

    model_config = ConfigDict(extra="forbid")

    stage: str
    """段の役割名（例: "retriever", "post_retrieval[0]"）。"""
    impl: str
    """実装の登録名（例: "hybrid"）。"""
    duration_ms: float
    rss_mb: float | None = None
    """段の終了時点のプロセスRSS。環境1でのメモリ制約の判断材料。"""
    info: dict[str, Any] = Field(default_factory=dict)
    """件数・トークン数など、段固有の観測値。"""


class QueryState(BaseModel):
    """クエリパイプラインを流れる状態。各段はこれを受け取り、更新して返す。"""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    queries: list[str] = Field(default_factory=list)
    """QueryTransform 後のクエリ。複数化しうる。"""
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    """Retriever の出力。"""
    contexts: list[ScoredChunk] = Field(default_factory=list)
    """PostRetrieval 後、プロンプトに渡す候補（提示順）。"""
    prompt: Prompt | None = None
    answer: Answer | None = None
    trace: list[StageTrace] = Field(default_factory=list)

    @classmethod
    def new(cls, query: str) -> QueryState:
        return cls(original_query=query, queries=[query])

    @property
    def total_duration_ms(self) -> float:
        return sum(t.duration_ms for t in self.trace)

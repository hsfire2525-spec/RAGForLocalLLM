# RAGForLocalLLM 設計方針

性能の低いローカルLLMでも回答精度を最大化することを目的とした、実験用RAGフレームワークの設計方針。

---

## 1. 目的とスコープ

### 目的

1. **性能の低いローカルLLMを前提に、回答精度を可能な限り引き上げる**
2. **RAGの各処理段を差し替え・組み合わせ可能にし、有効な構成を体系的に発見する**

### 前提環境

| 項目 | 想定 |
| --- | --- |
| LLM実行基盤 | **LM Studio**（OpenAI互換サーバ / `http://localhost:1234/v1`） |
| 対象言語 | **日本語中心** |
| 実装言語 | Python 3.11+ |
| 実行環境 | ローカルマシン（GPUあり想定）。CI/コンテナ上ではLLM非依存の単体テストのみ |

### 非スコープ

- 本番運用向けのAPIサーバ、認証、スケーリング
- 大規模分散インデックス
- モデルのファインチューニング（将来的な検討対象。当面は既存モデルの組み合わせ最適化に集中）

---

## 2. 設計原則

### 原則1: 評価基盤を最初に作る

**比較の物差しがない状態で手法を足しても、何が効いたか分からない。** ベースラインの数値が出るまでは、機能追加より評価基盤の整備を優先する。

### 原則2: 各段は「純関数 + 差し替え可能な実装」

各処理段は「入力状態 → 出力状態」の変換として定義し、副作用（I/O・キャッシュ）は外側の層に押し出す。段の実装は `Protocol` で規定し、レジストリ経由で名前解決する。

### 原則3: 実験は設定ファイル1枚

コードを書き換えずにYAMLの差分だけで実験できる状態を維持する。設定のハッシュを実行結果に紐付け、再現性を保証する。

### 原則4: 既存フレームワークをコア抽象にしない

LangChain / LlamaIndex の抽象に乗ると、比較実験のほうが窮屈になる（各段の境界がフレームワーク側の都合で決まる）。**自前の薄いインターフェースを定義し、必要に応じて各ライブラリの実装をアダプタとして取り込む**。

### 原則5: LLMの賢さに依存しない工夫を優先する

低性能モデルでは「LLMに考えさせる手法」（HyDE、LLMリランク、複雑な指示追従、CoT）の効果が薄いか逆効果になる。**検索・整形・制約といったLLM外部の工夫**を主軸に置き、LLM依存の手法は「効くかどうかを検証する対象」として扱う。

---

## 3. パイプラインの段分割

### 3.1 全体像

インデックス構築（オフライン）とクエリ実行（オンライン）を明確に分離する。クエリ側だけを変える実験でインデックスを再利用できることが、実験サイクルの速度を決める。

```
[Index Pipeline]  Loader → Chunker → Embedder → Indexer → Index Artifact
                                                              │
[Query Pipeline]  Query → QueryTransform → Retriever ─────────┘
                            → PostRetrieval → PromptBuilder
                            → Generator → PostGeneration → Answer
                                                              │
[Eval Pipeline]   Gold Dataset × Answer/Contexts → Metrics → Run Record
```

### 3.2 各段の責務と実装候補

#### (1) Loader — 文書 → 生テキスト + メタデータ

| 候補 | 備考 |
| --- | --- |
| プレーンテキスト / Markdown | 最初の実装。見出し階層をメタデータに保持 |
| PDF（テキスト層あり） | `pypdf` / `pdfplumber` |
| PDF（レイアウト解析） | 表・段組の扱いが精度に直結。日本語PDFでは効果が大きい |
| HTML | 本文抽出（ボイラープレート除去）の有無を実験軸に |

**設計上の要点**: 見出し・ページ番号・セクションパスをメタデータとして必ず保持する。後段のコンテキスト整形と引用生成で使う。

#### (2) Chunker — テキスト → チャンク列

| 候補 | 特徴 |
| --- | --- |
| 固定長（文字数 / トークン数） | ベースライン |
| 再帰分割 | 段落 → 文 → 文字の順で境界を探す |
| 文単位 | 日本語は `。！？` + 改行 + 括弧の対応を考慮。`ja_sentence_segmenter` 等 |
| セマンティック分割 | 隣接文の埋め込み類似度で境界を決める |
| **親子（parent-child）** | 小チャンクで検索し、親チャンクを生成に渡す。**低性能LLM向けに有望** |
| **sentence-window** | ヒット文の前後N文を付与して渡す。同上 |
| 命題分解（proposition） | LLMで事実単位に分解。コストが高く低性能モデルでは品質が不安定 |

**日本語固有の注意**: 単語境界が空白で区切られないため、`RecursiveCharacterTextSplitter` の英語向けデフォルト区切り文字（`" "` 等）はほぼ機能しない。日本語の区切り文字リストを明示的に与える。

#### (3) Embedder — テキスト → ベクトル

| 候補 | 備考 |
| --- | --- |
| `multilingual-e5`（small/base/large） | **`query: ` / `passage: ` のプレフィックスが必須**。付け忘れると精度が大きく落ちる |
| `bge-m3` | dense/sparse/multi-vector を同時に出せる。ハイブリッド検索と相性がよい |
| `Ruri` 系 | 日本語特化。プレフィックス規約はモデルカードを確認 |
| `GLuCoSE` | 日本語 |

**バックエンドは2系統を用意する**:
- `sentence_transformers`: ローカル直実行。プレフィックス規約や正規化を細かく制御できる
- `openai_compat`: LM Studio の `/v1/embeddings` 経由。モデル管理をLM Studioに寄せられる

**設計上の要点**: プレフィックス規約（query/passage）をモデル定義側に持たせ、呼び出し側が意識しなくてよい構造にする。ここは実装ミスが最も起きやすく、しかも「動くが精度が出ない」形で失敗するため、単体テストで固定する。

#### (4) Indexer — チャンク → 検索インデックス

| 候補 | 備考 |
| --- | --- |
| FAISS | ローカル完結、高速。まずこれ |
| Chroma / Qdrant | メタデータフィルタが柔軟 |
| **BM25（スパース）** | 日本語では**形態素解析が必須**。`SudachiPy` または `fugashi + unidic-lite` |
| SPLADE 系 | 学習済みスパース。日本語モデルの入手性を確認 |

**日本語固有の注意**: BM25を空白区切りトークナイザで使うと日本語では全く機能しない。トークナイザ自体を実験軸（Sudachi mode A/B/C、n-gram）に含める。固有名詞・型番の検索ではBM25がdenseを上回ることが多く、**ハイブリッドは日本語RAGでは特に効く**。

#### (5) QueryTransform — クエリの前処理

| 候補 | LLM依存 | 低性能LLMでの見込み |
| --- | --- | --- |
| 素通し（identity） | なし | ベースライン |
| ルールベース正規化（表記ゆれ・全半角・略語辞書） | なし | **有望**。低コストで効く |
| multi-query（言い換え複数生成） | 中 | 検証対象 |
| HyDE（仮想回答を生成して検索） | 高 | **低性能モデルでは逆効果の懸念**。要検証 |
| step-back（抽象化した質問を併用） | 高 | 同上 |
| 質問分解（マルチホップ用） | 高 | 分解自体が失敗しやすい。専用の小型モデル併用も検討 |

#### (6) Retriever — 検索

| 候補 | 備考 |
| --- | --- |
| dense のみ | ベースライン |
| sparse（BM25）のみ | 比較用 |
| **hybrid（RRF / 重み付き加算）** | 第一候補。RRFはスコア正規化不要で扱いやすい |
| メタデータフィルタ併用 | 文書種別・日付での絞り込み |

#### (7) PostRetrieval — 検索結果の後処理（**最重要**）

| 候補 | LLM依存 | 見込み |
| --- | --- | --- |
| **リランク（cross-encoder）** | なし | **最もROIが高い**。`bge-reranker-v2-m3`、日本語特化リランカー等 |
| リランク（ColBERT / late interaction） | なし | 有望 |
| リランク（LLM） | 高 | 低性能モデルでは期待薄。比較用 |
| 重複除去（近似重複の統合） | なし | コンテキスト長を節約でき、低性能モデルに有効 |
| **コンテキスト圧縮**（抽出型: 関連文のみ残す） | なし〜低 | **有望**。文単位のリランクで実現可能 |
| **並べ替え（lost-in-the-middle 対策）** | なし | 重要度の高い文書を先頭と末尾に配置。**低コストで効く** |
| MMR（多様性確保） | なし | 冗長な検索結果が多い場合に有効 |
| 親チャンクへの展開 | なし | (2)の親子分割と対で使う |

**方針**: この段に最も投資する。低性能LLMの回答精度は「渡されたコンテキストの質」でほぼ決まる。

#### (8) PromptBuilder — コンテキスト + 質問 → プロンプト

実験軸:

- コンテキストの提示形式（番号付き / XMLタグ / Markdown見出し）
- 引用の要求形式（`[1]` 形式 / 出典明記なし / JSONで出典フィールド）
- few-shot の有無と例数
- 「コンテキストに無い場合は『分かりません』と答える」棄権指示の有無と強さ
- 指示の言語（日本語 / 英語）— **モデルによって英語指示のほうが指示追従が安定する場合がある。実験軸として扱う**
- 出力スキーマの指定

**設計上の要点**: プロンプトはコードに埋め込まず、テンプレートファイルとして管理し、設定から選択する。プロンプトはこのリポジトリで最も頻繁に変更される要素であり、差分が追跡できる形にしておく。

#### (9) Generator — LLM推論

**LM Studio 前提の実装上の注意**:

| 項目 | 扱い |
| --- | --- |
| エンドポイント | OpenAI互換 `/v1/chat/completions`。汎用アダプタとして実装し、baseURL差し替えで Ollama / vLLM も同一コードで扱える |
| 構造化出力 | LM Studio は `response_format` の JSON Schema 指定に対応。**GBNF文法の直接指定はAPI経由では利用できない**ため、構造化はJSON Schemaで行う |
| モデル切り替え | `model` パラメータ。LM Studio 側でのロード状態に依存するため、実験ランナーは事前に `/v1/models` で存在確認する |
| コンテキスト長 | LM Studio 側のロード設定に依存。**設定値をランレコードに記録しないと再現不能になる**ため、起動時に取得して記録する |
| デコードパラメータ | `temperature`（既定0）、`top_p`、`repeat_penalty`、`seed`。**決定性のため seed 固定と temperature=0 を既定にする** |

実験軸: モデル（サイズ・量子化レベル）、temperature、構造化出力の有無、システムプロンプト。

#### (10) PostGeneration — 生成後の検証と修正

| 候補 | LLM依存 | 見込み |
| --- | --- | --- |
| 引用整合性チェック（引用IDが実在するか） | なし | **必須級**。低コストでハルシネーション検出可能 |
| 根拠検証（NLIモデルで含意判定） | なし | **有望**。日本語NLIモデル（JNLI系）を使用 |
| スキーマ検証 + リトライ | なし | 構造化出力の破損を吸収 |
| 棄権判定（根拠不足なら「分かりません」に置換） | なし〜低 | **精度指標の改善に直結** |
| 自己一貫性（複数サンプル生成 → 多数決） | 中 | コストとのトレードオフ。要検証 |
| 自己批判 → 再生成（self-refine） | 高 | 低性能モデルでは劣化リスク。比較用 |
| **corrective RAG**（根拠不足なら再検索） | 中 | 有望だが実装が重い。フェーズ4 |

#### (11) Evaluator — 評価

§7 で詳述。

---

## 4. コア抽象

### 4.1 データ型

Pydantic モデルで定義し、段間の受け渡しを型で固定する。

```python
class Document(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any]      # source, title, section_path, page, ...

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]
    parent_id: str | None = None  # 親子分割・sentence-window 用

class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    provenance: dict[str, float]  # dense/sparse/rerank の各スコアを保持

class QueryState(BaseModel):
    """クエリパイプラインを流れる状態。各段はこれを受け取り、更新して返す。"""
    original_query: str
    queries: list[str]                    # QueryTransform 後（複数化しうる）
    retrieved: list[ScoredChunk] = []
    contexts: list[ScoredChunk] = []      # PostRetrieval 後
    prompt: Prompt | None = None
    answer: Answer | None = None
    trace: list[StageTrace] = []          # 各段の入出力・所要時間・トークン数
```

**`trace` を必ず持たせる。** どの段で情報が落ちたのかを事後に追えないと、原因分析ができない。

### 4.2 段のインターフェース

```python
class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...

class Embedder(Protocol):
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    # query/passage を分けることで、e5系のプレフィックス規約を実装側に閉じ込める

class Retriever(Protocol):
    def retrieve(self, queries: list[str], top_k: int) -> list[ScoredChunk]: ...

class PostRetrievalStep(Protocol):
    def process(self, state: QueryState) -> QueryState: ...
    # リランク・圧縮・並べ替えは同一インターフェースにし、リストとして合成する

class Generator(Protocol):
    def generate(self, prompt: Prompt, schema: dict | None = None) -> Answer: ...

class PostGenerationStep(Protocol):
    def process(self, state: QueryState) -> QueryState: ...
```

**PostRetrieval と PostGeneration を「ステップのリスト」にする**のが要点。リランク→重複除去→圧縮→並べ替え、といった順序自体が実験軸になる。

### 4.3 レジストリ

```python
@register("chunker", "recursive_ja")
class RecursiveJapaneseChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 64): ...
```

設定の `{type: recursive_ja, chunk_size: 512}` から解決する。`__init__` の引数をそのままYAMLキーに対応させ、Pydanticで検証する。

### 4.4 設定

```yaml
# configs/baseline.yaml
name: baseline
index:
  loader:   {type: markdown}
  chunker:  {type: fixed, chunk_size: 512, overlap: 64}
  embedder: {type: sentence_transformers, model: intfloat/multilingual-e5-base}
  indexer:  {type: faiss, metric: cosine}
query:
  query_transform: {type: identity}
  retriever:       {type: dense, top_k: 5}
  post_retrieval:  []
  prompt:          {type: template, path: prompts/basic_ja.jinja}
  generator:
    type: openai_compat
    base_url: http://localhost:1234/v1
    model: <lmstudio-model-id>
    temperature: 0.0
    seed: 42
  post_generation: []
eval:
  dataset: data/gold/qa_v1.jsonl
  metrics: [recall@5, mrr, answer_f1, faithfulness_nli, abstention_rate]
```

```yaml
# configs/hybrid_rerank.yaml — baseline からの差分のみ
extends: baseline
name: hybrid_rerank
query:
  retriever: {type: hybrid, dense_top_k: 20, sparse_top_k: 20, fusion: rrf}
  post_retrieval:
    - {type: cross_encoder_rerank, model: BAAI/bge-reranker-v2-m3, top_k: 5}
    - {type: reorder_lost_in_middle}
```

**`extends` による差分継承を初期から入れる。** 実験が増えると、設定の重複がそのまま実験の信頼性を損なう。

### 4.5 キャッシュ

**初期から必須。** これがないと、実験ごとにコーパス全体の再埋め込みが発生し、試行回数が確保できない。

| 対象 | キー |
| --- | --- |
| 埋め込み | `hash(model_id + prefix_convention + text)` |
| リランクスコア | `hash(model_id + query + passage)` |
| LLM出力 | `hash(model_id + prompt + decode_params)` |
| インデックス成果物 | `hash(index設定 + コーパスのハッシュ)` |

ストレージはローカルのSQLite + ファイル（`.cache/`）。LLM出力のキャッシュは「同一設定の再実行が即座に終わる」ことを保証し、評価の反復を現実的にする。

---

## 5. 低性能ローカルLLMに対する戦略

このリポジトリの中心的な仮説。

### 5.1 主軸となる戦略（LLM能力に依存しない）

1. **リランカーを必ず入れる** — cross-encoder はLLMを介さずに関連性判定精度を大きく上げる。最優先で検証
2. **コンテキストを短く保つ** — 低性能モデルは長文で急激に劣化する。`top_k` は 3〜5 から始め、抽出型圧縮で更に削る。「多く渡せば良い」は成立しない
3. **重要文書を先頭・末尾に配置** — lost-in-the-middle の影響は小さいモデルでより顕著
4. **構造化出力を強制** — 指示追従力に期待せず、JSON Schema で形式を保証する
5. **1タスクを複数の単純なタスクに分割** — 「検索結果から関連箇所を選ぶ」「選んだ箇所から答える」を分ける等。1回で複雑な処理を要求しない
6. **抽出的な回答を優先** — 自由生成量を減らすほどハルシネーションが減る
7. **棄権パスを用意** — 根拠不足時に「分かりません」を返せる設計は、実用上の精度（誤答率）を大きく改善する
8. **後段の機械的検証** — 引用整合性チェックとNLIによる根拠検証は、LLMの能力とは独立に効く

### 5.2 検証対象（効果が不確実な戦略）

- HyDE、step-back、LLMリランク、self-refine — 上位モデルで有効とされるが、低性能モデルでは**逆効果になる可能性がある**。「効かないことを示す」のもこのリポジトリの成果として価値がある
- 自己一貫性（多数決）— コスト増に対する精度改善が見合うかの検証
- 指示プロンプトの言語（日本語 vs 英語）

### 5.3 モデル性能の階層化

「弱いモデル1つ」に全てを任せない構成も実験軸に含める。

- 生成は弱いローカルモデル、リランクは専用cross-encoder、根拠検証は専用NLIモデル
- クエリ書き換えのみ別の小型モデル、または完全にルールベース

---

## 6. 評価設計

### 6.1 評価データセット

`data/gold/qa_v1.jsonl` 形式:

```json
{
  "qid": "q001",
  "question": "...",
  "answer": "...",
  "answer_aliases": ["...", "..."],
  "gold_chunk_ids": ["doc1#c003", "doc1#c004"],
  "gold_doc_ids": ["doc1"],
  "answerable": true,
  "question_type": "factoid",
  "tags": ["single-hop", "numeric"]
}
```

**設計上の要点**:

- **`answerable: false`（コーパスに答えが無い質問）を10〜20%含める。** 棄権性能を測れないと、「何でも答える」モデルが高スコアになってしまう
- `gold_chunk_ids` を持たせ、**検索段だけを単独評価できる**ようにする。生成の失敗と検索の失敗を切り分けられることが重要
- `question_type` / `tags` で層別集計する。「どの質問タイプで効いたか」が手法選択の判断材料になる
- **初期は30〜50問で十分。** 手作業で作れる規模から始め、質を保つ

### 6.2 メトリクス

**検索段（LLM不要・高速・決定的）**:

| 指標 | 意味 |
| --- | --- |
| recall@k | gold チャンクが上位k件に含まれる割合 |
| MRR / nDCG@k | 順位を考慮した指標 |
| context precision | 取得コンテキスト中の関連チャンク比率 |

**生成段**:

| 指標 | 種別 | 備考 |
| --- | --- | --- |
| exact match / F1 | 決定的 | 短答型に有効 |
| faithfulness（NLI） | モデルベース・非LLM | 回答文がコンテキストに含意されるか |
| citation validity | 決定的 | 引用IDの実在性・妥当性 |
| abstention precision/recall | 決定的 | `answerable: false` に対して正しく棄権したか |
| answer correctness | **LLM-as-judge** | 上記で測れない自由記述の妥当性 |

### 6.3 LLM-as-judge の二層構成

**判定役に評価対象と同じ低性能ローカルLLMを使うと、評価自体が信頼できない。** これは本リポジトリの目的（低性能モデルの改善）と直接衝突するため、明示的に分離する。

| 層 | 用途 | 実装 |
| --- | --- | --- |
| **主軸: 非LLM指標** | 日常の反復・スイープ全件 | 検索メトリクス、EM/F1、NLI、引用検証、棄権判定。決定的で高速、CIでも回せる |
| **補助: 強モデル judge** | 有望構成の最終確認、非LLM指標が測れない項目 | Claude API 等の外部強モデル。または上位のローカルモデル |

判定器も `Judge` プロトコルでプラガブルにし、`local` / `external` を設定で切り替える。**API利用が不可な環境でも非LLM指標だけで完結して回る**ことを保証する（外部judgeは常にオプション）。

judge を使う際の実装規約:
- judge の出力は構造化（スコア + 根拠）で受け取り、判定理由をランレコードに残す
- judge のプロンプトとモデルIDもランレコードに記録する（judge を変えると数値が変わるため）
- 位置バイアス対策として、比較評価では提示順を入れ替えた2回実行を行う

### 6.4 実験管理

```
runs/
  20260730-143022-hybrid_rerank-a1b2c3/
    config.resolved.yaml    # extends を解決した最終設定
    env.json                # モデルID、量子化、コンテキスト長、ライブラリversion、git commit
    predictions.jsonl       # 質問ごとの回答・使用コンテキスト・trace
    metrics.json            # 集計結果（全体 + tags別の層別集計）
```

- ディレクトリ名に**設定ハッシュ**を含め、同一設定の再実行を検出する
- `env.json` に **LM Studio側のロード設定（量子化・コンテキスト長）を必ず記録する**。ここが記録されていない実験結果は再現不能であり、無価値になる
- 比較レポート生成CLI: 複数ランを表形式で並べ、有意な差があるかを確認する。**質問数が少ないうちは信頼区間を併記し、ノイズを差と誤認しないようにする**

### 6.5 スイープ

```yaml
# configs/sweeps/retrieval.yaml
base: baseline
axes:
  query.retriever.type: [dense, sparse, hybrid]
  query.post_retrieval: [[], [{type: cross_encoder_rerank, top_k: 5}]]
  query.retriever.top_k: [3, 5, 10]
```

全組み合わせ（グリッド）と、ベースラインから1軸ずつ変えるアブレーションの両方をサポートする。**アブレーションのほうが「何が効いたか」の解釈が容易**なため、既定はアブレーションとする。

---

## 7. ディレクトリ構成

```
RAGForLocalLLM/
├── README.md
├── pyproject.toml               # uv / hatch
├── docs/
│   ├── design.md                # 本ドキュメント
│   └── experiments/             # 実験ログ（考察を人間が書く場所）
├── src/ragforlocalllm/
│   ├── core/
│   │   ├── types.py             # Document, Chunk, QueryState, Answer, ...
│   │   ├── protocols.py         # 各段の Protocol
│   │   ├── registry.py          # コンポーネント登録・解決
│   │   ├── config.py            # Pydantic設定モデル、extends解決
│   │   ├── cache.py             # 埋め込み/LLM出力キャッシュ
│   │   └── pipeline.py          # 段の合成、trace記録
│   ├── stages/
│   │   ├── loader/
│   │   ├── chunker/
│   │   ├── embedder/
│   │   ├── indexer/
│   │   ├── query_transform/
│   │   ├── retriever/
│   │   ├── post_retrieval/
│   │   ├── prompt/
│   │   ├── generator/
│   │   └── post_generation/
│   ├── eval/
│   │   ├── dataset.py
│   │   ├── metrics/             # retrieval / generation / abstention
│   │   ├── judge/               # local / external
│   │   └── runner.py
│   ├── experiments/
│   │   ├── sweep.py
│   │   └── report.py
│   └── cli.py                   # index / query / eval / sweep / report
├── prompts/                     # Jinja2テンプレート（バージョン管理対象）
├── configs/
│   ├── baseline.yaml
│   ├── *.yaml
│   └── sweeps/
├── data/
│   ├── corpus/                  # .gitignore（サンプルのみコミット）
│   └── gold/                    # 評価データセット（コミット対象）
├── runs/                        # .gitignore
├── .cache/                      # .gitignore
└── tests/
```

---

## 8. 実装フェーズ

### Phase 0: 基盤

- リポジトリ構成、`pyproject.toml`、lint/format/型チェック
- `core/`: 型定義、Protocol、レジストリ、設定（extends解決）、キャッシュ
- CLIの骨組み
- **完了条件**: ダミー実装を登録した設定YAMLでパイプラインが通り、traceが出る

### Phase 1: 評価基盤

- 評価データセット形式とローダー
- 検索メトリクス（recall@k, MRR, nDCG）
- 生成メトリクス（EM/F1、引用検証、棄権判定）
- ランナー、`runs/` へのランレコード出力、比較レポートCLI
- 小規模な日本語コーパス + gold QA（30〜50問、`answerable: false` を含む）
- **完了条件**: ダミー回答器に対して評価が回り、数値が出る

### Phase 2: ベースラインRAG

- Loader（Markdown）、固定長Chunker、`multilingual-e5` Embedder、FAISS Indexer
- dense Retriever、基本プロンプト、LM Studio Generator
- **完了条件**: エンドツーエンドで動き、ベースライン数値が記録される。**ここが基準点になる**

### Phase 3: 差し替え可能な手法群（本題）

優先順:

1. **リランカー（cross-encoder）** — 最も効果が期待できる
2. **ハイブリッド検索（BM25 + RRF）** — 日本語では特に効く見込み
3. **並べ替え・重複除去・抽出型圧縮**
4. **親子分割 / sentence-window**
5. **構造化出力 + スキーマ検証リトライ**
6. **棄権パスとNLI根拠検証**
7. 埋め込みモデル・チャンクサイズのスイープ
8. クエリ変換（identity / ルールベース / multi-query / HyDE）の比較

各手法の投入後にアブレーションを実行し、`docs/experiments/` に考察を残す。

### Phase 4: 高度な手法

- 自己一貫性（多数決）
- corrective RAG（根拠不足時の再検索ループ）
- LLM-as-judge（外部強モデル）による最終評価
- タスク分割型パイプライン（関連箇所選択 → 回答生成）

---

## 9. 未決事項

| 項目 | 決定に必要なこと |
| --- | --- |
| 評価用コーパスの題材 | 検証したいドメイン（技術文書 / 社内規程 / 学術 など）。質問の性質が変わる |
| 対象ローカルLLMの具体的なモデル・サイズ | 「性能が低い」の水準（例: 3B〜8Bの4bit量子化）を決めると、コンテキスト長やtop_kの初期値が定まる |
| ベクトルストア | FAISS で始めてよいが、メタデータフィルタを重視するなら Qdrant を早めに検討 |
| 外部judgeの利用可否 | ネットワーク/コスト制約。不可の場合は非LLM指標のみで完結させる |
| GPU/VRAM量 | リランカーと生成モデルを同時にロードできるかに影響する |

# RAGForLocalLLM 設計方針

性能の低いローカルLLMでも回答精度を最大化することを目的とした、実験用RAGフレームワークの設計方針。

---

## 1. 目的とスコープ

### 目的

1. **性能の低いローカルLLMを前提に、回答精度を可能な限り引き上げる**
2. **RAGの各処理段を差し替え・組み合わせ可能にし、有効な構成を体系的に発見する**

### 確定した前提

| 項目 | 内容 |
| --- | --- |
| LLM実行基盤 | **LM Studio**（OpenAI互換サーバ / `http://localhost:1234/v1`） |
| 対象言語 | **日本語中心** |
| 検証対象モデル | **4Bパラメータ規模・4bit量子化**（将来的に大規模モデルへ拡張）<br>・`gemma-4-e4b-it-qat`<br>・`internscience/agents-a1-4b-q4_k_m-gguf` |
| 評価用コーパス | **IPA「中小企業の情報セキュリティ対策ガイドライン 第4.0版」（PDF）**<br>**リポジトリへのコミット禁止**（§2.6 参照） |
| ベクトルストア | **FAISS** |
| 外部judge | **利用不可**（§6.4 参照。評価は非LLM指標＋人手抽出検査で完結させる） |
| 実行環境 | **環境1**: ディスクリートGPUなし（Intel内蔵GPU / VRAM 8GB 共有）<br>**環境2**: RTX 5060 Ti 16GB |
| 実装言語 | Python 3.11+ |

### 非スコープ

- 本番運用向けのAPIサーバ、認証、スケーリング
- 大規模分散インデックス
- モデルのファインチューニング（当面は既存モデルの組み合わせ最適化に集中）

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

4B級モデルでは「LLMに考えさせる手法」（HyDE、LLMリランク、複雑な指示追従、CoT）の効果が薄いか逆効果になる。**検索・整形・制約といったLLM外部の工夫**を主軸に置き、LLM依存の手法は「効くかどうかを検証する対象」として扱う。

### 原則6: 精度とコストを常に同時に測る

環境1（ディスクリートGPUなし）では、精度改善が実用不能なレイテンシと引き換えになりうる。**精度指標とレイテンシ・メモリ使用量を必ずセットで記録する。**

### 原則7: コーパス本体をリポジトリに置かずに再現性を保つ

コーパスはコミットできない。**取得手順・URL・SHA-256チェックサム・抽出結果のハッシュをコミットすることで再現性を担保する**（§2.6）。

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

対象コーパスが**日本語PDF 1点**に確定したため、PDF抽出は Phase 2 から必須の第一級要素となる。IPAガイドラインは以下を含み、抽出品質が精度の上限を直接決める。

- 2段組を含むレイアウト
- **表**（対策一覧、診断シート等の付録）
- 図中テキスト、囲み記事、チェックリスト
- 見出し階層（部・章・節）とページ番号

| 候補 | 備考 |
| --- | --- |
| `pypdf` | 最軽量。ベースライン。表は崩れる |
| `pdfplumber` | 表抽出APIあり。段組の読み順に注意 |
| `PyMuPDF` (fitz) | 高速、ブロック単位の座標が取れる。読み順の再構成に有利 |
| レイアウト解析系（Docling / unstructured 等） | 表をMarkdown/HTMLとして保持できる。**表を含む質問での効果を検証する価値が高い**。ただし環境1では処理コストに注意 |
| OCR | テキスト層があるため原則不要。図中テキストが必要になった場合のみ |

**設計上の要点**:

- 見出し階層（`section_path`）とページ番号をメタデータに必ず保持する。後段のコンテキスト整形・引用生成・評価アンカー解決（§6.2）で使う
- **PDF抽出は「実験軸」であると同時に「評価の前提」**でもある。抽出器を変えると gold の解決結果も変わるため、抽出器IDを常にランレコードに記録する
- 表を「1つの意味単位」として扱えるか（行がバラバラのチャンクに散らないか）は、この段と Chunker の協調で決まる

#### (2) Chunker — テキスト → チャンク列

| 候補 | 特徴 |
| --- | --- |
| 固定長（文字数 / トークン数） | ベースライン |
| 再帰分割 | 段落 → 文 → 文字の順で境界を探す |
| 文単位 | 日本語は `。！？` + 改行 + 括弧の対応を考慮 |
| **見出し階層ベース** | 節単位で分割し、`section_path` をチャンク先頭に付与。**構造の明確なガイドライン文書では有望** |
| セマンティック分割 | 隣接文の埋め込み類似度で境界を決める |
| **親子（parent-child）** | 小チャンクで検索し、親チャンクを生成に渡す。**4B級モデル向けに有望** |
| **sentence-window** | ヒット文の前後N文を付与して渡す。同上 |
| 表を分割しない特別扱い | 表は行単位で切らず、1チャンク（またはヘッダ付き行グループ）として保持 |
| 命題分解（proposition） | LLMで事実単位に分解。4B級では品質が不安定。優先度低 |

**日本語固有の注意**: 単語境界が空白で区切られないため、`RecursiveCharacterTextSplitter` の英語向けデフォルト区切り文字（`" "` 等）はほぼ機能しない。日本語の区切り文字リストを明示的に与える。

#### (3) Embedder — テキスト → ベクトル

| 候補 | パラメータ規模 | 想定環境 |
| --- | --- | --- |
| `multilingual-e5-small` | 118M | 環境1・環境2 |
| `multilingual-e5-base` | 278M | 環境1・環境2 |
| `multilingual-e5-large` | 560M | 主に環境2 |
| `bge-m3` | 568M | 主に環境2。dense/sparse/multi-vector を同時に出せる |
| `Ruri` 系 | 各サイズ | 日本語特化。プレフィックス規約はモデルカードを確認 |

**`multilingual-e5` 系は `query: ` / `passage: ` プレフィックスが必須。** 付け忘れても動作するが精度が大きく落ちる、という発見しにくい失敗をするため、プレフィックス規約はモデル定義側に持たせ、単体テストで固定する。

**バックエンドは2系統**:
- `sentence_transformers`: ローカル直実行。プレフィックス規約・正規化・プーリングを細かく制御できる。**既定**
- `openai_compat`: LM Studio の `/v1/embeddings` 経由。モデル管理をLM Studioに寄せたい場合

**環境1での注意**: 埋め込みモデルはCPU実行になる可能性が高い。コーパス全体の埋め込みはインデックス構築時の一度きりだが、**クエリ側の埋め込みは毎回走る**ためレイテンシに直接効く。環境1では `small` / `base` を既定とし、`large` は環境2での比較用とする。

#### (4) Indexer — チャンク → 検索インデックス

| 候補 | 備考 |
| --- | --- |
| **FAISS** | 確定。`IndexFlatIP`（正規化済みベクトルの内積 = cosine）で開始。数千〜数万チャンク規模では近似不要 |
| **BM25（スパース）** | 日本語では**形態素解析が必須**。`SudachiPy` または `fugashi + unidic-lite` |
| SPLADE 系 | 学習済みスパース。日本語モデルの入手性を確認。優先度低 |

**日本語固有の注意**: BM25を空白区切りトークナイザで使うと日本語では全く機能しない。トークナイザ自体を実験軸（Sudachi mode A/B/C、文字n-gram）に含める。

**このコーパスではBM25が特に効く見込み**: 「不正アクセス」「情報セキュリティ基本方針」「5分でできる！情報セキュリティ自社診断」といった**固有の用語・章タイトルでの参照**が多く、語彙一致が強い手がかりになる。ハイブリッド検索を早期に検証する。

#### (5) QueryTransform — クエリの前処理

| 候補 | LLM依存 | 4B級での見込み |
| --- | --- | --- |
| 素通し（identity） | なし | ベースライン |
| ルールベース正規化（NFKC、表記ゆれ、略語辞書） | なし | **有望**。低コストで効く |
| 用語辞書によるクエリ拡張 | なし | **有望**。セキュリティ用語の同義語（例: 標的型攻撃 / スピアフィッシング）を手作業辞書で展開 |
| multi-query（言い換え複数生成） | 中 | 検証対象 |
| HyDE（仮想回答を生成して検索） | 高 | **4B級では逆効果の懸念**。要検証 |
| step-back（抽象化した質問を併用） | 高 | 同上 |
| 質問分解（マルチホップ用） | 高 | 分解自体が失敗しやすい。優先度低 |

#### (6) Retriever — 検索

| 候補 | 備考 |
| --- | --- |
| dense のみ | ベースライン |
| sparse（BM25）のみ | 比較用 |
| **hybrid（RRF / 重み付き加算）** | 第一候補。RRFはスコア正規化不要で扱いやすい |
| メタデータフィルタ併用 | 章・付録の区別など |

#### (7) PostRetrieval — 検索結果の後処理（**最重要**）

| 候補 | LLM依存 | 見込み |
| --- | --- | --- |
| **リランク（cross-encoder）** | なし | **最もROIが高い**。最優先で検証 |
| リランク（ColBERT / late interaction） | なし | 有望 |
| リランク（LLM） | 高 | 4B級では期待薄。比較用 |
| 重複除去（近似重複の統合） | なし | コンテキスト長を節約でき、4B級に有効 |
| **コンテキスト圧縮**（抽出型: 関連文のみ残す） | なし〜低 | **有望**。文単位のリランクで実現可能 |
| **並べ替え（lost-in-the-middle 対策）** | なし | 重要文書を先頭と末尾に配置。**低コストで効く** |
| MMR（多様性確保） | なし | 冗長な検索結果が多い場合に有効 |
| 親チャンクへの展開 | なし | (2)の親子分割と対で使う |
| `section_path` の付与 | なし | チャンク単体では文脈不明な箇所を補う。低コスト |

**リランカーは環境ごとにサイズを分ける**:

| 環境 | 候補 |
| --- | --- |
| 環境1（GPUなし） | 小型 cross-encoder（MiniLM系 / 日本語 reranker の xsmall・small） |
| 環境2（RTX 5060 Ti） | `bge-reranker-v2-m3` 等の大型 |

**リランカーのサイズは精度とレイテンシのトレードオフそのもの**なので、実験軸として明示的に扱う（「環境1では小型リランカーで十分な改善が得られるか」が実用上の主要な問いになる）。

**方針**: この段に最も投資する。4B級モデルの回答精度は「渡されたコンテキストの質」でほぼ決まる。

#### (8) PromptBuilder — コンテキスト + 質問 → プロンプト

実験軸:

- コンテキストの提示形式（番号付き / XMLタグ / Markdown見出し / `section_path` 付き）
- 引用の要求形式（`[1]` 形式 / 出典なし / JSONの出典フィールド）
- few-shot の有無と例数
- 棄権指示（「コンテキストに無い場合は『分かりません』」）の有無と強さ
- 指示の言語（日本語 / 英語）— モデルによって英語指示のほうが指示追従が安定する場合がある
- 出力スキーマの指定
- **system / user のどこに指示を置くか** — 後述のGemma系の制約により実験軸になる

**コンテキスト予算を第一級の設定にする**:

```yaml
prompt:
  context_token_budget: 1536   # コンテキストに割り当てる上限トークン数
  overflow_policy: drop_lowest # drop_lowest | truncate_each | compress
```

4B級・4bit量子化モデルでは、名目コンテキスト長に達する前に品質が劣化する。`top_k` を件数で決めるのではなく**トークン予算で決める**ほうが、チャンクサイズを変える実験と整合する。

**トークン数の算定**: LM StudioのOpenAI互換APIにトークン化エンドポイントはないため、(a) 対応するHFモデルの `transformers` トークナイザで正確に数える、(b) 取得できない場合は文字数ベースの保守的な推定（安全マージン込み）にフォールバックする。**どちらの方式を使ったかをランレコードに記録する**（数値の比較可能性に影響するため）。

**設計上の要点**: プロンプトはコードに埋め込まず、テンプレートファイルとして管理し、設定から選択する。プロンプトはこのリポジトリで最も頻繁に変更される要素であり、差分が追跡できる形にしておく。

#### (9) Generator — LLM推論

**LM Studio 前提の実装上の注意**:

| 項目 | 扱い |
| --- | --- |
| エンドポイント | `/v1/chat/completions` を使う。**プロンプト文字列を自前で組まない**（チャットテンプレートの適用はLM Studio側に任せる）。汎用アダプタとして実装し、baseURL差し替えで Ollama / vLLM も同一コードで扱える |
| **Gemma系の system ロール** | Gemmaのチャットテンプレートは公式には system ロールを持たない。実装側が system を最初の user ターンに畳み込むため、**同じ指示でもモデル間で扱いが変わる**。指示を system / user のどちらに置くかを実験軸に含め、`agents-a1-4b` と比較する |
| 構造化出力 | LM Studio は `response_format` の JSON Schema 指定に対応。**GBNF文法の直接指定はAPI経由では利用できない**ため、構造化はJSON Schemaで行う。4B級では構造化制約が回答品質を下げる場合もあるため、有無を必ず比較する |
| モデル切り替え | `model` パラメータ。LM Studio側のロード状態に依存するため、実験ランナーは事前に `/v1/models` で存在確認する |
| コンテキスト長・量子化 | LM Studio側のロード設定に依存。**記録しない実験結果は再現不能**。起動時に取得してランレコードに記録する |
| デコードパラメータ | `temperature`（既定 0）、`top_p`、`repeat_penalty`、`seed`。**決定性のため seed 固定と temperature=0 を既定にする** |
| モデルの実メモリ使用量 | `gemma-4-e4b-it-qat` の "E4B" は Gemma 3n 系の *effective* パラメータ数表記であり、実際のロード時フットプリントは素の4B密モデルと異なりうる。**環境1（VRAM 8GB共有）ではリランカー・埋め込みモデルとの同時常駐が成立するかを実測して記録する** |

実験軸: モデル（2種）、temperature、構造化出力の有無、指示の配置、コンテキスト予算。

#### (10) PostGeneration — 生成後の検証と修正

| 候補 | LLM依存 | 見込み |
| --- | --- | --- |
| 引用整合性チェック（引用IDが実在するか） | なし | **必須級**。低コストでハルシネーション検出可能 |
| **根拠検証（NLIモデルで含意判定）** | なし | **有望**。日本語NLIモデル（JNLI/JSNLI系）を使用 |
| スキーマ検証 + リトライ | なし | 構造化出力の破損を吸収 |
| **棄権判定（根拠不足なら「分かりません」に置換）** | なし〜低 | **誤答率の改善に直結** |
| 自己一貫性（複数サンプル生成 → 多数決） | 中 | コストとのトレードオフ。要検証 |
| 自己批判 → 再生成（self-refine） | 高 | 4B級では劣化リスク。比較用 |
| **corrective RAG**（根拠不足なら再検索） | 中 | 有望だが実装が重い。Phase 4 |

**環境1での注意**: NLIモデルの推論も回答ごとに走る。回答は1件だが検証対象の文数に比例するため、文数上限とキャッシュを設ける。

#### (11) Evaluator — 評価

§6 で詳述。

---

## 4. コア抽象

### 4.1 データ型

Pydantic モデルで定義し、段間の受け渡しを型で固定する。

```python
class Document(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any]  # source, title, section_path, page, ...


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]  # page, section_path, is_table, ...
    parent_id: str | None = None  # 親子分割・sentence-window 用


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    provenance: dict[str, float]  # dense/sparse/rerank の各スコアを保持


class QueryState(BaseModel):
    """クエリパイプラインを流れる状態。各段はこれを受け取り、更新して返す。"""

    original_query: str
    queries: list[str]  # QueryTransform 後（複数化しうる）
    retrieved: list[ScoredChunk] = []
    contexts: list[ScoredChunk] = []  # PostRetrieval 後
    prompt: Prompt | None = None
    answer: Answer | None = None
    trace: list[StageTrace] = []  # 各段の入出力・所要時間・トークン数・ピークメモリ
```

**`trace` を必ず持たせる。** どの段で情報が落ちたのか、どの段が時間を食っているのかを事後に追えないと、原因分析ができない。環境1での実験ではレイテンシの内訳が重要になるため、`StageTrace` に所要時間とピークメモリを含める。

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
corpus: data/corpus/ipa_sme_guideline_v4.0.pdf   # gitignore対象。scripts/fetch_corpus.py で取得
index:
  loader:   {type: pymupdf, keep_layout: true}
  chunker:  {type: fixed, chunk_size: 512, overlap: 64}
  embedder: {type: sentence_transformers, model: intfloat/multilingual-e5-base}
  indexer:  {type: faiss, metric: cosine}
query:
  query_transform: {type: identity}
  retriever:       {type: dense, top_k: 5}
  post_retrieval:  []
  prompt:
    type: template
    path: prompts/basic_ja.jinja
    context_token_budget: 1536
  generator:
    type: openai_compat
    base_url: http://localhost:1234/v1
    model: gemma-4-e4b-it-qat
    temperature: 0.0
    seed: 42
  post_generation: []
eval:
  dataset: data/gold/qa_v1.jsonl
  metrics: [recall@5, mrr, answer_char_f1, citation_validity, abstention, latency]
```

```yaml
# configs/hybrid_rerank.yaml — baseline からの差分のみ
extends: baseline
name: hybrid_rerank
query:
  retriever: {type: hybrid, dense_top_k: 20, sparse_top_k: 20, fusion: rrf}
  post_retrieval:
    - {type: cross_encoder_rerank, model: <reranker-id>, top_k: 5}
    - {type: reorder_lost_in_middle}
```

**`extends` による差分継承を初期から入れる。** 実験が増えると、設定の重複がそのまま実験の信頼性を損なう。

### 4.5 キャッシュ

**初期から必須。** これがないと、実験ごとにコーパス全体の再埋め込みが発生し、試行回数が確保できない。環境1では特に効く。

| 対象 | キー |
| --- | --- |
| PDF抽出結果 | `hash(loader設定 + PDFのSHA-256)` |
| 埋め込み | `hash(model_id + prefix_convention + text)` |
| リランクスコア | `hash(model_id + query + passage)` |
| NLI判定 | `hash(model_id + premise + hypothesis)` |
| LLM出力 | `hash(model_id + prompt + decode_params + schema)` |
| インデックス成果物 | `hash(index設定 + コーパスのSHA-256)` |

ストレージはローカルのSQLite + ファイル（`.cache/`）。LLM出力のキャッシュは「同一設定の再実行が即座に終わる」ことを保証し、評価の反復を現実的にする。

### 4.6 コーパスの取得と再現性

コーパスはコミットしない。代わりに以下をコミットする。

- `scripts/fetch_corpus.py` — 公開URLからPDFを `data/corpus/` へダウンロードし、SHA-256を検証する
- `data/corpus/README.md` — 出典・URL・版・期待するSHA-256・**コミット禁止の明記**
- ランレコードに、コーパスのSHA-256と抽出結果のハッシュを記録する

これにより、PDF本体がリポジトリに無くても「どの版のどのファイルに対する結果か」が一意に特定できる。**PDFが差し替わった（版が上がった）場合はハッシュ不一致で検出でき、過去の結果と混ざらない。**

単体テストには、著作物を含まない小さな合成コーパス（`data/corpus/sample/`）を用意してコミットする。

---

## 5. 4B級ローカルLLMに対する戦略

このリポジトリの中心的な仮説。

### 5.1 主軸となる戦略（LLM能力に依存しない）

1. **リランカーを必ず入れる** — cross-encoder はLLMを介さずに関連性判定精度を上げる。最優先で検証。環境1では小型リランカーで代替可能かを併せて確認する
2. **コンテキストをトークン予算で管理する** — 4B級は名目コンテキスト長に達する前に劣化する。件数ではなくトークン予算（1〜2k程度から）で制御し、抽出型圧縮で更に削る。「多く渡せば良い」は成立しない
3. **重要文書を先頭・末尾に配置** — lost-in-the-middle の影響は小さいモデルでより顕著
4. **構造化出力を強制** — 指示追従力に期待せず、JSON Schema で形式を保証する。ただし4B級では制約が品質を下げる例もあるため、有無を比較する
5. **1タスクを複数の単純なタスクに分割** — 「関連箇所を選ぶ」「選んだ箇所から答える」を分ける等。1回で複雑な処理を要求しない
6. **抽出的な回答を優先** — 自由生成量を減らすほどハルシネーションが減る
7. **棄権パスを用意** — 根拠不足時に「分かりません」を返せる設計は、実用上の誤答率を大きく改善する
8. **後段の機械的検証** — 引用整合性チェックとNLIによる根拠検証は、LLMの能力とは独立に効く
9. **`section_path` をコンテキストに付与** — チャンク単体では何の話か分からない箇所を、低コストで補える

### 5.2 検証対象（効果が不確実な戦略）

- HyDE、step-back、LLMリランク、self-refine — 上位モデルで有効とされるが、4B級では**逆効果になる可能性がある**。「効かないことを示す」のもこのリポジトリの成果として価値がある
- 自己一貫性（多数決）— コスト増に対する精度改善が見合うかの検証
- 指示プロンプトの言語（日本語 vs 英語）、system/user の配置

### 5.3 モデル役割の分離

「弱いモデル1つ」に全てを任せない構成も実験軸に含める。

- 生成は4B級ローカルモデル、リランクは専用cross-encoder、根拠検証は専用NLIモデル
- クエリ拡張は完全にルールベース＋用語辞書
- `agents-a1-4b` がツール呼び出しに調整されている場合、ツール利用型RAG（検索を明示的なツールとして呼ばせる）を Phase 4 の比較対象に加える

### 5.4 2モデルの比較観点

`gemma-4-e4b-it-qat` と `agents-a1-4b-q4_k_m` は素性が異なるため、**同一構成での比較を常に取る**。

- 指示追従の安定性（棄権指示に従うか、引用形式を守るか）
- 構造化出力（JSON Schema）の成功率
- system ロールの扱いの差
- 日本語の流暢さと、コンテキストからの逸脱率
- レイテンシとメモリフットプリント

「構成の良さ」と「モデルの相性」を混同しないため、**モデルは常に実験軸の1つとして明示する**（片方でのみ効く工夫があり得る）。

---

## 6. 評価設計

### 6.1 コーパスに即した質問設計

IPAガイドラインの性質から、以下の質問タイプを想定する。`question_type` として層別集計する。

| タイプ | 例 | 機械採点の容易さ |
| --- | --- | --- |
| 用語定義 | 「情報セキュリティ基本方針とは何か」 | 中 |
| **数値・期限・件数** | 「自社診断の項目数は」 | **高** |
| **列挙** | 「〇〇に含まれる対策を挙げよ」 | **高**（集合一致で採点） |
| 手順 | 「インシデント発生時の初動対応の手順は」 | 低 |
| 責任・体制 | 「〇〇を承認するのは誰か」 | 中 |
| 参照解決 | 「付録◯の診断シートは何を確認するものか」 | 中 |
| **表の参照** | 表内の特定セルの値 | **高**（抽出品質の診断にも有用） |
| **回答不能** | コーパスに答えが無い質問 | **高** |

**外部judgeが使えないため、データセット設計を機械採点可能な形に寄せる。** 自由記述の妥当性を自動判定する手段がない状況では、「短答・数値・列挙・回答不能」を中心に構成するのが唯一の実用的な方針である。手順のような長文回答は少数に留め、人手抽出検査（§6.4）で扱う。

### 6.2 評価データセットの形式

`data/gold/qa_v1.jsonl`:

```json
{
  "qid": "q001",
  "question": "情報セキュリティ基本方針を承認するのは誰か。",
  "answer": "経営者",
  "answer_aliases": ["経営者", "社長", "経営者自身"],
  "answer_type": "short",
  "evidence": [
    {"page": 23, "quote": "基本方針は経営者が承認し"}
  ],
  "answerable": true,
  "question_type": "responsibility",
  "tags": ["single-hop"]
}
```

**gold の根拠は「チャンクID」ではなく「ページ + 引用文」でアンカーする。**

前版の設計では `gold_chunk_ids` としていたが、これは誤りだった。チャンクIDは Chunker の設定に依存するため、チャンク戦略を変えるたびに gold が無効になり、**このリポジトリの主目的（チャンク戦略の比較）と両立しない。**

代わりに、評価時に「ページ + 引用文」を実際のチャンク集合へ解決する。

```python
def resolve_gold(evidence: list[Evidence], chunks: list[Chunk]) -> set[str]:
    """引用文を含むチャンクIDを返す。NFKC正規化＋空白除去で比較。"""
```

これにより、**同じ gold データセットで任意の Loader / Chunker 構成を評価できる。**

副産物として重要な指標が得られる:

> **gold quote resolvability rate（引用解決率）** — 引用文がどのチャンクにも見つからない場合、Loader または Chunker が情報を破壊している（典型例: 表のセルが分断された、2段組の読み順が壊れた）。これは Loader / Chunker の情報損失を直接測る指標であり、抽出器の比較に使える。

**その他の設計上の要点**:

- **`answerable: false` を10〜20%含める。** 棄権性能を測れないと、「何でも答える」モデルが高スコアになる
- **引用文は gold に含める（決定済み）。** ただし**識別に必要な最小限の長さ**に留める（コーパス本体はコミットできない前提を尊重する）。ページ・見出しのみでのアンカーも選択可能にし、引用文なしでも動く設計とする（`Evidence` は `quote` / `page` / `section_path` のいずれか1つ以上を要求する）
- `question_type` / `tags` で層別集計する。「どの質問タイプで効いたか」が手法選択の判断材料になる
- **初期は30〜50問で十分。** 手作業で作れる規模から始め、質を保つ。表参照と回答不能を必ず含める

### 6.3 メトリクス

すべて非LLM。外部APIを一切必要としない。

**検索段（高速・決定的）**:

| 指標 | 意味 |
| --- | --- |
| recall@k | gold根拠を含むチャンクが上位k件に入る割合 |
| MRR / nDCG@k | 順位を考慮した指標 |
| context precision | 取得コンテキスト中の関連チャンク比率 |
| **gold quote resolvability** | 抽出・分割の情報損失（§6.2） |

**生成段**:

| 指標 | 種別 | 備考 |
| --- | --- | --- |
| exact match | 決定的 | NFKC正規化 + 記号除去 + エイリアス照合 |
| **answer char F1** | 決定的 | **日本語では文字レベルF1を主指標にする**。単語F1は形態素解析器に依存し、解析器を変えると数値が動く |
| answer token F1 | 決定的 | 補助。Sudachi で分割。使用した解析器とモードを記録する |
| 列挙型の集合F1 | 決定的 | 列挙質問向け |
| **faithfulness（NLI）** | モデルベース・非LLM | 回答文がコンテキストに含意されるか。日本語NLIモデル使用 |
| citation validity | 決定的 | 引用IDの実在性と、引用先が gold 根拠を含むか |
| **正答率 / 誤答率 / 棄権率** | 決定的 | **単一の精度指標では棄権の効果が見えない。この3値を常に併記する** |
| abstention precision / recall | 決定的 | `answerable: false` に正しく棄権したか、`true` を誤って棄権していないか |
| schema validity | 決定的 | 構造化出力の成功率 |

**コスト指標（環境比較に必須）**:

| 指標 | 備考 |
| --- | --- |
| end-to-end レイテンシ（p50 / p95） | 段別の内訳も記録 |
| 段別レイテンシ | どこがボトルネックか |
| ピークメモリ / VRAM | 環境1で同時常駐が成立するかの判断材料 |
| 生成トークン数 | |

**主要指標の定義**: 実用上の目的は「誤答を避けつつ正答を増やす」ことなので、**主指標は「誤答率を一定以下に抑えたうえでの正答率」**とする。単なる正答率だけを最大化すると、棄権を捨てて誤答を増やす構成が勝ってしまう。

### 6.4 judge の扱い（外部judge不可）

外部judgeを使えないため、評価は次の3層で構成する。**LLM-as-judge を自動評価の主軸には置かない。**

| 層 | 用途 | 実装 |
| --- | --- | --- |
| **第1層: 非LLM指標（主軸）** | 日常の反復・スイープ全件 | §6.3 のすべて。決定的・高速・CI実行可能。**すべての構成比較はこの層で行う** |
| **第2層: 人手抽出検査** | 自由記述の妥当性確認、非LLM指標の妥当性検証 | 各ランからN件（例: 10件）を層別サンプリングし、CLIで人手判定。結果を `runs/.../human_review.jsonl` に記録 |
| **第3層: ローカル強モデル judge（任意・低信頼）** | 参考値のみ | 環境2で、評価対象より大きいローカルモデルを judge に使う。**評価対象と同一モデルは使わない**（評価が自己言及になる）。あくまで参考値であり、これを根拠に構成を決定しない |

**人手抽出検査を軽量に回せることが、外部judge不可の環境では決定的に重要。** そのためのCLIを Phase 1 に含める:

```
$ rag review runs/20260730-143022-baseline-a1b2c3 --n 10 --stratify question_type
```

- 質問・回答・使用コンテキスト・自動判定結果を並べて表示
- 判定は「正答 / 誤答 / 妥当な棄権 / 不当な棄権」の4値 + 自由記述コメント
- **同じ質問に対する過去の人手判定を再利用**（回答文字列が一致すれば再判定不要）。これがないと人手コストが実験回数に比例して爆発する

**自動指標の妥当性検証**: 人手判定と自動指標（char F1 + NLI + 棄権判定）の一致率を定期的に測り、自動指標が信頼できる範囲を把握する。乖離が大きい質問タイプは、データセット側を機械採点しやすい形に修正する。

### 6.5 実験管理

```
runs/
  20260730-143022-hybrid_rerank-a1b2c3/
    config.resolved.yaml    # extends を解決した最終設定
    env.json                # 下記参照
    predictions.jsonl       # 質問ごとの回答・使用コンテキスト・trace
    metrics.json            # 集計結果（全体 + question_type / tags 別の層別集計）
    human_review.jsonl      # 人手抽出検査の結果（実施した場合）
```

`env.json` に記録する項目（**これが欠けた実験は再現不能であり無価値**）:

| 項目 | 理由 |
| --- | --- |
| **環境ラベル**（`env1_igpu` / `env2_rtx5060ti`） | 環境間比較の軸 |
| LM Studio のモデルID・量子化・コンテキスト長設定 | 同じモデル名でも設定が違えば別物 |
| デコードパラメータ・seed | 決定性 |
| **コーパスのSHA-256**・抽出結果のハッシュ | コーパスがコミットされていないため必須 |
| 埋め込み・リランカー・NLI の各モデルID | |
| トークン数算定方式（正確 / 推定） | 数値の比較可能性 |
| git commit、主要ライブラリのバージョン | |

**比較レポートCLI**: 複数ランを表形式で並べる。質問数が30〜50と少ないため、**信頼区間（ブートストラップ）を必ず併記し、ノイズを差と誤認しないようにする**。50問での数ポイントの差は多くの場合有意でない。

### 6.6 スイープと環境の切り分け

```yaml
# configs/sweeps/retrieval.yaml
base: baseline
mode: ablation           # ablation | grid
axes:
  query.retriever.type: [dense, sparse, hybrid]
  query.post_retrieval: [[], [{type: cross_encoder_rerank, top_k: 5}]]
  query.prompt.context_token_budget: [768, 1536, 3072]
  query.generator.model: [gemma-4-e4b-it-qat, agents-a1-4b-q4_k_m]
```

既定は**アブレーション**（ベースラインから1軸ずつ変える）。「何が効いたか」の解釈が容易なため。

**2環境の使い分け**:

| 目的 | 環境 |
| --- | --- |
| 精度の探索（多数の構成を比較） | **環境2**（RTX 5060 Ti）。試行回数を稼ぐ |
| レイテンシ・メモリの実測 | **両環境**。有望構成に絞って実施 |
| 環境1で成立する構成の絞り込み | **環境1**。小型リランカー・小型埋め込みでの精度低下幅を測る |

**注意**: 同一設定・同一seedでも、CPU実行とGPU実行で数値演算が一致するとは限らず、生成結果が完全一致しない場合がある。したがって「精度は環境2で測り、環境1では速度だけ測る」という前提は無条件には置けない。Phase 2 で**同一設定を両環境で実行し、回答の一致率を確認する**。一致率が高ければ精度探索を環境2に集約でき、低ければ環境ごとに精度も測る必要がある。この確認は実験計画全体のコストを左右する。

---

## 7. ディレクトリ構成

```
RAGForLocalLLM/
├── README.md
├── pyproject.toml               # uv / hatch
├── docs/
│   ├── design.md                # 本ドキュメント
│   └── experiments/             # 実験ログ（考察を人間が書く場所）
├── scripts/
│   └── fetch_corpus.py          # PDF取得 + SHA-256検証
├── src/ragforlocalllm/
│   ├── core/
│   │   ├── types.py             # Document, Chunk, QueryState, Answer, ...
│   │   ├── protocols.py         # 各段の Protocol
│   │   ├── registry.py          # コンポーネント登録・解決
│   │   ├── config.py            # Pydantic設定モデル、extends解決
│   │   ├── cache.py             # 抽出/埋め込み/リランク/NLI/LLM出力キャッシュ
│   │   ├── env.py               # 環境情報の収集（LM Studio設定・VRAM・バージョン）
│   │   └── pipeline.py          # 段の合成、trace記録
│   ├── stages/
│   │   ├── loader/              # pypdf / pdfplumber / pymupdf / layout
│   │   ├── chunker/
│   │   ├── embedder/
│   │   ├── indexer/             # faiss / bm25
│   │   ├── query_transform/
│   │   ├── retriever/
│   │   ├── post_retrieval/
│   │   ├── prompt/
│   │   ├── generator/
│   │   └── post_generation/
│   ├── eval/
│   │   ├── dataset.py
│   │   ├── resolve.py           # gold引用 → チャンクID の解決
│   │   ├── metrics/             # retrieval / generation / abstention / cost
│   │   ├── review.py            # 人手抽出検査CLI
│   │   └── runner.py
│   ├── experiments/
│   │   ├── sweep.py
│   │   └── report.py            # 信頼区間付き比較表
│   └── cli.py                   # index / query / eval / sweep / report / review
├── prompts/                     # Jinja2テンプレート（バージョン管理対象）
├── configs/
│   ├── baseline.yaml
│   ├── *.yaml
│   └── sweeps/
├── data/
│   ├── corpus/
│   │   ├── README.md            # 出典・URL・SHA-256・コミット禁止の明記
│   │   └── sample/              # 単体テスト用の合成コーパス（コミット対象）
│   └── gold/                    # 評価データセット（コミット対象）
├── runs/                        # .gitignore
├── .cache/                      # .gitignore
└── tests/
```

---

## 8. 実装フェーズ

### Phase 0: 基盤

- リポジトリ構成、`pyproject.toml`、lint/format/型チェック
- `core/`: 型定義、Protocol、レジストリ、設定（extends解決）、キャッシュ、環境情報収集
- `scripts/fetch_corpus.py` と `data/corpus/README.md`
- CLIの骨組み
- **完了条件**: ダミー実装を登録した設定YAMLでパイプラインが通り、trace と `env.json` が出る

### Phase 1: 評価基盤

- 評価データセット形式とローダー
- **gold引用 → チャンクID の解決器**（`eval/resolve.py`）と引用解決率
- 検索メトリクス（recall@k, MRR, nDCG）
- 生成メトリクス（EM、char F1、引用検証、棄権判定、正答/誤答/棄権の3値）
- コスト指標（段別レイテンシ、ピークメモリ）
- ランナー、`runs/` へのランレコード出力、**信頼区間付き**比較レポートCLI
- **人手抽出検査CLI**（過去判定の再利用込み）
- gold QA 30〜50問（表参照・回答不能を含む）
- **完了条件**: ダミー回答器に対して評価が回り、数値と信頼区間が出る

### Phase 2: ベースラインRAG

- PDF Loader（まず `pymupdf`）、固定長Chunker、`multilingual-e5-base`、FAISS
- dense Retriever、基本プロンプト、LM Studio Generator（2モデル）
- **両環境で同一設定を実行し、回答一致率を確認**（§6.6 の実験計画に直結）
- **完了条件**: エンドツーエンドで動き、2モデル × 2環境のベースライン数値が記録される。**ここが基準点になる**

### Phase 3: 差し替え可能な手法群（本題）

優先順:

1. **リランカー（cross-encoder、大小2サイズ）** — 最も効果が期待できる。環境1での実用性も同時に評価
2. **ハイブリッド検索（BM25 + RRF）** — このコーパスでは特に効く見込み
3. **コンテキスト予算のスイープ**と抽出型圧縮・並べ替え・重複除去
4. **PDF Loader の比較**（`pymupdf` vs レイアウト解析系）— 引用解決率と表参照質問の精度で評価
5. **親子分割 / sentence-window / 見出し階層分割**
6. **棄権パスとNLI根拠検証** — 誤答率の改善
7. **構造化出力の有無**、指示の言語・配置（Gemmaのsystemロール問題を含む）
8. 埋め込みモデル・チャンクサイズのスイープ
9. クエリ変換（identity / ルールベース+用語辞書 / multi-query / HyDE）の比較

各手法の投入後にアブレーションを実行し、`docs/experiments/` に考察を残す。

### Phase 4: 高度な手法

- 自己一貫性（多数決）
- corrective RAG（根拠不足時の再検索ループ）
- タスク分割型パイプライン（関連箇所選択 → 回答生成）
- `agents-a1-4b` のツール呼び出し能力を用いたツール型RAG
- ローカル強モデルによる参考judge（環境2、参考値扱い）
- より大規模なモデルへの拡張

---

## 9. 実装状況

### Phase 0（基盤）— 完了

| 要素 | 実装 |
| --- | --- |
| データ型 | `core/types.py` — `Document` / `Chunk` / `ScoredChunk` / `Prompt` / `Answer` / `StageTrace` / `QueryState` |
| Protocol | `core/protocols.py` — 11段のインターフェース |
| レジストリ | `core/registry.py` — 未知の名前・引数を候補提示付きで即座に報告する |
| 設定 | `core/config.py` — `extends` 差分継承、`index_signature` / `config_hash` |
| キャッシュ | `core/cache.py` — SQLite（JSON）+ .npy（配列）、名前空間付き |
| 環境情報 | `core/env.py` — 環境ラベル、GPU、パッケージ版、git、コーパスSHA-256、RSS |
| トークン計数 | `core/tokens.py` — HFトークナイザ / 文字数推定（**方式を必ず記録**） |
| パイプライン | `core/pipeline.py` — 段の合成と trace（所要時間・RSS・観測値） |
| インデックス | `core/indexing.py` — 構築・再利用（署名一致で再利用） |
| CLI | `cli.py` — `index` / `query` / `env` / `components` / `gold` |
| gold スキーマ | `eval/dataset.py` — 引用文アンカー、`answerable` 整合性検証 |

疎通確認用に、追加依存もLLMも不要で動く実装群を用意した（`configs/smoke.yaml`）。
LM Studio 用の Generator アダプタ（`openai_compat`）も実装・動作確認済み。

> **`extractive` ジェネレータ（LLM非依存の下限ベースライン）について**
> 検索とリランクだけでどこまで到達できるかを測る参照点。文字bi-gram の
> Dice 係数で文を選び、閾値を下回れば棄権する。**語彙的に無関係な質問には
> 棄権できるが、コーパスが話題に触れているだけで答えを含まない質問
> （例:「責任者の氏名は何か」）には誤答する。** これは語彙一致手法の
> 原理的な限界であり、NLI による根拠検証（Phase 3）が必要な理由を示す
> 具体例として記録しておく。

### 残る未決事項

| 項目 | 決定に必要なこと |
| --- | --- |
| **対象4Bモデルの入手** | `gemma-4-e4b-it-qat` と `agents-a1-4b-q4_k_m` は現行の検証機の LM Studio に未ダウンロード。Phase 2 のベースライン測定前に必要 |
| 検証機と設計上の2環境の対応 | 現行の検証機（32コア / 124GB / AMD dGPU）は設計書の環境1・環境2のどちらとも異なる。環境ラベルの定義を実機に合わせて見直すか、3環境目として扱うか |
| 日本語NLIモデルの選定 | 入手可能なモデルの精度と、環境1でのレイテンシの実測が必要 |
| リランカーの具体的なモデル | 日本語対応・小型（環境1向け）の候補を実測で絞る |
| gold QA の作成担当と分量 | 30〜50問の作成は人手作業。誰がどの範囲を作るか |
| 環境1でのVRAM同時常駐 | 4B(q4) + リランカー + 埋め込み + NLI が8GB共有で成立するか。実測待ち。成立しない場合は逐次ロード＋オフロード戦略が必要 |

# RAGForLocalLLM

性能の低いローカルLLMでも回答精度を最大化することを目的とした、実験用RAGフレームワーク。

RAGの処理フローを段（chunking / retrieval / rerank / prompt / generation / verification など）に分割し、
各段のアルゴリズムを設定ファイルだけで差し替え・組み合わせできるようにすることで、
有効な構成を体系的に比較・検証することを目指す。

## 想定環境

- LLM実行基盤: **LM Studio**（OpenAI互換サーバ）
- 対象言語: **日本語中心**
- 検証対象モデル: 4Bパラメータ規模・4bit量子化（将来的に大規模モデルへ拡張）
- Python 3.11+

## セットアップ

```bash
uv sync --group dev
```

## クイックスタート

追加依存もLLMも不要な疎通確認（架空の自作サンプルコーパスを使用）:

```bash
# 登録済みコンポーネントの一覧
uv run rag components

# インデックス構築
uv run rag index -c smoke

# 質問（段ごとの trace 付き）
uv run rag query "情報セキュリティ基本方針を承認するのは誰か。" -c smoke --contexts

# 評価データセットの検証
uv run rag gold check data/gold/sample_qa.jsonl

# 評価（runs/ にランレコードが出る）
uv run rag eval -c smoke

# ランの一覧と、信頼区間つきの比較
uv run rag runs
uv run rag report <ラン名A> <ラン名B>

# ランレコードに記録される環境情報
uv run rag env
```

LM Studio を使う場合（`configs/smoke_lmstudio.yaml` の `model` をロード済みのIDに合わせる）:

```bash
uv run rag env -c smoke_lmstudio   # サーバ上のモデル一覧を確認
uv run rag query "情報セキュリティ基本方針を承認するのは誰か。" -c smoke_lmstudio
```

## 社内共有のWeb UI

```bash
uv sync --extra server

# 自分だけで試す（認証なし・ループバックのみ）
uv run rag serve --warmup hybrid_budget

# 社内に公開する
uv run rag serve --new-token                    # 共有トークンを生成
RAG_SERVER_TOKEN=<token> uv run rag serve \
    --host 0.0.0.0 --warmup private_docs
```

ブラウザで開くと、質問・回答に加えて**根拠の原文が常に表示される**。
機密資料では「回答が正しいか」を利用者が検証できないと使えないため、
引用された箇所を強調し、予算で使われなかった箇所も明示する。

`--warmup` を付けないと、最初の質問だけ埋め込みモデルの読み込みで
十数秒かかる（実測 17秒 → 2.2秒）。1設定あたり約1.5GB常駐する。

### このUIが保証すること / しないこと

| | |
| --- | --- |
| する | 共有トークンによる認証（定数時間比較） |
| する | 全問い合わせの監査ログ（誰が・いつ・何を・どこを参照したか） |
| する | 認証なしでの社内公開を**拒否**する |
| **しない** | **個人の識別** — トークンは全員共有。ログに残るのはIPまで |
| **しない** | **資料ごとのアクセス権** — 見せる資料は設定単位でしか分けられない |
| **しない** | **通信の暗号化** — TLS終端はリバースプロキシ側で行う |

**個人単位の追跡や資料ごとの権限が必要なら、SSO を挟むまで社内公開しないこと。**
監査ログに回答本文は残していない（ログ自体が二次的な機密の集積になるため）。

## 社外秘資料を扱う場合

コーパス・gold・人手判定・ラン記録は**すべて本文の断片を含みうる**。
機密資料を対象にする場合は `data/private/` 以下にまとめて置く。
中身は `README.md` を除いて一切コミットされない。

```bash
uv run rag gold draft -c private_docs --n 40 -o data/private/gold/qa_v1.jsonl
uv run rag eval  -c private_docs --runs-root data/private/runs
uv run rag review <ラン名> --runs-root data/private/runs \
    --store data/private/reviews/judgments.jsonl
```

`.gitignore` は `data/gold/` と `data/reviews/` も**原則禁止 + 明示許可**に
してあるため、新しく作った gold は既定で無視される。加えて、
`git add -A` の事故に備えてステージ内容を検査できる:

```bash
python scripts/check_private.py          # ステージ済みの内容
python scripts/check_private.py --all    # 追跡中の全ファイル
```

pre-commit フックとして入れておくのが確実。詳細は
[`data/private/README.md`](data/private/README.md) を参照。

## 評価用コーパス

評価には IPA「中小企業の情報セキュリティ対策ガイドライン 第4.0版」を使用する。
**コーパス本体はリポジトリにコミットしない。** 取得手順と再現性の担保方法は
[`data/corpus/README.md`](data/corpus/README.md) を参照。

```bash
python scripts/fetch_corpus.py --write-lock   # 初回
python scripts/fetch_corpus.py               # 以降（SHA-256を検証）
```

### 評価データセット（gold QA）の作り方

`configs/ipa_draft.yaml` は追加依存なしで実コーパスを扱える設定で、
起草と検証に使う。**本番設定（`baseline.yaml`）と Loader・Chunker を
揃えてある**ので、ここで検証した引用文はそのまま本番でも解決する。

```bash
# 1. チャンクを層別に抽出して下書きを作る（question / answer は TODO）
uv run rag gold draft -c ipa_draft --n 40 -o data/gold/qa_v1.jsonl

# 2. 人手で question / answer / question_type を記入する
#    （配分と方針は docs/design/design.md §10.4）

# 3. 引用文が実際の抽出テキストに解決するか検証する（凍結前に必須）
uv run rag gold check data/gold/qa_v1.jsonl -c ipa_draft
```

手打ちの引用文が抽出結果と一致しないのが最大の失敗要因で、これは
検索メトリクスを静かに壊す。3 を通してから凍結すること。

## 開発

```bash
uv run pytest                    # 既定は決定的で速いものだけ
uv run pytest -m slow            # モデルのダウンロードや実LLMを要するもの
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

追加依存（PDF・FAISS・埋め込みモデル）を入れる場合:

```bash
uv sync --group dev --extra pdf --extra retrieval --extra models
```

`--extra models` は torch を伴うため初回は時間がかかる。
PyPI の既定 wheel は NVIDIA CUDA ライブラリを同梱するので、
**GPUを使わない環境やAMD環境では CPU 版を明示するほうが軽い**:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

新しい手法を追加する手順:

1. `src/ragforlocalllm/stages/<段>/` に実装を置き、`@register("<段>", "<名前>")` を付ける
2. `src/ragforlocalllm/stages/__init__.py` に import を1行足す
3. `configs/` に設定を追加（既存設定からの差分は `extends` で書く）

## ドキュメント

- [設計方針](docs/design/design.md) — パイプラインの段分割、コア抽象、評価設計、実装フェーズ
- `docs/experiments/` — 実験ログと考察

## ステータス

**Phase 0（基盤）・Phase 1（評価基盤）完了。Phase 2（ベースライン）は実装完了。**

- 設定YAMLでパイプラインが通り、trace と環境情報が記録される
- PDF Loader（`pymupdf` / `pypdf`）を評価用コーパスに対して実装・実測済み
- gold引用 → チャンクID の解決器、検索・生成・コストの各メトリクス、
  ランレコード、**信頼区間つき**比較レポート、人手抽出検査CLI が動く
- `multilingual-e5` + FAISS のベースライン（`configs/baseline.yaml`）が実コーパスで動く
  （249チャンク / 768次元、埋め込み器の常駐量 1,460MB）

**残るのは gold QA 40問の作成**（上記「評価データセットの作り方」）。
凍結できれば `rag eval -c baseline` で基準点が記録され、Phase 3（リランカー・
ハイブリッド検索などの手法比較）に進める。詳細は設計方針 §8・§9 を参照。

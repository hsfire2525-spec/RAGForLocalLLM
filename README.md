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
uv run rag gold data/gold/sample_qa.jsonl

# ランレコードに記録される環境情報
uv run rag env
```

LM Studio を使う場合（`configs/smoke_lmstudio.yaml` の `model` をロード済みのIDに合わせる）:

```bash
uv run rag env -c smoke_lmstudio   # サーバ上のモデル一覧を確認
uv run rag query "情報セキュリティ基本方針を承認するのは誰か。" -c smoke_lmstudio
```

## 評価用コーパス

評価には IPA「中小企業の情報セキュリティ対策ガイドライン 第4.0版」を使用する。
**コーパス本体はリポジトリにコミットしない。** 取得手順と再現性の担保方法は
[`data/corpus/README.md`](data/corpus/README.md) を参照。

```bash
python scripts/fetch_corpus.py --write-lock   # 初回
python scripts/fetch_corpus.py               # 以降（SHA-256を検証）
```

## 開発

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

新しい手法を追加する手順:

1. `src/ragforlocalllm/stages/<段>/` に実装を置き、`@register("<段>", "<名前>")` を付ける
2. `src/ragforlocalllm/stages/__init__.py` に import を1行足す
3. `configs/` に設定を追加（既存設定からの差分は `extends` で書く）

## ドキュメント

- [設計方針](docs/design/design.md) — パイプラインの段分割、コア抽象、評価設計、実装フェーズ
- `docs/experiments/` — 実験ログと考察

## ステータス

**Phase 0（基盤）完了。** 設定YAMLでパイプラインが通り、trace と環境情報が記録される。
PDF Loader（`pymupdf` / `pypdf`）は評価用コーパスに対して実装・実測済み（設計方針 §9）。
次は Phase 1（評価基盤: gold引用の解決器、検索・生成メトリクス、人手抽出検査CLI、
`rag gold draft` / `rag footprint`）。
実装フェーズの詳細は設計方針 §8 を参照。

# RAGForLocalLLM

性能の低いローカルLLMでも回答精度を最大化することを目的とした、実験用RAGフレームワーク。

RAGの処理フローを段（chunking / retrieval / rerank / prompt / generation / verification など）に分割し、
各段のアルゴリズムを設定ファイルだけで差し替え・組み合わせできるようにすることで、
有効な構成を体系的に比較・検証することを目指す。

## 想定環境

- LLM実行基盤: **LM Studio**（OpenAI互換サーバ）
- 対象言語: **日本語中心**
- Python 3.11+

## ドキュメント

- [設計方針](docs/design/design.md) — パイプラインの段分割、コア抽象、評価設計、実装フェーズ
- `docs/experiments/` — 実験ログと考察

## ステータス

設計フェーズ。実装は未着手。

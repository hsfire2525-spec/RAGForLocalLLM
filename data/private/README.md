# 機密資料を扱う領域

**このディレクトリの中身は `README.md` を除いて一切コミットされない。**
社外秘資料を対象に RAG を構築・評価する場合、コーパス・gold・人手判定・
ラン記録をすべてここに置く。

```
data/private/
  corpus/     … 資料本体
  gold/       … 評価データセット（本文の引用を含む）
  reviews/    … 人手判定（モデルの回答＝本文由来の文字列を含む）
  runs/       … ラン記録（回答・trace）
```

## なぜ分けるのか

公開コーパス（IPAガイドライン）用の gold は `data/gold/` にコミットしている。
出典が公開物であり、引用を残すことで**引用解決率の検証が再現できる**ためである。

機密資料ではこれが成立しない。gold には本文の引用が入り、人手判定にはモデルの
回答が入り、ラン記録には両方が入る。**評価を回すこと自体が本文の断片を
リポジトリに残す行為**になる。

`.gitignore` は「原則禁止 + 明示許可」にしてある。`data/gold/` に新しく
作ったファイルは既定で無視されるので、`git add -A` の事故が起きにくい。
それでも取り違えは起こりうるため、`scripts/check_private.py` で
ステージ内容を検査できるようにしてある（下記）。

## 使い方

```bash
# 1. 資料を置く
mkdir -p data/private/corpus data/private/gold
cp /path/to/社外秘資料.pdf data/private/corpus/

# 2. 設定を作る（configs/ 以下はコミットされるので、パス以外に
#    機密情報を書かないこと。ファイル名にも注意）
cp configs/hybrid_budget.yaml configs/private_docs.yaml
#    corpus: data/private/corpus/... と eval.dataset: data/private/gold/... に変更

# 3. gold を起草・検証する
uv run rag gold draft -c private_docs --n 40 -o data/private/gold/qa_v1.jsonl
uv run rag gold check data/private/gold/qa_v1.jsonl -c private_docs

# 4. 評価する（ラン記録もこの下に出す）
uv run rag eval -c private_docs --runs-root data/private/runs

# 5. 人手検査（判定の保存先を分ける）
uv run rag review <ラン名> --store data/private/reviews/judgments.jsonl \
    --runs-root data/private/runs
```

## コミット前の確認

```bash
python scripts/check_private.py          # ステージ済みの内容を検査
python scripts/check_private.py --all    # 追跡中の全ファイルを検査
```

pre-commit フックとして入れておくと確実:

```bash
printf '#!/bin/sh\nexec python scripts/check_private.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## 注意点

- **設定ファイル（`configs/*.yaml`）はコミットされる。** パス名・ファイル名に
  機密が現れないようにする（「A社向け見積根拠.pdf」のような名前は避ける）
- **`.cache/` にはチャンク本文がそのまま入る。** gitignore 済みだが、
  バックアップや同期対象から外しておくこと
- コーパスのハッシュを `data/corpus/corpus.lock.json` に書く運用は
  公開資料向け。機密資料のロックは `data/private/` 側に置く

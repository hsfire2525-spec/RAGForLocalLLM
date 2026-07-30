# 評価用コーパス

## ⚠️ コミット禁止

**このディレクトリ配下のコーパス本体（PDF等）は、いかなる形でもリポジトリにコミットしないこと。**
`.gitignore` で除外済みだが、`git add -f` 等で強制追加しないよう注意する。

コミットしてよいのは以下のみ:

- この `README.md`
- `sample/` — 単体テスト用の合成コーパス（著作物を含まない自作テキスト）

## 対象文書

| 項目 | 内容 |
| --- | --- |
| 名称 | 中小企業の情報セキュリティ対策ガイドライン 第4.0版 |
| 発行 | 独立行政法人 情報処理推進機構（IPA） |
| URL | https://www.ipa.go.jp/security/guide/sme/ug65p90000019cbk-att/sme_guideline_v4.0.pdf |
| ローカル配置先 | `data/corpus/ipa_sme_guideline_v4.0.pdf` |
| SHA-256 | （初回取得時に `scripts/fetch_corpus.py` が記録する。値を確定させたら本欄と `corpus.lock.json` に転記する） |

## 取得手順

```bash
python scripts/fetch_corpus.py
```

- 未取得の場合はダウンロードする
- `corpus.lock.json` に期待値が記録済みなら、SHA-256を検証する
- ハッシュが一致しない場合はエラーで停止する（版の差し替えを検出するため）

初回のみ、ハッシュを記録する:

```bash
python scripts/fetch_corpus.py --write-lock
```

## なぜハッシュを記録するのか

コーパス本体をコミットできないため、**「どの版のどのファイルに対する実験結果か」をハッシュでのみ特定できる。**
文書が改訂されてURLの内容が差し替わった場合、ハッシュ不一致で検出でき、過去の実験結果と混ざることを防げる。

すべてのランレコード（`runs/*/env.json`）にコーパスのSHA-256を記録する。

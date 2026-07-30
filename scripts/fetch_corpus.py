#!/usr/bin/env python3
"""評価用コーパスを取得し、SHA-256 で同一性を検証する。

コーパス本体はリポジトリにコミットしないため、
「どの版のどのファイルに対する実験結果か」はハッシュでのみ特定できる。
詳細は data/corpus/README.md を参照。

使い方:
    python scripts/fetch_corpus.py               # 取得 + 検証
    python scripts/fetch_corpus.py --write-lock  # 初回: ハッシュを記録
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
LOCK_PATH = CORPUS_DIR / "corpus.lock.json"


@dataclass(frozen=True)
class CorpusEntry:
    key: str
    filename: str
    url: str
    description: str


CORPUS = [
    CorpusEntry(
        key="ipa_sme_guideline_v4",
        filename="ipa_sme_guideline_v4.0.pdf",
        url=(
            "https://www.ipa.go.jp/security/guide/sme/ug65p90000019cbk-att/sme_guideline_v4.0.pdf"
        ),
        description="中小企業の情報セキュリティ対策ガイドライン 第4.0版（IPA）",
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lock() -> dict[str, dict[str, str]]:
    if not LOCK_PATH.exists():
        return {}
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def save_lock(lock: dict[str, dict[str, str]]) -> None:
    LOCK_PATH.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1024 * 256):
            out.write(chunk)
    tmp.replace(dest)


def process(entry: CorpusEntry, lock: dict[str, dict[str, str]], write_lock: bool) -> bool:
    """1件を取得・検証する。問題なければ True。"""
    print(f"[{entry.key}] {entry.description}")
    dest = CORPUS_DIR / entry.filename

    if not dest.exists():
        try:
            download(entry.url, dest)
        except Exception as exc:
            print(f"  ERROR: ダウンロードに失敗しました: {exc}", file=sys.stderr)
            print(f"  手動で取得し {dest} に配置してください。", file=sys.stderr)
            return False
    else:
        print(f"  既に存在: {dest.relative_to(REPO_ROOT)}")

    actual = sha256_of(dest)
    expected = lock.get(entry.key, {}).get("sha256")

    if expected is None:
        if write_lock:
            lock[entry.key] = {
                "filename": entry.filename,
                "url": entry.url,
                "sha256": actual,
            }
            print(f"  ハッシュを記録しました: {actual}")
            return True
        print(f"  WARNING: ハッシュ未記録です。実測値: {actual}")
        print("  この値で固定してよければ --write-lock を付けて再実行してください。")
        return True

    if actual != expected:
        print("  ERROR: SHA-256 が一致しません。", file=sys.stderr)
        print(f"    expected: {expected}", file=sys.stderr)
        print(f"    actual:   {actual}", file=sys.stderr)
        print(
            "  文書が改訂された可能性があります。過去の実験結果と混同しないよう、\n"
            "  版を確認したうえで corpus.lock.json を更新してください。",
            file=sys.stderr,
        )
        return False

    print(f"  OK (sha256={actual[:16]}...)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="未記録のハッシュを corpus.lock.json に書き込む",
    )
    args = parser.parse_args()

    lock = load_lock()
    ok = all(process(entry, lock, args.write_lock) for entry in CORPUS)

    if args.write_lock:
        save_lock(lock)
        print(f"\n{LOCK_PATH.relative_to(REPO_ROOT)} を更新しました。")

    if not ok:
        return 1
    print("\nすべてのコーパスを検証しました。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

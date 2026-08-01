#!/usr/bin/env python3
"""機密資料由来のファイルがコミット対象に入っていないか検査する。

**`.gitignore` だけでは足りない。** 無視ルールは追跡されていないファイルに
しか効かず、一度 `git add -f` された、あるいはルール追加前から追跡されていた
ファイルは素通りする。実際の事故経路は `git add -A` なので、
ステージされた内容そのものを見る。

pre-commit フックとして入れておくのが確実:

    printf '#!/bin/sh\\nexec python scripts/check_private.py\\n' > .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

使い方:

    python scripts/check_private.py          # ステージ済みの内容を検査
    python scripts/check_private.py --all    # 追跡中の全ファイルを検査
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ここ以下は理由を問わずコミットしない
FORBIDDEN_PREFIXES = ("data/private/",)

# 明示的に許可したものだけコミットしてよい領域。
# 公開コーパス（IPAガイドライン）由来の gold と人手判定のみを許可する。
GUARDED: dict[str, frozenset[str]] = {
    "data/gold/": frozenset(
        {
            "data/gold/sample_qa.jsonl",
            "data/gold/qa_v1.jsonl",
            "data/gold/qa_v2.jsonl",
        }
    ),
    "data/reviews/": frozenset({"data/reviews/human_judgments.jsonl"}),
    "data/corpus/": frozenset(
        {
            "data/corpus/README.md",
            "data/corpus/corpus.lock.json",
        }
    ),
    "runs/": frozenset(),
    ".cache/": frozenset(),
}

ALLOWED_SUBTREES = ("data/corpus/sample/", "data/private/README.md")


def tracked_or_staged(check_all: bool) -> list[str]:
    if check_all:
        cmd = ["git", "ls-files"]
    else:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def offending(paths: list[str]) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    for path in paths:
        if any(path == a or path.startswith(a) for a in ALLOWED_SUBTREES):
            continue
        if any(path.startswith(p) for p in FORBIDDEN_PREFIXES):
            problems.append((path, "data/private/ は一切コミットしない領域です"))
            continue
        for prefix, allowed in GUARDED.items():
            if path.startswith(prefix) and path not in allowed:
                problems.append(
                    (path, f"{prefix} で明示的に許可されていません（本文由来の断片を含みうる）")
                )
                break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="追跡中の全ファイルを検査")
    args = parser.parse_args()

    try:
        paths = tracked_or_staged(args.all)
    except subprocess.CalledProcessError as exc:
        print(f"git の実行に失敗しました: {exc}", file=sys.stderr)
        return 2

    problems = offending(paths)
    if not problems:
        target = "追跡中のファイル" if args.all else "ステージされた内容"
        print(f"OK: {target} {len(paths)} 件に機密領域のファイルはありません。")
        return 0

    print("機密領域のファイルがコミット対象に含まれています:\n", file=sys.stderr)
    for path, reason in problems:
        print(f"  {path}\n      {reason}", file=sys.stderr)
    print(
        "\n対処:\n"
        "  git restore --staged <path>        # ステージから外す\n"
        "  git rm --cached <path>             # 追跡をやめる（ファイルは残る）\n"
        "\n公開してよいファイルなら scripts/check_private.py の GUARDED に追加してください。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())

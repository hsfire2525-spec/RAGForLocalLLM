"""機密領域ガードの検証。

**`.gitignore` だけでは足りない。** 無視ルールは追跡されていないファイルに
しか効かず、`git add -f` されたものやルール追加前から追跡されていたものは
素通りする。実際の事故経路は `git add -A` なので、ステージされた内容を
直接見るガードを別に持つ。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_private.py"
spec = importlib.util.spec_from_file_location("check_private", MODULE_PATH)
assert spec is not None and spec.loader is not None
check_private = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_private)

offending = check_private.offending


def paths_of(problems: list[tuple[str, str]]) -> set[str]:
    return {path for path, _ in problems}


def test_private_tree_is_never_committable() -> None:
    problems = offending(
        [
            "data/private/corpus/社外秘.pdf",
            "data/private/gold/qa_v1.jsonl",
            "data/private/runs/20260801-000000-x-y/predictions.jsonl",
        ]
    )
    assert len(problems) == 3


def test_private_readme_is_allowed() -> None:
    """運用手順を書いた README だけは追跡する。"""
    assert offending(["data/private/README.md"]) == []


def test_new_gold_files_are_blocked_by_default() -> None:
    """**新しい gold は既定で弾く。**

    機密資料の gold には本文の引用が入る。許可制にしておかないと、
    `git add -A` で素通りする。
    """
    assert paths_of(offending(["data/gold/社外秘_qa.jsonl"])) == {"data/gold/社外秘_qa.jsonl"}


def test_public_gold_files_remain_allowed() -> None:
    """IPAガイドライン（公開物）由来の gold は引用ごとコミットする。

    引用を残すことで引用解決率の検証が再現できる。
    """
    allowed = [
        "data/gold/qa_v1.jsonl",
        "data/gold/qa_v2.jsonl",
        "data/gold/sample_qa.jsonl",
    ]
    assert offending(allowed) == []


def test_new_review_files_are_blocked() -> None:
    """人手判定にはモデルの回答（＝本文由来の文字列）が入る。"""
    assert paths_of(offending(["data/reviews/private.jsonl"])) == {"data/reviews/private.jsonl"}


def test_corpus_body_is_blocked_but_lock_and_sample_are_not() -> None:
    problems = offending(
        [
            "data/corpus/ipa_sme_guideline_v4.0.pdf",
            "data/corpus/README.md",
            "data/corpus/corpus.lock.json",
            "data/corpus/sample/security_policy_sample.md",
        ]
    )
    assert paths_of(problems) == {"data/corpus/ipa_sme_guideline_v4.0.pdf"}


def test_run_records_and_cache_are_blocked() -> None:
    """ラン記録には回答が、キャッシュにはチャンク本文がそのまま入る。"""
    problems = offending(
        [
            "runs/20260801-000000-x-y/predictions.jsonl",
            ".cache/indexes/abc/chunks.jsonl",
        ]
    )
    assert len(problems) == 2


def test_ordinary_source_files_pass() -> None:
    assert offending(["src/ragforlocalllm/cli.py", "configs/baseline.yaml", "README.md"]) == []

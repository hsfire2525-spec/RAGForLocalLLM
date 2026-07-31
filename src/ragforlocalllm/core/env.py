"""実行環境の記録。

``env.json`` に残す情報を集める。ここが欠けた実験結果は再現不能で
あり無価値になる（docs/design/design.md §6.5）。特に:

- 環境ラベル（環境1 / 環境2 の区別）
- LM Studio 側のモデルID・量子化・コンテキスト長
- コーパスの SHA-256（コーパスはコミットされないため）
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "numpy",
    "pydantic",
    "faiss-cpu",
    "rank-bm25",
    "sentence-transformers",
    "torch",
    "sudachipy",
    "pymupdf",
    "pypdf",
    "pdfplumber",
    "httpx",
)


def collect_env(
    *,
    label: str | None = None,
    corpus_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ランレコード用の環境情報を返す。"""
    env: dict[str, Any] = {
        "label": label or detect_label(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "gpu": detect_gpu(),
        "packages": installed_versions(),
        "git": git_info(),
    }
    if corpus_path is not None:
        env["corpus"] = corpus_info(corpus_path)
    if extra:
        env.update(extra)
    return env


def detect_label() -> str:
    """環境ラベル。``RAG_ENV_LABEL`` で明示指定できる。

    2環境（Intel内蔵GPU / RTX 5060 Ti）を比較するため、
    ラベルは実験の第一級の軸として扱う。自動判定は補助であり、
    実験時は明示指定を推奨する。
    """
    explicit = os.environ.get("RAG_ENV_LABEL")
    if explicit:
        return explicit
    gpu = detect_gpu()
    if gpu and gpu.get("names"):
        first = str(gpu["names"][0]).lower().replace(" ", "-")
        return f"gpu-{first}"
    return "cpu"


def detect_gpu() -> dict[str, Any] | None:
    """NVIDIA GPU があれば名前とVRAMを返す。無ければ None。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None

    names: list[str] = []
    memory: list[str] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts:
            names.append(parts[0])
        if len(parts) > 1:
            memory.append(parts[1])
    if not names:
        return None
    return {"names": names, "memory_total": memory}


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def git_info() -> dict[str, Any]:
    """コミットと作業ツリーの汚れ具合。

    dirty な状態の実験結果は厳密には再現できないため、記録して
    後から判別できるようにする。
    """
    if shutil.which("git") is None:
        return {"available": False}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {"available": False}
    return {"available": True, "commit": commit, "dirty": bool(status.strip())}


def corpus_info(path: Path) -> dict[str, Any]:
    """コーパスの同一性。

    コーパス本体をコミットできないため、「どの版のどのファイルに
    対する結果か」はハッシュでのみ特定できる。

    コーパスは単一ファイル（PDF）とディレクトリ（テスト用の合成
    コーパス）の両方がありうるため、どちらでも同じ形の情報を返す。
    """
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return {
            "path": str(path),
            "exists": True,
            "kind": "directory",
            "n_files": len(files),
            "size_bytes": sum(p.stat().st_size for p in files),
            "sha256": corpus_signature(path),
        }
    return {
        "path": str(path),
        "exists": True,
        "kind": "file",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }


def corpus_signature(path: Path) -> str:
    """ファイル・ディレクトリのどちらでも使えるコーパスの同一性ハッシュ。

    インデックス署名とランレコードで**同じ値**を使う必要がある。
    別々に算出すると、同じコーパスなのに再利用判定と記録が食い違う。
    """
    path = Path(path)
    if path.is_file():
        return sha256_of(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode("utf-8"))
        digest.update(sha256_of(file).encode("ascii"))
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _total_ram_gb() -> float | None:
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def rss_mb() -> float | None:
    """現在のプロセスRSS（MB）。段ごとのメモリ観測に使う。"""
    return _proc_status_mb("VmRSS")


def peak_rss_mb() -> float | None:
    """プロセス開始以降のピークRSS（MB）。"""
    return _proc_status_mb("VmHWM")


def _proc_status_mb(field: str) -> float | None:
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(f"{field}:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None

"""コンテンツハッシュによるキャッシュ。

初期から必須。これがないと実験ごとにコーパス全体の再埋め込みが
発生し、試行回数を確保できない（docs/design/design.md §4.5）。

小さな値は SQLite に JSON で、埋め込み行列のような大きな値は
``.cache/blobs/`` の .npy ファイルに置く。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CACHE_DIR = Path(".cache")


def content_key(*parts: Any) -> str:
    """引数列からコンテンツハッシュを作る。

    dict はキー順に正規化するため、同じ内容なら順序に依存しない。
    """
    canonical = json.dumps(
        parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Cache:
    """名前空間付きの永続キャッシュ。"""

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_dir = self.root / "blobs"
        self.blob_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.root / "cache.sqlite", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                namespace TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (julianday('now')),
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # JSON 値
    # ------------------------------------------------------------------

    def get_json(self, namespace: str, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put_json(self, namespace: str, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO entries (namespace, key, value) VALUES (?, ?, ?)",
                (namespace, key, payload),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 配列（埋め込み等）
    # ------------------------------------------------------------------

    def get_array(self, namespace: str, key: str) -> np.ndarray | None:
        path = self._blob_path(namespace, key)
        if not path.exists():
            return None
        loaded: np.ndarray = np.load(path)
        return loaded

    def put_array(self, namespace: str, key: str, value: np.ndarray) -> None:
        path = self._blob_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npy.part")
        # np.save はパス文字列を渡すと拡張子 .npy を自動付加してしまうため、
        # 一時ファイル名を保つようファイルオブジェクトを渡す。
        with tmp.open("wb") as fh:
            np.save(fh, value, allow_pickle=False)
        tmp.replace(path)

    def _blob_path(self, namespace: str, key: str) -> Path:
        # キーを2階層に分けて1ディレクトリ内のファイル数を抑える
        return self.blob_dir / namespace / key[:2] / f"{key}.npy"

    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace, COUNT(*) FROM entries GROUP BY namespace"
            ).fetchall()
        counts = {ns: int(n) for ns, n in rows}
        blobs = sum(1 for _ in self.blob_dir.rglob("*.npy"))
        if blobs:
            counts["_blobs"] = blobs
        return counts

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullCache(Cache):
    """キャッシュを無効化する実装。テストと ``--no-cache`` 用。"""

    def __init__(self) -> None:
        self.root = Path()
        self.blob_dir = Path()

    def get_json(self, namespace: str, key: str) -> Any | None:
        return None

    def put_json(self, namespace: str, key: str, value: Any) -> None:
        return None

    def get_array(self, namespace: str, key: str) -> np.ndarray | None:
        return None

    def put_array(self, namespace: str, key: str, value: np.ndarray) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        return None

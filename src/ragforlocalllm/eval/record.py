"""ランレコード（``runs/`` 以下）の読み書き。

**環境情報が欠けた実験結果は再現不能であり無価値**（docs/design/design.md §6.5）。
特にコーパスがリポジトリにコミットされないため、コーパスのSHA-256を
残さないと、後から「どのPDFに対する数値なのか」が永久に分からなくなる。

```
runs/20260731-143022-baseline-a1b2c3/
  config.resolved.yaml    # extends を解決した最終設定
  env.json                # 環境ラベル・モデルID・コーパスSHA-256・git
  predictions.jsonl       # 質問ごとの回答・使用コンテキスト・trace
  metrics.json            # 全体 + question_type / tags 別の層別集計
  human_review.jsonl      # 人手抽出検査の結果（実施した場合）
```
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

RUNS_ROOT = Path("runs")

CONFIG_FILE = "config.resolved.yaml"
ENV_FILE = "env.json"
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"
REVIEW_FILE = "human_review.jsonl"


@dataclass(frozen=True)
class RunRecord:
    """1回の評価実行に対応するディレクトリ。"""

    directory: Path

    @property
    def name(self) -> str:
        return self.directory.name

    # -- 書き込み ------------------------------------------------------

    def write_config(self, resolved: dict[str, Any]) -> None:
        self._write_text(CONFIG_FILE, yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False))

    def write_env(self, env: dict[str, Any]) -> None:
        self._write_json(ENV_FILE, env)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        self._write_json(METRICS_FILE, metrics)

    def write_predictions(self, predictions: list[dict[str, Any]]) -> None:
        self._write_jsonl(PREDICTIONS_FILE, predictions)

    def append_review(self, entry: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / REVIEW_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # -- 読み込み ------------------------------------------------------

    def read_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = self._read_json(METRICS_FILE)
        return payload

    def read_env(self) -> dict[str, Any]:
        payload: dict[str, Any] = self._read_json(ENV_FILE)
        return payload

    def read_predictions(self) -> list[dict[str, Any]]:
        return self._read_jsonl(PREDICTIONS_FILE)

    def read_reviews(self) -> list[dict[str, Any]]:
        path = self.directory / REVIEW_FILE
        return self._read_jsonl(REVIEW_FILE) if path.exists() else []

    @property
    def exists(self) -> bool:
        return (self.directory / METRICS_FILE).exists()

    # -- 内部 ----------------------------------------------------------

    def _write_text(self, filename: str, text: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / filename).write_text(text, encoding="utf-8")

    def _write_json(self, filename: str, payload: Any) -> None:
        self._write_text(
            filename, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        )

    def _write_jsonl(self, filename: str, rows: list[dict[str, Any]]) -> None:
        self._write_text(
            filename,
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        )

    def _read_json(self, filename: str) -> Any:
        return json.loads((self.directory / filename).read_text(encoding="utf-8"))

    def _read_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.directory / filename
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def create_run(
    name: str,
    config_hash: str,
    *,
    root: Path | str = RUNS_ROOT,
    now: datetime | None = None,
) -> RunRecord:
    """新しいランのディレクトリを作る。

    ディレクトリ名に設定ハッシュを含めるのは、**同じ名前で中身が違う実験**を
    後から区別するため。名前だけでは上書き比較の事故が起きる。
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    directory = Path(root) / f"{stamp}-{name}-{config_hash}"
    directory.mkdir(parents=True, exist_ok=True)
    return RunRecord(directory)


def find_runs(root: Path | str = RUNS_ROOT) -> list[RunRecord]:
    """完了したランを新しい順に返す。"""
    base = Path(root)
    if not base.exists():
        return []
    records = [RunRecord(p) for p in base.iterdir() if p.is_dir()]
    return sorted((r for r in records if r.exists), key=lambda r: r.name, reverse=True)


def resolve_run(reference: str, *, root: Path | str = RUNS_ROOT) -> RunRecord:
    """パスまたはラン名（前方一致）からランを特定する。"""
    as_path = Path(reference)
    if as_path.is_dir():
        return RunRecord(as_path)

    candidates = [r for r in find_runs(root) if r.name.startswith(reference)]
    if not candidates:
        raise FileNotFoundError(f"ランが見つかりません: {reference}")
    if len(candidates) > 1:
        names = "\n  ".join(r.name for r in candidates[:5])
        raise ValueError(f"ランの指定が曖昧です: {reference}\n  {names}")
    return candidates[0]

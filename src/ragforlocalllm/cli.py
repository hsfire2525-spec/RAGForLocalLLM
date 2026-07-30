"""コマンドラインインターフェース。

Phase 0 の範囲では ``index`` / ``query`` / ``env`` / ``components`` /
``gold`` が動く。``eval`` / ``sweep`` / ``report`` / ``review`` は
Phase 1 で実装する（未実装であることを明示して終了する）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ragforlocalllm import stages  # noqa: F401 - レジストリ登録のため
from ragforlocalllm.core import registry
from ragforlocalllm.core.cache import Cache, NullCache
from ragforlocalllm.core.config import ConfigError, ExperimentConfig, load_config
from ragforlocalllm.core.env import collect_env
from ragforlocalllm.core.indexing import build_index
from ragforlocalllm.core.pipeline import QueryPipeline
from ragforlocalllm.eval.dataset import load_gold

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="性能の低いローカルLLM向け RAG 実験フレームワーク",
)
console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="実験設定YAML（configs/ 以下）")]
NoCacheOption = Annotated[bool, typer.Option("--no-cache", help="キャッシュを使わない")]


def _cache(no_cache: bool) -> Cache:
    return NullCache() if no_cache else Cache()


def _load(config_path: Path) -> ExperimentConfig:
    try:
        return load_config(config_path, search_dir=Path("configs"))
    except ConfigError as exc:
        console.print(f"[red]設定エラー[/red]\n{exc}")
        raise typer.Exit(code=2) from exc


@app.command("index")
def cmd_index(
    config: ConfigOption,
    force: Annotated[
        bool, typer.Option("--force", help="既存インデックスを無視して再構築")
    ] = False,
    no_cache: NoCacheOption = False,
) -> None:
    """コーパスからインデックスを構築する。"""
    cfg = _load(config)
    with _cache(no_cache) as cache:
        built = build_index(cfg, cache=cache, force=force)

    table = Table(title=f"index: {cfg.name}", show_header=False)
    table.add_row("signature", built.signature)
    table.add_row("directory", str(built.directory))
    for key, value in built.stats.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command("query")
def cmd_query(
    question: Annotated[str, typer.Argument(help="質問文")],
    config: ConfigOption,
    show_contexts: Annotated[
        bool, typer.Option("--contexts", help="使用コンテキストを表示")
    ] = False,
    show_trace: Annotated[bool, typer.Option("--trace", help="段ごとの trace を表示")] = True,
    as_json: Annotated[bool, typer.Option("--json", help="QueryState を JSON で出力")] = False,
    no_cache: NoCacheOption = False,
) -> None:
    """1件の質問をパイプラインに通す。"""
    cfg = _load(config)
    with _cache(no_cache) as cache:
        built = build_index(cfg, cache=cache)
        pipeline = QueryPipeline.from_config(cfg, embedder=built.embedder, index=built.index)
        state = pipeline.run(question)

    if as_json:
        console.print_json(state.model_dump_json())
        return

    answer = state.answer
    console.print(f"\n[bold]質問[/bold] {question}")
    if answer is not None:
        style = "yellow" if answer.abstained else "green"
        console.print(f"[bold {style}]回答[/bold {style}] {answer.text}")
        if answer.citations:
            console.print(f"[dim]引用: {', '.join(answer.citations)}[/dim]")

    if show_contexts:
        ctx_table = Table(title="コンテキスト（提示順）")
        ctx_table.add_column("#", justify="right")
        ctx_table.add_column("chunk_id")
        ctx_table.add_column("score", justify="right")
        ctx_table.add_column("先頭")
        for i, item in enumerate(state.contexts, start=1):
            head = item.chunk.text.replace("\n", " ")[:60]
            ctx_table.add_row(str(i), item.chunk.chunk_id, f"{item.score:.4f}", head)
        console.print(ctx_table)

    if show_trace:
        trace_table = Table(title=f"trace（合計 {state.total_duration_ms:.1f} ms）")
        trace_table.add_column("段")
        trace_table.add_column("実装")
        trace_table.add_column("ms", justify="right")
        trace_table.add_column("RSS MB", justify="right")
        trace_table.add_column("info")
        for entry in state.trace:
            trace_table.add_row(
                entry.stage,
                entry.impl,
                f"{entry.duration_ms:.1f}",
                "-" if entry.rss_mb is None else f"{entry.rss_mb:.0f}",
                json.dumps(entry.info, ensure_ascii=False),
            )
        console.print(trace_table)


@app.command("env")
def cmd_env(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="設定YAML（省略可）")
    ] = None,
    label: Annotated[str | None, typer.Option("--label", help="環境ラベルを明示指定")] = None,
) -> None:
    """ランレコードに記録する環境情報を表示する。"""
    corpus = None
    generator_info = None
    if config is not None:
        cfg = _load(config)
        corpus = cfg.corpus
        generator = registry.build("generator", cfg.query.generator.as_spec())
        describe = getattr(generator, "describe", None)
        if callable(describe):
            generator_info = describe()

    env = collect_env(label=label, corpus_path=corpus)
    if generator_info is not None:
        env["generator"] = generator_info
    console.print_json(json.dumps(env, ensure_ascii=False, default=str))


@app.command("components")
def cmd_components() -> None:
    """レジストリに登録済みの実装を一覧する。"""
    table = Table(title="登録済みコンポーネント")
    table.add_column("種別")
    table.add_column("実装")
    for kind in registry.kinds():
        table.add_row(kind, ", ".join(registry.available(kind)))
    console.print(table)


@app.command("gold")
def cmd_gold(
    dataset: Annotated[Path, typer.Argument(help="gold データセット（JSONL）")],
) -> None:
    """gold データセットを検証し、構成を要約する。"""
    try:
        gold = load_gold(dataset)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]gold エラー[/red]\n{exc}")
        raise typer.Exit(code=2) from exc

    summary = gold.summary()
    console.print_json(json.dumps(summary, ensure_ascii=False))

    ratio = gold.unanswerable_ratio
    if ratio < 0.10:
        console.print(
            f"[yellow]警告[/yellow] answerable=false の割合が {ratio:.0%} です。"
            "棄権性能を測るため 10〜20% を推奨します。"
        )


def _not_implemented(name: str, phase: str) -> None:
    console.print(
        f"[yellow]{name} は未実装です（{phase} で実装予定）。[/yellow]\n"
        "docs/design/design.md §8 の実装フェーズを参照してください。"
    )
    raise typer.Exit(code=1)


@app.command("eval")
def cmd_eval(config: ConfigOption) -> None:
    """評価を実行する（Phase 1）。"""
    _not_implemented("eval", "Phase 1")


@app.command("sweep")
def cmd_sweep(config: ConfigOption) -> None:
    """スイープを実行する（Phase 1）。"""
    _not_implemented("sweep", "Phase 1")


@app.command("report")
def cmd_report() -> None:
    """複数ランを比較する（Phase 1）。"""
    _not_implemented("report", "Phase 1")


@app.command("review")
def cmd_review() -> None:
    """人手抽出検査を行う（Phase 1）。"""
    _not_implemented("review", "Phase 1")


if __name__ == "__main__":
    app()

"""コマンドラインインターフェース。

``index`` / ``query`` / ``env`` / ``components`` / ``gold`` / ``eval`` /
``report`` / ``review`` が動く。``sweep`` は Phase 3 で実装する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

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
from ragforlocalllm.eval.draft import draft_candidates, todo_qids, verify_quotes
from ragforlocalllm.eval.record import RUNS_ROOT, find_runs, resolve_run
from ragforlocalllm.eval.review import (
    VERDICT_LABELS,
    VERDICTS,
    JudgmentStore,
    Verdict,
    agreement,
    stratified_sample,
)
from ragforlocalllm.eval.runner import run_evaluation
from ragforlocalllm.experiments.footprint import measure
from ragforlocalllm.experiments.report import compare, shared_qid_count

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="性能の低いローカルLLM向け RAG 実験フレームワーク",
)
console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="実験設定YAML（configs/ 以下）")]
NoCacheOption = Annotated[bool, typer.Option("--no-cache", help="キャッシュを使わない")]
RunsRootOption = Annotated[
    Path,
    typer.Option(
        "--runs-root",
        help="ラン記録の出力先。機密資料では data/private/runs を指定する",
    ),
]


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


gold_app = typer.Typer(no_args_is_help=True, help="gold データセットの検証・起草")
app.add_typer(gold_app, name="gold")


def _chunks_of(config_path: Path, no_cache: bool) -> list[Any]:
    cfg = _load(config_path)
    with _cache(no_cache) as cache:
        built = build_index(cfg, cache=cache)
    chunks = getattr(built.index, "chunks", None)
    if chunks is None:
        console.print("[red]このインデックスはチャンク本文を公開していません[/red]")
        raise typer.Exit(code=2)
    return list(chunks)


@gold_app.command("check")
def cmd_gold_check(
    dataset: Annotated[Path, typer.Argument(help="gold データセット（JSONL）")],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="指定すると引用文が実際に解決するかも検証する"),
    ] = None,
    no_cache: NoCacheOption = False,
) -> None:
    """gold を検証し、構成を要約する。設定を渡すと引用文の解決も確かめる。"""
    try:
        gold = load_gold(dataset)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]gold エラー[/red]\n{exc}")
        raise typer.Exit(code=2) from exc

    console.print_json(json.dumps(gold.summary(), ensure_ascii=False))

    ratio = gold.unanswerable_ratio
    if ratio < 0.10:
        console.print(
            f"[yellow]警告[/yellow] answerable=false の割合が {ratio:.0%} です。"
            "棄権性能を測るため 10〜20% を推奨します。"
        )

    pending = todo_qids(gold)
    if pending:
        console.print(
            f"[yellow]未記入[/yellow] {len(pending)} 件が TODO のままです: "
            f"{', '.join(pending[:10])}"
        )

    if config is None:
        console.print(
            "[dim]--config を付けると、引用文が抽出テキストに解決するかを検証します。"
            "凍結前に必ず実行してください。[/dim]"
        )
        return

    issues = verify_quotes(gold, _chunks_of(config, no_cache))
    if not issues:
        console.print("[green]すべての引用文が解決しました。[/green]")
        return

    table = Table(title=f"解決できない引用 {len(issues)} 件")
    table.add_column("qid")
    table.add_column("状態")
    table.add_column("引用文", overflow="fold")
    table.add_column("対処", overflow="fold")
    for issue in issues:
        table.add_row(issue.qid, issue.status, issue.quote, issue.hint)
    console.print(table)
    raise typer.Exit(code=1)


@gold_app.command("draft")
def cmd_gold_draft(
    config: ConfigOption,
    n: Annotated[int, typer.Option("--n", help="下書きの件数")] = 40,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="出力先（省略時は表示のみ）")
    ] = None,
    no_cache: NoCacheOption = False,
) -> None:
    """チャンクを層別に抽出し、gold の下書きを作る。

    質問と回答は TODO のまま出力する。**採否と文言は人が決める**
    （機械生成の候補は「抽出しやすい質問」に偏るため）。
    """
    chunks = _chunks_of(config, no_cache)
    candidates = draft_candidates(chunks, n)
    if not candidates:
        console.print("[red]下書き候補を作れませんでした[/red]")
        raise typer.Exit(code=2)

    table = Table(title=f"下書き候補 {len(candidates)} 件")
    table.add_column("qid")
    table.add_column("p", justify="right")
    table.add_column("節", overflow="fold")
    table.add_column("抜粋", overflow="fold")
    for candidate in candidates:
        table.add_row(
            candidate.qid,
            str(candidate.chunk.page or "-"),
            (candidate.chunk.section_path or "-")[-40:],
            candidate.passage[:80],
        )
    console.print(table)

    if out is None:
        console.print("[dim]--out で JSONL に書き出せます。[/dim]")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# rag gold draft で生成した下書き。question / answer を人手で記入する。\n")
        fh.write("# 記入後に `rag gold check <file> -c <設定>` で引用文の解決を検証する。\n")
        for candidate in candidates:
            fh.write(json.dumps(candidate.as_gold_row(), ensure_ascii=False) + "\n")
    console.print(f"[green]書き出しました[/green] {out}")
    console.print(
        "[dim]質問タイプの配分（numeric 8 / enumeration 5 / definition 5 / table 4 / "
        "responsibility 4 / reference 3 / procedure 3 / unanswerable 8）は "
        "docs/design/design.md §10.4 を参照。[/dim]"
    )


@app.command("footprint")
def cmd_footprint(config: ConfigOption) -> None:
    """コンポーネントごとの常駐メモリを実測する。

    環境1（iGPU・8GB共有）で同時常駐が成立するかの判断材料。
    """
    result = measure(_load(config))
    table = Table(title="コンポーネント別フットプリント")
    table.add_column("段")
    table.add_column("実装")
    table.add_column("増分 MB", justify="right")
    table.add_column("うち構築時", justify="right")
    table.add_column("累積 RSS MB", justify="right")
    for component in result["components"]:
        if component["built"]:
            table.add_row(
                component["kind"],
                component["impl"],
                str(component["delta_mb"]),
                str(component["build_delta_mb"]),
                str(component["rss_after_mb"]),
            )
        else:
            table.add_row(
                component["kind"],
                component["impl"],
                "[red]構築失敗[/red]",
                "-",
                component["error"][:40],
            )
    console.print(table)
    console.print(f"[bold]同時常駐時の合計 RSS[/bold] {result['resident_total_rss_mb']} MB")
    if result["gpu"]:
        console.print_json(json.dumps(result["gpu"], ensure_ascii=False, default=str))
    console.print(f"[dim]{result['note']}[/dim]")


@app.command("eval")
def cmd_eval(
    config: ConfigOption,
    dataset: Annotated[
        Path | None, typer.Option("--dataset", "-d", help="gold データセット（設定より優先）")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="先頭N件だけ評価する")] = None,
    label: Annotated[str | None, typer.Option("--label", help="環境ラベル")] = None,
    runs_root: RunsRootOption = RUNS_ROOT,
    no_cache: NoCacheOption = False,
) -> None:
    """gold データセットで評価し、ランレコードを書き出す。"""
    cfg = _load(config)
    gold_path = dataset or cfg.eval.dataset
    if gold_path is None:
        console.print(
            "[red]gold データセットが指定されていません[/red]（--dataset か eval.dataset）"
        )
        raise typer.Exit(code=2)

    try:
        gold = load_gold(gold_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]gold エラー[/red]\n{exc}")
        raise typer.Exit(code=2) from exc

    n_total = min(len(gold), limit) if limit is not None else len(gold)
    with _cache(no_cache) as cache, console.status("評価中…") as status:

        def progress(i: int, item: object) -> None:
            status.update(f"評価中… {i}/{n_total}  {getattr(item, 'qid', '')}")

        result = run_evaluation(
            cfg,
            gold,
            cache=cache,
            limit=limit,
            root=runs_root,
            env_label=label,
            on_item=progress,
        )

    console.print(f"[green]完了[/green] {result.record.directory}")
    _print_metrics(result.metrics)


def _print_metrics(metrics: dict[str, Any]) -> None:
    generation: dict[str, Any] = metrics.get("generation") or {}
    retrieval: dict[str, Any] = metrics.get("retrieval") or {}
    resolution: dict[str, Any] = metrics.get("resolution") or {}

    table = Table(title="生成（正答 / 誤答 / 棄権は必ず併記する）", show_header=False)
    ci: dict[str, Any] = generation.get("accuracy_ci") or {}
    table.add_row(
        "正答率",
        f"{generation.get('accuracy')}  [dim]95%CI [{ci.get('ci_low')}, {ci.get('ci_high')}][/dim]",
    )
    table.add_row("誤答率", str(generation.get("error_rate")))
    table.add_row("棄権率", str(generation.get("abstention_rate")))
    table.add_row("棄権 適合率", str(generation.get("abstention_precision")))
    table.add_row("棄権 再現率", str(generation.get("abstention_recall")))
    table.add_row("char F1", str(generation.get("char_f1")))
    console.print(table)

    if retrieval.get("n_measured"):
        rt = Table(title=f"検索（測定できた質問 {retrieval['n_measured']} 件）", show_header=False)
        for key, value in retrieval.items():
            if key.endswith("_ci") or key == "n_measured" or value is None:
                continue
            rt.add_row(key, str(value))
        console.print(rt)

    resolvability = resolution.get("quote_resolvability")
    if resolvability is not None:
        counts: dict[str, Any] = resolution.get("quote_status_counts") or {}
        console.print(
            f"[bold]引用解決率[/bold] {resolvability}  "
            f"[dim]分断={counts.get('split_across_chunks')} "
            f"抽出漏れ={counts.get('missing_from_corpus')}[/dim]"
        )
        if counts.get("missing_from_corpus"):
            console.print(
                "[yellow]警告[/yellow] 本文にも見つからない引用があります。"
                "Loader が情報を落としている可能性が高いです。"
            )


@app.command("runs")
def cmd_runs(
    limit: Annotated[int, typer.Option("--limit", "-n", help="表示件数")] = 20,
    runs_root: RunsRootOption = RUNS_ROOT,
) -> None:
    """ランを新しい順に一覧する。"""
    records = find_runs(runs_root)[:limit]
    if not records:
        console.print(
            f"[dim]{runs_root} にランがありません。`rag eval -c <設定>` を実行してください。[/dim]"
        )
        return
    table = Table(title="ラン")
    table.add_column("ラン")
    table.add_column("件数", justify="right")
    table.add_column("正答率", justify="right")
    table.add_column("誤答率", justify="right")
    table.add_column("棄権率", justify="right")
    for record in records:
        metrics = record.read_metrics()
        generation = metrics.get("generation") or {}
        table.add_row(
            record.name,
            str(metrics.get("n_items")),
            str(generation.get("accuracy")),
            str(generation.get("error_rate")),
            str(generation.get("abstention_rate")),
        )
    console.print(table)


@app.command("report")
def cmd_report(
    runs: Annotated[list[str], typer.Argument(help="比較するラン（先頭が基準）")],
    metrics: Annotated[
        str, typer.Option("--metrics", help="カンマ区切りの指標名")
    ] = "accuracy,error_rate,abstention_rate,char_f1",
    runs_root: RunsRootOption = RUNS_ROOT,
) -> None:
    """複数ランを信頼区間つきで比較する。"""
    try:
        records = [resolve_run(r, root=runs_root) for r in runs]
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    try:
        rows = compare(records, metrics=tuple(m.strip() for m in metrics.split(",") if m.strip()))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    console.print(f"[dim]共通する質問 {shared_qid_count(records)} 件で比較[/dim]")
    for record in records:
        console.print(f"[dim]  {record.name}[/dim]")

    # 折り返しは許すが省略はさせない。信頼区間が「…」で切れると
    # 表の目的（ノイズと差の区別）が果たせなくなる。
    table = Table(title="ラン比較（95% ブートストラップ信頼区間）")
    table.add_column("指標", overflow="fold")
    table.add_column("ラン", overflow="fold")
    table.add_column("値 [CI]", overflow="fold")
    table.add_column("基準との差 [CI]", overflow="fold")
    table.add_column("判定", overflow="fold")
    for row in rows:
        style = "green" if row.verdict == "改善" else "red" if row.verdict == "悪化" else ""
        table.add_row(
            row.metric,
            row.run,
            str(row.interval),
            "-" if row.diff is None else str(row.diff),
            f"[{style}]{row.verdict}[/{style}]" if style else row.verdict,
        )
    console.print(table)
    console.print("[dim]差の信頼区間が0を跨ぐ場合、その差はノイズと区別できません。[/dim]")


@app.command("review")
def cmd_review(
    run: Annotated[str, typer.Argument(help="ランのパスまたは名前（前方一致）")],
    n: Annotated[int, typer.Option("--n", help="検査件数")] = 10,
    stratify: Annotated[
        str | None, typer.Option("--stratify", help="層別に使うフィールド")
    ] = "question_type",
    store_path: Annotated[Path | None, typer.Option("--store", help="人手判定の保存先")] = None,
) -> None:
    """人手抽出検査を行う。過去に判定済みの回答は再判定しない。"""
    try:
        record = resolve_run(run)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    store = JudgmentStore(store_path) if store_path else JudgmentStore()
    rows = record.read_predictions()
    if not rows:
        console.print("[red]predictions.jsonl が空です[/red]")
        raise typer.Exit(code=2)

    sample = stratified_sample(rows, n, stratify_by=stratify)
    pending = [r for r in sample if store.get(r["qid"], r.get("answer")) is None]
    console.print(
        f"抽出 {len(sample)} 件中 {len(sample) - len(pending)} 件は判定済み（再利用）。"
        f" 残り {len(pending)} 件を判定します。\n"
    )

    for i, row in enumerate(pending, start=1):
        _show_for_review(i, len(pending), row)
        verdict = _ask_verdict()
        if verdict is None:
            console.print("[dim]中断しました。ここまでの判定は保存済みです。[/dim]")
            break
        comment = typer.prompt("コメント（任意）", default="", show_default=False)
        judgment = store.record(
            row["qid"], row.get("answer"), verdict, comment=comment, run=record.name
        )
        record.append_review({**judgment.as_dict(), "auto_outcome": row.get("outcome")})

    result = agreement(rows, store)
    console.print("\n[bold]自動採点との一致率[/bold]")
    console.print_json(json.dumps(result, ensure_ascii=False))
    if result["agreement_rate"] is not None and result["agreement_rate"] < 0.8:
        console.print(
            "[yellow]警告[/yellow] 一致率が低いです。自動指標をそのまま構成比較に"
            "使うのは危険です。データセット側を機械採点しやすい形に直すか、"
            "その質問タイプの検査比率を上げてください。"
        )


def _show_for_review(i: int, total: int, row: dict[str, Any]) -> None:
    console.rule(f"{i}/{total}  {row.get('qid')}  [{row.get('question_type')}]")
    console.print(f"[bold]質問[/bold] {row.get('question')}")
    console.print(f"[bold]gold[/bold] {row.get('gold_answer')}")
    style = "yellow" if row.get("abstained") else "cyan"
    console.print(f"[bold {style}]回答[/bold {style}] {row.get('answer')}")
    console.print(f"[dim]自動判定: {row.get('outcome')}  char_f1={row.get('char_f1')}[/dim]")
    console.print(f"[dim]検索: {_retrieval_note(row)}[/dim]")


def _retrieval_note(row: dict[str, Any]) -> str:
    """検索が根拠を取れたかを人手検査の画面に出す。

    **gold の根拠IDをそのまま表示してはいけない。** 検査者はそれを
    「システムが持っていた根拠」と読んでしまい、「根拠があるのに棄権した」と
    誤診する。実際に一致率検査でこの誤読が起きた。棄権や誤答の原因が
    検索側にあるのか生成側にあるのかは、検査で最も知りたいことなので、
    順位まで含めて明示する。
    """
    gold_ids = set(row.get("gold_chunk_ids") or [])
    if not gold_ids:
        return "根拠なし（回答不能な質問）"
    retrieved = [r["chunk_id"] for r in row.get("retrieved") or []]
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_ids:
            in_context = chunk_id in set(row.get("context_chunk_ids") or [])
            note = "" if in_context else "（ただしプロンプトには入っていない）"
            return f"[green]根拠を{rank}位で取得[/green]{note}"
    return f"[red]根拠を取得できず[/red]（上位{len(retrieved)}件に無し。棄権や誤答は検索側の問題）"


def _ask_verdict() -> Verdict | None:
    choices = {str(i): v for i, v in enumerate(VERDICTS, start=1)}
    menu = "  ".join(f"{i}={VERDICT_LABELS[v]}" for i, v in choices.items())
    answer = typer.prompt(f"判定 [{menu}  q=中断]", default="").strip()
    if answer.lower() in ("q", "quit"):
        return None
    return choices.get(answer)


@app.command("sweep")
def cmd_sweep(config: ConfigOption) -> None:
    """スイープを実行する（Phase 3 で実装）。"""
    console.print(
        "[yellow]sweep は未実装です（Phase 3 で実装予定）。[/yellow]\n"
        "docs/design/design.md §8 の実装フェーズを参照してください。"
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

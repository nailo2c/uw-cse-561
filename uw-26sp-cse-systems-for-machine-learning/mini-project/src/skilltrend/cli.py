from __future__ import annotations

import asyncio
import subprocess
import sys

import typer
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from rich.table import Table

from .agent.pipeline import run_pipeline
from .scanner import count_companies, scan_all
from .settings import settings
from .storage import (load_postings, load_run_summaries,
                      postings_missing_extraction)
from .trends import compute_trend, write_report

app = typer.Typer(help="skilltrend — agentic skill-demand trend analyzer.",
                  no_args_is_help=True)
console = Console()


@app.command()
def scan(
    limit: int = typer.Option(None, help="Max postings per company. "
                                          "Defaults to SKILLTREND_MAX_POSTINGS_PER_COMPANY."),
) -> None:
    """Scan all configured ATS providers and append/refresh postings in storage."""
    cfg = settings.load_companies()
    total_companies = count_companies(cfg)

    state = {"fetched": 0, "fail": 0, "last": ""}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]fetched={task.fields[fetched]}[/] "
                   "[red]fail={task.fields[fail]}[/]"),
        TextColumn("[dim]{task.fields[last]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("scanning", total=total_companies,
                                    fetched=0, fail=0, last="")

        def on_progress(done: int, total: int, provider_name: str, slug: str,
                        count: int, err: str | None) -> None:
            if err is None:
                state["fetched"] += count
                state["last"] = f"{provider_name}/{slug} +{count}"
            else:
                state["fail"] += 1
                state["last"] = f"[red]{provider_name}/{slug} FAIL[/]"
            progress.update(task_id, completed=done,
                            fetched=state["fetched"], fail=state["fail"],
                            last=state["last"])

        result = asyncio.run(scan_all(limit_per_company=limit, on_progress=on_progress))

    console.print(f"[bold green]fetched[/]: {result.total_fetched}  "
                  f"[bold cyan]added[/]: {result.added}  "
                  f"[bold yellow]refreshed[/]: {result.refreshed}")
    if result.failures:
        console.print("[red]failures:[/]")
        for prov, slug, err in result.failures:
            console.print(f"  - {prov}/{slug}: {err}")


@app.command()
def extract(
    mode: str = typer.Option("concurrent", help="sequential | concurrent"),
    workers: int = typer.Option(None, help="Concurrent worker count "
                                            "(ignored in sequential mode)."),
    only_missing: bool = typer.Option(True, help="Skip postings already extracted."),
    limit: int = typer.Option(None, help="Cap how many postings to extract this run."),
) -> None:
    """Run LLM-based extraction over postings, with selectable concurrency."""
    workers = workers or settings.workers
    df = postings_missing_extraction() if only_missing else load_postings()
    if df.empty:
        console.print("[yellow]nothing to extract.[/]")
        raise typer.Exit(code=0)
    if limit:
        df = df.head(limit)
    console.print(f"running [bold]{mode}[/] over {len(df)} postings "
                  f"(workers={workers if mode == 'concurrent' else 1}, "
                  f"model={settings.model})")

    state = {"ok": 0, "fail": 0, "latencies": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[green]ok={task.fields[ok]}[/] [red]fail={task.fields[fail]}[/] "
                   "p50={task.fields[p50]:.0f}ms"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("extracting", total=len(df), ok=0, fail=0, p50=0.0)

        def on_progress(done: int, total: int, metric) -> None:
            if metric.ok:
                state["ok"] += 1
                state["latencies"].append(metric.latency_ms)
            else:
                state["fail"] += 1
            p50 = (sorted(state["latencies"])[len(state["latencies"]) // 2]
                   if state["latencies"] else 0.0)
            progress.update(task_id, completed=done,
                            ok=state["ok"], fail=state["fail"], p50=p50)

        summary = asyncio.run(run_pipeline(df, mode=mode, workers=workers,
                                           on_progress=on_progress))

    console.print(
        f"[green]done[/] run_id={summary.run_id}  "
        f"wall={summary.wall_clock_s:.2f}s  "
        f"throughput={summary.throughput_postings_per_s:.2f} postings/s  "
        f"p50={summary.p50_latency_ms:.0f}ms  p95={summary.p95_latency_ms:.0f}ms  "
        f"ok={summary.successful}/{summary.total_postings}"
    )


@app.command()
def trend(
    window_days: int = typer.Option(30, "--window"),
    baseline_days: int = typer.Option(90, "--baseline"),
    top_n: int = typer.Option(15),
    save: bool = typer.Option(True, help="Also write a markdown report to data/reports/."),
) -> None:
    """Compute rising/declining skill report over a time window."""
    report = compute_trend(window_days=window_days, baseline_days=baseline_days, top_n=top_n)
    table = Table(title=f"Top {top_n} rising skills — last {window_days}d "
                        f"vs prior {baseline_days - window_days}d")
    table.add_column("Skill"); table.add_column("Current", justify="right")
    table.add_column("Baseline", justify="right"); table.add_column("Δ share (pp)", justify="right")
    for t in report.rising:
        table.add_row(t.skill, str(t.current_count), str(t.baseline_count), f"{t.delta_pct:+.2f}")
    console.print(table)
    table2 = Table(title=f"Top {top_n} declining skills")
    table2.add_column("Skill"); table2.add_column("Current", justify="right")
    table2.add_column("Baseline", justify="right"); table2.add_column("Δ share (pp)", justify="right")
    for t in report.declining:
        table2.add_row(t.skill, str(t.current_count), str(t.baseline_count), f"{t.delta_pct:+.2f}")
    console.print(table2)
    if save:
        path = write_report(report, settings.reports_dir)
        console.print(f"[dim]wrote {path}[/]")


@app.command()
def runs() -> None:
    """List past extraction runs with their measured metrics."""
    summaries = load_run_summaries()
    if not summaries:
        console.print("[yellow]no runs yet. try `skilltrend extract`.[/]")
        return
    table = Table(title="Extraction runs")
    for col in ("run_id", "mode", "workers", "postings", "wall_s", "throughput",
                "p50_ms", "p95_ms", "prompt_tok", "compl_tok"):
        table.add_column(col)
    for s in summaries:
        table.add_row(s.run_id, s.mode, str(s.workers), str(s.total_postings),
                      f"{s.wall_clock_s:.2f}", f"{s.throughput_postings_per_s:.2f}",
                      f"{s.p50_latency_ms:.0f}", f"{s.p95_latency_ms:.0f}",
                      str(s.total_prompt_tokens), str(s.total_completion_tokens))
    console.print(table)


@app.command()
def tui() -> None:
    """Launch the Textual TUI."""
    from .ui.tui import SkillTrendApp
    SkillTrendApp().run()


@app.command()
def web(
    port: int = typer.Option(8501),
    host: str = typer.Option("0.0.0.0"),
) -> None:
    """Launch the Streamlit web dashboard."""
    from importlib.resources import files
    app_path = files("skilltrend.ui").joinpath("web.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.port", str(port), "--server.address", host,
           "--browser.gatherUsageStats", "false"]
    subprocess.run(cmd, check=False)


@app.command()
def status() -> None:
    """Quick health check: how much data do we have on disk?"""
    p = load_postings()
    summaries = load_run_summaries()
    console.print(f"postings stored:   [bold]{len(p)}[/]")
    if not p.empty:
        console.print(f"unique companies:  {p['company'].nunique()}")
        console.print(f"providers:         {sorted(p['source'].unique().tolist())}")
    console.print(f"extraction runs:   [bold]{len(summaries)}[/]")
    console.print(f"data dir:          {settings.data_dir}")
    console.print(f"model:             {settings.model}")
    console.print(f"base url:          {settings.openai_base_url}")
    console.print(f"fake llm:          {settings.fake_llm}")
    console.print(f"workers (default): {settings.workers}")
    rpm_str = f"{settings.rpm} req/min" if settings.rpm > 0 else "unlimited"
    console.print(f"rpm cap:           {rpm_str}")


if __name__ == "__main__":
    app()

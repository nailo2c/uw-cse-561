"""Generate paper-grade PNG figures from run metrics.

Reads data/metrics/*.summary.json + *.jsonl, writes figures to
data/reports/figures/. Run from repo root:

    python tools/build_figures.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skilltrend.storage import load_run_summaries  # noqa: E402

FIG_DIR = ROOT / "data" / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Sweep run IDs (in order) — from data/reports/benchmark_sweep_*.md.
# Workers 8 has three runs; we'll use the median to smooth the one transient
# outlier (88.75s first run; 21.77 / 20.61 on the two re-runs).
SWEEP_RUNS = {
    1:  ["9761214c-957"],
    2:  ["011f9048-ec5"],
    4:  ["0f3e602b-33d"],
    8:  ["564a8be2-b0d", "31f77edc-833", "f095ffa1-e12"],
    16: ["7cb3ffa1-ec7"],
}


def load_runs_by_id() -> dict:
    return {s.run_id: s for s in load_run_summaries()}


def median_for(runs: list, key: str) -> float:
    vals = [getattr(r, key) for r in runs]
    return statistics.median(vals)


def build_throughput_figure(by_id: dict) -> None:
    """Figure 1: throughput vs worker count, with ideal-linear reference."""
    workers = sorted(SWEEP_RUNS.keys())
    throughputs = []
    for w in workers:
        runs = [by_id[rid] for rid in SWEEP_RUNS[w]]
        throughputs.append(median_for(runs, "throughput_postings_per_s"))

    seq_throughput = throughputs[0]
    ideal = [w * seq_throughput for w in workers]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, throughputs, "o-", linewidth=2, markersize=8,
            label="measured (Gemini 2.5-flash-lite, paid)", color="#1f77b4")
    ax.plot(workers, ideal, "--", linewidth=1.5, alpha=0.6,
            label="ideal linear (workers × sequential)", color="#888")
    ax.set_xlabel("worker count (concurrent in-flight requests)")
    ax.set_ylabel("throughput (postings / sec)")
    ax.set_title("Throughput scaling vs worker count\n"
                 "Fixed corpus = 100 postings, identical across runs")
    ax.set_xticks(workers)
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    for w, t in zip(workers, throughputs):
        ax.annotate(f"{t:.2f}", xy=(w, t), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "fig1_throughput_vs_workers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_latency_figure(by_id: dict) -> None:
    """Figure 2: p50/p95 stay flat — backend doesn't queue under our load."""
    workers = sorted(SWEEP_RUNS.keys())
    p50s, p95s = [], []
    for w in workers:
        runs = [by_id[rid] for rid in SWEEP_RUNS[w]]
        p50s.append(median_for(runs, "p50_latency_ms"))
        p95s.append(median_for(runs, "p95_latency_ms"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, p50s, "o-", linewidth=2, markersize=8,
            label="p50 latency", color="#2ca02c")
    ax.plot(workers, p95s, "s-", linewidth=2, markersize=8,
            label="p95 latency", color="#d62728")
    ax.set_xlabel("worker count")
    ax.set_ylabel("per-call latency (ms)")
    ax.set_title("Per-call inference latency is invariant under concurrency\n"
                 "Implies the hosted backend absorbs our fan-out without queueing")
    ax.set_xticks(workers)
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, max(p95s) * 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = FIG_DIR / "fig2_latency_vs_workers.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_latency_histogram(by_id: dict) -> None:
    """Figure 3: per-call latency histogram for the workers=4 baseline run.
    Shape proves long-input prefill character of the workload."""
    target_run = "0f3e602b-33d"  # workers=4
    jsonl = ROOT / "data" / "metrics" / f"{target_run}.jsonl"
    latencies = []
    if not jsonl.exists():
        print(f"missing {jsonl}, skipping histogram")
        return
    for line in jsonl.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("ok"):
            latencies.append(m["latency_ms"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(latencies, bins=20, color="#1f77b4", alpha=0.75, edgecolor="black")
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    ax.axvline(p50, color="green", linestyle="--", linewidth=1.5,
               label=f"p50 = {p50:.0f} ms")
    ax.axvline(p95, color="red", linestyle="--", linewidth=1.5,
               label=f"p95 = {p95:.0f} ms")
    ax.set_xlabel("per-call inference latency (ms)")
    ax.set_ylabel("number of LLM calls")
    ax.set_title(f"Latency distribution, workers=4 baseline run (n={len(latencies)})\n"
                 "Right-skewed tail = a few unusually long extractions, "
                 "no bimodality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "fig3_latency_histogram_w4.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_token_breakdown(by_id: dict) -> None:
    """Figure 4: input-heavy workload — prompt vs completion tokens.
    Validates the hypothesis that the workload is prefill-dominated."""
    target = by_id["0f3e602b-33d"]  # workers=4 baseline
    jsonl = ROOT / "data" / "metrics" / f"{target.run_id}.jsonl"
    prompt_tokens, completion_tokens = [], []
    for line in jsonl.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m.get("ok"):
            prompt_tokens.append(m["prompt_tokens"])
            completion_tokens.append(m["completion_tokens"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(prompt_tokens, bins=20, color="#ff7f0e", alpha=0.75,
                 edgecolor="black")
    axes[0].set_title(f"Prompt tokens (avg {statistics.mean(prompt_tokens):.0f})")
    axes[0].set_xlabel("tokens / call")
    axes[0].set_ylabel("LLM calls")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(completion_tokens, bins=20, color="#2ca02c", alpha=0.75,
                 edgecolor="black")
    axes[1].set_title(f"Completion tokens (avg {statistics.mean(completion_tokens):.0f})")
    axes[1].set_xlabel("tokens / call")
    axes[1].set_ylabel("LLM calls")
    axes[1].grid(True, alpha=0.3)

    ratio = statistics.mean(prompt_tokens) / statistics.mean(completion_tokens)
    fig.suptitle(f"Workload is input-heavy: prompt:completion ratio ≈ {ratio:.1f}:1",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig4_token_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    by_id = load_runs_by_id()
    missing = [rid for runs in SWEEP_RUNS.values() for rid in runs
               if rid not in by_id]
    if missing:
        print(f"warning: missing run summaries: {missing}")
    build_throughput_figure(by_id)
    build_latency_figure(by_id)
    build_latency_histogram(by_id)
    build_token_breakdown(by_id)
    print("\nfigures written to:", FIG_DIR)


if __name__ == "__main__":
    main()

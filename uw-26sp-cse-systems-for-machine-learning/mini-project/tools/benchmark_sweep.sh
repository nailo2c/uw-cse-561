#!/usr/bin/env bash
# Controlled concurrency sweep for the paper's systems analysis.
#
# Re-extracts the SAME first-N postings 5 times under different worker counts.
# Per-call metrics from each run go into data/metrics/<run_id>.jsonl; the
# run summary into <run_id>.summary.json. Output is a clean markdown table
# you can paste straight into the paper.
#
# Pollution control: we snapshot extractions.csv before the sweep and
# restore it afterwards, so trend statistics don't get distorted by 5×N
# duplicate extractions from the same postings.
#
# Usage:
#   bash tools/benchmark_sweep.sh           # default N=100
#   bash tools/benchmark_sweep.sh 200       # N=200
set -euo pipefail

N=${1:-100}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  echo "no .venv found in $ROOT" >&2
  echo "run: python -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

EXT="data/extractions/extractions.csv"
SNAPSHOT="data/extractions/.benchmark_snapshot.csv"
SWEEP_DIR="data/metrics/sweep-$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="data/reports/benchmark_sweep_$(date -u +%Y%m%dT%H%M%SZ).md"

cleanup() {
  if [ -f "$SNAPSHOT" ]; then
    echo ""
    echo "restoring extractions.csv from snapshot (trend stays clean)…"
    mv "$SNAPSHOT" "$EXT"
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$SWEEP_DIR"

if [ -f "$EXT" ]; then
  cp "$EXT" "$SNAPSHOT"
  echo "snapshotted $EXT -> $SNAPSHOT"
fi

echo ""
echo "=========================================================="
echo "  controlled sweep — N=$N postings, 5 configurations"
echo "=========================================================="

# Format: "mode workers label"
declare -a CONFIGS=(
  "sequential 1  seq-1"
  "concurrent 2  conc-2"
  "concurrent 4  conc-4"
  "concurrent 8  conc-8"
  "concurrent 16 conc-16"
)

RUN_IDS=()

for cfg in "${CONFIGS[@]}"; do
  read -r mode workers label <<< "$cfg"
  echo ""
  echo "─── [$label]  mode=$mode  workers=$workers  N=$N ───"
  # --no-only-missing → re-extract the same first N postings every time
  #                     (Typer/Click bool flag convention; =false is invalid)
  # --limit "$N"      → deterministic head(N) from postings.csv
  skilltrend extract \
    --mode "$mode" \
    --workers "$workers" \
    --limit "$N" \
    --no-only-missing
done

echo ""
echo "=========================================================="
echo "  rendering paper-grade summary table"
echo "=========================================================="

python - <<EOF > "$OUTPUT"
from skilltrend.storage import load_run_summaries

# Last 5 runs are this sweep, in chronological order
summaries = sorted(load_run_summaries(), key=lambda s: s.started_at)[-5:]

print("# Controlled concurrency sweep")
print()
print(f"Corpus: first **${N}** postings of \`postings.csv\` (deterministic head).")
print(f"Model:  \`{summaries[0].model}\`")
print(f"Date:   {summaries[0].started_at} → {summaries[-1].finished_at} (UTC)")
print()
print("| Mode | Workers | Wall (s) | Throughput (postings/s) | p50 (ms) | p95 (ms) | Σ prompt tok | Σ completion tok | ok/total |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for s in summaries:
    print(
        f"| {s.mode} | {s.workers} | {s.wall_clock_s:.2f} | "
        f"{s.throughput_postings_per_s:.2f} | "
        f"{s.p50_latency_ms:.0f} | {s.p95_latency_ms:.0f} | "
        f"{s.total_prompt_tokens:,} | {s.total_completion_tokens:,} | "
        f"{s.successful}/{s.total_postings} |"
    )
print()
print("## How to read")
print()
print("- **Throughput**: postings completed per wall-clock second. Should rise with workers up to the saturation point predicted in the hypothesis.")
print("- **p50/p95**: per-call latency (pure inference, excludes client-side rate-limit wait). A flat p50 with rising p95 indicates queueing saturation on the backend.")
print("- **Σ prompt tok**: dominated by job-description input; should stay constant across runs since the corpus is fixed.")
print()
print("## Run IDs (for cross-reference with per-call jsonl logs)")
print()
for s in summaries:
    print(f"- \`{s.run_id}\`  ({s.mode}, workers={s.workers})")
EOF

echo ""
echo "wrote $OUTPUT"
echo ""
cat "$OUTPUT"

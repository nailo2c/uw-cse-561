# Usage

All commands are exposed through the `skilltrend` CLI, installed as a console
script by `pip install -e .`. Use `skilltrend --help` and
`skilltrend <command> --help` for live reference.

## `skilltrend status`

Quick health check. Prints how many postings are stored, which providers
they came from, how many extraction runs you've done, and what backend
config is in effect.

```bash
skilltrend status
```

Run this first after setup to confirm `OPENAI_BASE_URL` / `SKILLTREND_MODEL`
look right.

## `skilltrend scan`

Fetch postings from every company in `config/companies.yaml` and append
them to `data/postings/postings.csv`. Re-running is safe: existing postings
have their `last_seen` bumped instead of being duplicated, and new ones are
appended.

```bash
skilltrend scan                    # uses SKILLTREND_MAX_POSTINGS_PER_COMPANY
skilltrend scan --limit 20         # cap to 20 postings per company
```

Failures on individual companies (private board, slug typo, rate limit) are
reported at the end but never abort the run.

### How often should I scan?

For trend analysis the question is **how fast skill demand actually
changes**, not how fast postings appear. Recommended cadence:

| Goal | Cadence |
|---|---|
| Quick demo / local dev | once before each `extract` (manual) |
| Real trend analysis | **daily** — set up a `cron` / GCE scheduled task |
| Stress-test scan + storage | hourly, but throttle `--limit` so you don't burn through ATS rate limits |

ATSes update their public boards on the order of hours, so anything more
frequent than once an hour mostly produces duplicate snapshots. The
`last_seen` column already captures "still listed on day X", which is what
the time-window analysis actually needs.

### Cron example on GCE

```cron
# /etc/cron.d/skilltrend
0 4 * * *  aaron  cd /opt/skilltrend && docker compose run --rm scan
0 5 * * *  aaron  cd /opt/skilltrend && docker compose run --rm extract
```

## `skilltrend extract`

Run LLM-based skill extraction on postings. Two execution modes:

```bash
skilltrend extract --mode sequential
skilltrend extract --mode concurrent --workers 8
skilltrend extract --limit 100              # cap this run for benchmarking
skilltrend extract --no-only-missing        # re-extract everything
```

Each run gets a unique `run_id`. Per-call metrics land in
`data/metrics/<run_id>.jsonl`, and an aggregate summary in
`data/metrics/<run_id>.summary.json`. These are the inputs the writeup's
systems analysis reads.

## `skilltrend trend`

Compute rising/declining skills over a time window vs a baseline.

```bash
skilltrend trend --window 30 --baseline 90        # last 30d vs prior 60d
skilltrend trend --window 90 --baseline 180
skilltrend trend --window 365 --baseline 730 --top-n 25
```

Prints rising and declining tables to the terminal, and writes a markdown
report to `data/reports/`. Reports are append-only — old ones are not
overwritten so the writeup can cite specific snapshots.

## `skilltrend tui`

Launches the Textual TUI. Bindings:

| Key | Action |
|---|---|
| `tab` / `shift+tab` | Switch window (30d / 90d / 180d / 365d) |
| `↑` / `↓` | Move row |
| `r` | Reload from disk |
| `q` | Quit |

The right pane shows sample evidence postings for the selected skill, and
the bottom pane summarises your last few extraction runs (this is the
"systems context" view).

## `skilltrend web`

Launches the Streamlit dashboard:

```bash
skilltrend web                  # http://localhost:8501
skilltrend web --port 8080      # custom port
```

In Docker, `docker compose up -d web` does the same but as a long-running
service. The container exposes 8501 — open `http://<vm-ip>:8501` on GCE.

## `skilltrend runs`

List past extraction runs and their measured throughput / latency.

```bash
skilltrend runs
```

Useful before writing up the systems analysis — it's the canonical answer
to "did concurrent beat sequential, and by how much?"

## Mapping to the final writeup

Concrete command sequences for each writeup question:

### "What is the application and the design pattern?"

The architecture diagram in [README.md](../README.md) covers it. The agent
design pattern is **planning + multi-agent collaboration + dynamic tool
orchestration**: the scanner orchestrates one provider per ATS; the
extractor is the LLM tool; the normalizer is a deterministic refinement
step; the trend analyzer produces the final report.

### "What is the primary system bottleneck?"

Produce the sequential vs concurrent comparison:

```bash
skilltrend scan
# fix the corpus so both runs see identical inputs
cp data/postings/postings.csv data/postings/benchmark.csv

skilltrend extract --mode sequential --limit 200
skilltrend extract --mode concurrent --workers 4 --limit 200
skilltrend extract --mode concurrent --workers 8 --limit 200
skilltrend extract --mode concurrent --workers 16 --limit 200
skilltrend runs
```

Compare `throughput`, `p50`, `p95` across runs. The hypothesis is that
prefill latency dominates and saturates around N workers — the table is
your evidence.

### "What are the optimization opportunities?"

Each row in `data/metrics/<run_id>.jsonl` records prompt_tokens. The
extractor already separates a stable taxonomy-bearing system prompt from
the per-posting user prompt, so swapping in a prompt-caching-aware backend
will give you a measurable hit-rate. The `taxonomy.yaml` normalizer
illustrates the "replace LLM calls with deterministic preprocessing" lever.
Documented further in the README's writeup-mapping table.

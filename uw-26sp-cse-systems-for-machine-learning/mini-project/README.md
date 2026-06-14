# skilltrend — agentic skill-demand trend analyzer

MVP for the UW CSE Systems-for-ML final project. Scans public ATS endpoints
for job postings, extracts skills with an agentic LLM pipeline, computes
which skills are rising and declining over configurable time windows, and
serves the results through a CLI, a Textual TUI, and a Streamlit web UI.

```
ATS providers ─► raw postings (CSV)
                    │
                    ▼
         agentic extractor (LLM)
          ├─ sequential or
          └─ concurrent workers ─► extractions (CSV) + per-call metrics (JSONL)
                    │
                    ▼
          time-window trend analysis
          ├─ CLI report (markdown)
          ├─ Textual TUI
          └─ Streamlit web UI
```

## Quick start (local)

```bash
# 1. Set up the environment
cp .env.example .env
# edit .env: at minimum set OPENAI_API_KEY (or set SKILLTREND_FAKE_LLM=true
# for an offline smoke test that uses keyword-based extraction)

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. Run the pipeline end-to-end
skilltrend scan                    # fetch postings from configured ATS
skilltrend extract --mode concurrent --workers 4
skilltrend trend --window 30 --baseline 90
skilltrend tui                     # interactive TUI
skilltrend web                     # Streamlit on http://localhost:8501
```

See [docs/usage.md](docs/usage.md) for the full command reference and
suggested scan cadence.

## Quick start (Docker / single VM)

```bash
cp .env.example .env
docker compose build

# one-off jobs (cron-friendly on GCE)
docker compose run --rm scan
docker compose run --rm extract
docker compose run --rm trend

# long-running web UI
docker compose up -d web           # exposes 8501

# interactive TUI inside the container
docker compose run --rm tui
```

`./data` is bind-mounted into the container, so every CSV/JSONL artifact
persists on the host. On GCE you can clone the repo, point `./data` at a
persistent disk, set `.env`, and the whole stack works without any other
service — exactly the "all in one VM" setup requested.

## Repo layout

```
src/skilltrend/
  cli.py            Typer entrypoint (scan / extract / trend / tui / web / runs / status)
  settings.py       Pydantic settings (env + config files)
  models.py         Pydantic data models (Posting / Extraction / RunMetric)
  storage.py        CSV + JSONL persistence
  scanner.py        Provider orchestration
  providers/
    base.py         Common interface
    greenhouse.py   Public Greenhouse boards API
    lever.py        Public Lever postings API
    ashby.py        Public Ashby job board API
    workday.py      Public Workday CXS API
    amazon.py       Amazon Jobs custom API
    apple.py        Apple Jobs API with CSRF bootstrap
    google.py       Google Careers HTML parser
    microsoft.py    Microsoft Careers API + detail JSON-LD
    meta.py         Meta Careers GraphQL + detail payload parser
    filters.py      Shared provider-side title/location include/exclude filters
    jobspy_provider.py  LinkedIn / aggregator fallback
  llm.py            OpenAI-compatible async client (+ deterministic fake mode)
  agent/
    extractor.py    Single-posting LLM extraction with structured output
    normalizer.py   Alias-based canonical skill names
    pipeline.py     Sequential + concurrent execution modes (the systems comparison)
  trends.py         Time-window trend computation
  ui/
    tui.py          Textual app
    web.py          Streamlit app

config/
  companies.yaml    ATS provider -> companies to scan
  taxonomy.yaml     Canonical skills + aliases (used by normalizer)

data/               Bind-mounted in Docker; CSVs/JSONL written here
docs/usage.md       Full command reference & operational cadence
```

## How the MVP supports the final writeup

The project rubric asks for **application-level performance** and a
**systems-level analysis of the inference engine**. The codebase is built so
each writeup section has a concrete artifact to point at:

| Writeup section | Artifact in this codebase |
|---|---|
| Application + design pattern | `scanner.py` (multi-agent collaboration via providers), `agent/pipeline.py` (planning + worker decomposition), `agent/extractor.py` (structured tool output), `agent/normalizer.py` (taxonomy-bounded refinement) |
| Execution plan | `Dockerfile` + `docker-compose.yml` for reproducible runs; `llm.py` swappable backend (OpenAI / vLLM / Ollama via `OPENAI_BASE_URL`) |
| App-level metrics | `RunSummary` written to `data/metrics/<run_id>.summary.json`: wall_clock_s, throughput_postings_per_s, p50/p95 latency, prompt/completion token totals |
| Inference engine profiling | Per-call metrics in `data/metrics/<run_id>.jsonl`: per-posting latency, token counts. Re-running `extract --mode sequential` vs `--mode concurrent --workers N` directly produces the comparison points the hypothesis predicts |
| Optimization opportunities | Concurrency sweep, prompt-prefix vs per-posting separation (system prompt vs user prompt split is already factored), and the `taxonomy.yaml`-based deterministic normalizer that takes load off the LLM |

See [docs/usage.md#mapping-to-the-final-writeup](docs/usage.md#mapping-to-the-final-writeup)
for the exact commands that produce each figure.

## Configuration

Everything is env-driven via `.env`. Highlights:

- `OPENAI_BASE_URL` — point at your vLLM-on-TPU instance, an Ollama daemon,
  or the OpenAI API. The rest of the code is unchanged.
- `SKILLTREND_MODEL` — model name passed to the backend.
- `SKILLTREND_WORKERS` — default concurrent worker count for `extract`.
- `SKILLTREND_FAKE_LLM=true` — bypass the LLM entirely, return a deterministic
  result. Useful for CI / offline smoke tests.

Companies and the skill taxonomy are in `config/*.yaml` — no code changes
needed to add a new ATS-hosted company or a new canonical skill.

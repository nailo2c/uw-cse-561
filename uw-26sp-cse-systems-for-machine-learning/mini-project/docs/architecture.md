# Architecture notes

Short companion to the README. Records design choices that aren't obvious
from the code and matter for the final writeup.

## Why CSV + JSONL instead of SQLite

For the MVP the data layer is intentionally trivial: pandas reads/writes
`postings.csv` and `extractions.csv`; metrics are JSONL append-only.

Reasons:

1. Project scope explicitly allows it ("CSV initially is fine"). Adding
   SQLite at this stage is premature complexity.
2. Reproducibility for the writeup is easier when the dataset is a checked-
   in (or rsynced) flat file.
3. The `storage.py` module is the only place that knows about CSV — porting
   to SQLite later is a localised change.

The deal-breaker for CSV would be concurrent writes to the same file. The
scan path writes once at the end of the scan; the extract path appends in
bulk after the run; metrics use a per-run JSONL file. None of these contend
with each other.

## Why one CSV per kind, not per scan

The proposal's trend analysis needs to compare windows. Sharding postings
by scan date would force the analyzer to merge across files for every
query. One flat `postings.csv` with `first_seen` / `last_seen` columns
keeps the queries declarative.

## Sequential vs concurrent in one file

`agent/pipeline.py` keeps both modes in the same module on purpose. The
writeup's systems comparison is essentially a diff between the two paths
— having them side-by-side in code makes the "what changed?" answer
trivially copy-paste-able.

## Agent design pattern mapping

The proposal listed four candidate patterns. The MVP exhibits:

- **Planning and decomposition** — the CLI plans the macro task (scan ->
  extract -> trend); `pipeline.py` plans the micro task (which postings,
  in what order, with what concurrency).
- **Multi-agent collaboration** — each ATS provider is an independent agent
  with the same interface. They run concurrently and merge into one
  storage layer.
- **Dynamic tool orchestration** — the LLM extractor receives a structured
  output schema and is invoked as a tool by `pipeline.py`. The normalizer
  is a deterministic "tool" that runs after the LLM.
- **Reflection / self-correction** — not in v1. The hook for it is in
  `agent/extractor.py`: a second pass that re-prompts the model when
  `evidence` is missing for a claimed skill. Adding it gives a third data
  point for the writeup's quality-vs-latency tradeoff.

## What the cloud migration looks like

The README claims "no rearchitecting needed to move to GCE". Concretely:

1. `git clone` the repo on the VM.
2. `cp .env.example .env`, fill in `OPENAI_API_KEY`. If running vLLM on the
   same VM (or another GCE TPU), set `OPENAI_BASE_URL=http://<vllm>:8000/v1`.
3. `docker compose up -d web` — Streamlit on port 8501.
4. Add the cron stanza in `docs/usage.md` for daily scan + extract.
5. `./data` is the only persistent state. Snapshot or back it up to GCS
   periodically; nothing else on the VM is stateful.

The "open a database container" question deliberately does not arise in
v1 — everything is files in `./data`.

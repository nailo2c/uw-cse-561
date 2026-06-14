"""CSV-backed storage layer.

Kept deliberately simple: pandas read/write for analytical access, plus a
streaming JSONL writer for per-call metrics. The whole module can later be
replaced by a SQLite or Postgres adapter without touching the rest of the
code, because callers only ever import the helpers defined here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import Extraction, Posting, RunMetric, RunSummary
from .settings import settings

POSTING_COLUMNS = [
    "posting_id", "url", "company", "title", "location", "source",
    "first_seen", "last_seen", "active", "posted_at", "description",
]

EXTRACTION_COLUMNS = [
    "posting_id", "run_id", "extracted_at", "role_family", "seniority",
    "domain_tags", "required_skills", "preferred_skills",
    "model", "prompt_tokens", "completion_tokens", "latency_ms",
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_postings() -> pd.DataFrame:
    path = settings.postings_csv
    if not path.exists():
        return pd.DataFrame(columns=POSTING_COLUMNS)
    df = pd.read_csv(path)
    if "active" in df.columns:
        df["active"] = df["active"].astype(bool)
    return df


def upsert_postings(new: list[Posting]) -> tuple[int, int]:
    """Append new postings and update last_seen for postings we've seen before.

    Returns (added, refreshed). Past snapshots are preserved — refreshed only
    bumps the most-recent snapshot's last_seen so trend windows have an
    accurate "still active on date X" signal.
    """
    if not new:
        return 0, 0
    df = load_postings()
    existing_ids = set(df["posting_id"]) if not df.empty else set()
    added = 0
    refreshed = 0
    rows_to_add: list[dict] = []
    for p in new:
        row = p.model_dump()
        if p.posting_id in existing_ids:
            mask = df["posting_id"] == p.posting_id
            df.loc[mask, "last_seen"] = p.last_seen
            df.loc[mask, "active"] = p.active
            refreshed += 1
        else:
            rows_to_add.append(row)
            added += 1
    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add, columns=POSTING_COLUMNS)], ignore_index=True)
    _ensure_parent(settings.postings_csv)
    df.to_csv(settings.postings_csv, index=False)
    return added, refreshed


def load_extractions() -> pd.DataFrame:
    path = settings.extractions_csv
    if not path.exists():
        return pd.DataFrame(columns=EXTRACTION_COLUMNS)
    df = pd.read_csv(path)
    return df


def append_extractions(items: list[Extraction]) -> int:
    if not items:
        return 0
    rows = []
    for e in items:
        rows.append({
            "posting_id": e.posting_id,
            "run_id": e.run_id,
            "extracted_at": e.extracted_at,
            "role_family": e.role_family,
            "seniority": e.seniority,
            "domain_tags": json.dumps(e.domain_tags),
            "required_skills": json.dumps([s.model_dump() for s in e.required_skills]),
            "preferred_skills": json.dumps([s.model_dump() for s in e.preferred_skills]),
            "model": e.model,
            "prompt_tokens": e.prompt_tokens,
            "completion_tokens": e.completion_tokens,
            "latency_ms": e.latency_ms,
        })
    new = pd.DataFrame(rows, columns=EXTRACTION_COLUMNS)
    path = settings.extractions_csv
    _ensure_parent(path)
    if path.exists():
        new.to_csv(path, mode="a", header=False, index=False)
    else:
        new.to_csv(path, index=False)
    return len(rows)


def postings_missing_extraction(run_id: str | None = None) -> pd.DataFrame:
    """Return postings that have no extraction yet (or none for `run_id`)."""
    postings = load_postings()
    if postings.empty:
        return postings
    extractions = load_extractions()
    if extractions.empty:
        return postings
    if run_id is not None:
        extracted_ids = set(extractions.loc[extractions["run_id"] == run_id, "posting_id"])
    else:
        extracted_ids = set(extractions["posting_id"])
    mask = ~postings["posting_id"].isin(extracted_ids)
    return postings[mask].copy()


def metrics_path(run_id: str) -> Path:
    return settings.metrics_dir / f"{run_id}.jsonl"


def append_metric(metric: RunMetric) -> None:
    path = metrics_path(metric.run_id)
    _ensure_parent(path)
    with path.open("a") as f:
        f.write(metric.model_dump_json() + "\n")


def write_run_summary(summary: RunSummary) -> Path:
    path = settings.metrics_dir / f"{summary.run_id}.summary.json"
    _ensure_parent(path)
    path.write_text(summary.model_dump_json(indent=2))
    return path


def load_run_summaries() -> list[RunSummary]:
    out: list[RunSummary] = []
    if not settings.metrics_dir.exists():
        return out
    for p in sorted(settings.metrics_dir.glob("*.summary.json")):
        try:
            out.append(RunSummary.model_validate_json(p.read_text()))
        except Exception:
            continue
    return out

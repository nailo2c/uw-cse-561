"""End-to-end smoke test.

Bypasses real ATS HTTP calls and real LLM calls — exercises the storage /
agent / trend / pipeline plumbing only. Anything that does network I/O is
mocked or stubbed.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_data(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    for sub in ("postings", "extractions", "metrics", "reports"):
        (data / sub).mkdir(parents=True)
    monkeypatch.setenv("SKILLTREND_DATA_DIR", str(data))
    monkeypatch.setenv("SKILLTREND_FAKE_LLM", "true")
    # Reload settings so the new env wins.
    import importlib
    from skilltrend import settings as s_mod
    importlib.reload(s_mod)
    from skilltrend import storage as st_mod
    importlib.reload(st_mod)
    yield data


def _make_posting(i: int):
    from skilltrend.models import Posting, utcnow_iso
    now = utcnow_iso()
    return Posting(
        posting_id=f"pid-{i:03d}",
        url=f"https://example.com/jobs/{i}",
        company="Acme",
        title="Senior ML Engineer",
        location="Remote",
        source="greenhouse",
        first_seen=now,
        last_seen=now,
        active=True,
        posted_at=now,
        description=("We use Python, PyTorch, Kubernetes, AWS, vLLM, and "
                     "are building agentic workflows with RAG and MCP. "
                     "Experience with fine-tuning preferred."),
    )


def test_pipeline_handles_nan_csv_fields(temp_data):
    """When provider scrapes a job with no location, CSV stores it as an
    empty cell, which pandas reads back as NaN. Regression for the
    ValidationError on Pydantic str field."""
    import asyncio
    import pandas as pd
    from skilltrend.agent.pipeline import _row_to_posting, run_pipeline
    from skilltrend.storage import upsert_postings, load_postings

    p = _make_posting(0)
    p.location = ""
    p.description = ""
    upsert_postings([p])

    # Re-read from CSV — empty strings now come back as NaN
    df = load_postings()
    assert df["location"].isna().any() or (df["location"] == "").any()

    posting = _row_to_posting(df.iloc[0])
    assert posting.location == ""
    assert posting.description == ""

    summary = asyncio.run(run_pipeline(df, mode="sequential", workers=1))
    assert summary.successful == 1


def test_storage_roundtrip(temp_data):
    from skilltrend.storage import (load_postings, upsert_postings,
                                    load_extractions)
    postings = [_make_posting(i) for i in range(3)]
    added, refreshed = upsert_postings(postings)
    assert added == 3 and refreshed == 0
    df = load_postings()
    assert len(df) == 3
    # second upsert should refresh, not add
    added2, refreshed2 = upsert_postings(postings)
    assert added2 == 0 and refreshed2 == 3
    assert load_extractions().empty


def test_pipeline_extracts_and_normalizes(temp_data):
    from skilltrend.agent.pipeline import run_pipeline
    from skilltrend.storage import (load_extractions, load_postings,
                                    load_run_summaries, upsert_postings)
    postings = [_make_posting(i) for i in range(5)]
    upsert_postings(postings)
    df = load_postings()
    summary = asyncio.run(run_pipeline(df, mode="concurrent", workers=3))
    assert summary.total_postings == 5
    assert summary.successful == 5
    ext = load_extractions()
    assert len(ext) == 5
    # Canonical normalizer should rewrite e.g. "k8s" -> "Kubernetes"
    import json
    skill_names = set()
    for row in ext.itertuples():
        for s in json.loads(row.required_skills):
            skill_names.add(s["name"])
    assert "Kubernetes" in skill_names
    assert "Python" in skill_names
    summaries = load_run_summaries()
    assert len(summaries) == 1


def test_trend_report(temp_data):
    from skilltrend.agent.pipeline import run_pipeline
    from skilltrend.storage import load_postings, upsert_postings
    from skilltrend.trends import compute_trend
    upsert_postings([_make_posting(i) for i in range(4)])
    asyncio.run(run_pipeline(load_postings(), mode="sequential", workers=1))
    report = compute_trend(window_days=30, baseline_days=90, top_n=5)
    assert report.window_total_postings == 4
    # Rising list should at least include some canonical skills.
    skills = [t.skill for t in report.rising]
    assert any(s in ("Python", "PyTorch", "Kubernetes") for s in skills)

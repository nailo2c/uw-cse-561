"""Trend computation tests that focus on timestamp parsing edge cases."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from skilltrend.trends import _parse_ts


@pytest.mark.parametrize("value, expected", [
    ("2026-05-29T19:34:24Z", datetime(2026, 5, 29, 19, 34, 24, tzinfo=timezone.utc)),
    ("2026-05-28T22:12:28.603Z", datetime(2026, 5, 28, 22, 12, 28, 603000, tzinfo=timezone.utc)),
    ("2026-05-22T16:05:41-04:00", datetime(2026, 5, 22, 20, 5, 41, tzinfo=timezone.utc)),
])
def test_iso_with_offset(value, expected):
    got = _parse_ts(value)
    assert got is not None
    assert got.astimezone(timezone.utc) == expected.astimezone(timezone.utc)


def test_iso_without_offset_treated_as_utc():
    """Microsoft JSON-LD emits naive ISO timestamps; we must promote them to
    UTC so they compare cleanly against the window boundaries."""
    got = _parse_ts("2026-05-29T19:34:24")
    assert got is not None
    assert got.tzinfo is not None
    assert got == datetime(2026, 5, 29, 19, 34, 24, tzinfo=timezone.utc)


def test_epoch_milliseconds():
    got = _parse_ts("1753687796431")
    assert got is not None
    assert got.tzinfo is timezone.utc


@pytest.mark.parametrize("value", [None, "", "   ", "nan", "not-a-date"])
def test_unparseable_returns_none(value):
    assert _parse_ts(value) is None


def test_empty_posted_at_falls_back_to_first_seen_not_last_seen(tmp_path, monkeypatch):
    """A posting with empty posted_at should be anchored to first_seen, so
    that a re-scan today doesn't keep it artificially in the current window
    forever. Regression for the 'Google/Meta postings never age into baseline'
    issue."""
    import importlib
    import pandas as pd
    data_dir = tmp_path / "data"
    for sub in ("postings", "extractions", "metrics", "reports"):
        (data_dir / sub).mkdir(parents=True)
    monkeypatch.setenv("SKILLTREND_DATA_DIR", str(data_dir))
    from skilltrend import settings as s_mod
    importlib.reload(s_mod)
    from skilltrend import storage as st_mod
    importlib.reload(st_mod)
    from skilltrend import trends as t_mod
    importlib.reload(t_mod)

    # posting first observed 60 days ago, scanned again today, no posted_at
    pd.DataFrame([{
        "posting_id": "p1", "url": "u", "company": "Google",
        "title": "SWE", "location": "", "source": "google",
        "first_seen": "2026-03-31T00:00:00Z",   # 60 days before today (2026-05-30)
        "last_seen": "2026-05-30T00:00:00Z",    # today
        "active": True, "posted_at": "",
        "description": "",
    }]).to_csv(data_dir / "postings" / "postings.csv", index=False)
    pd.DataFrame([{
        "posting_id": "p1", "run_id": "r", "extracted_at": "2026-05-30T00:00:00Z",
        "role_family": "ai", "seniority": "senior", "domain_tags": "[]",
        "required_skills": '[{"name": "Python", "required": true, "evidence": ""}]',
        "preferred_skills": "[]", "model": "x", "prompt_tokens": 1,
        "completion_tokens": 1, "latency_ms": 1.0,
    }]).to_csv(data_dir / "extractions" / "extractions.csv", index=False)

    # window=30, baseline=90 → 60-day-old posting belongs in BASELINE, not current
    report = t_mod.compute_trend(window_days=30, baseline_days=90, top_n=5)
    assert report.baseline_total_postings == 1
    assert report.window_total_postings == 0


def test_compute_trend_does_not_crash_on_naive_timestamps(tmp_path, monkeypatch):
    """End-to-end: a posting whose posted_at is naive should not break trend
    comparison against the UTC-aware window boundary."""
    import importlib
    import pandas as pd
    data_dir = tmp_path / "data"
    for sub in ("postings", "extractions", "metrics", "reports"):
        (data_dir / sub).mkdir(parents=True)
    monkeypatch.setenv("SKILLTREND_DATA_DIR", str(data_dir))
    from skilltrend import settings as s_mod
    importlib.reload(s_mod)
    from skilltrend import storage as st_mod
    importlib.reload(st_mod)
    from skilltrend import trends as t_mod
    importlib.reload(t_mod)

    pd.DataFrame([{
        "posting_id": "p1", "url": "u1", "company": "Microsoft",
        "title": "SRE", "location": "", "source": "microsoft",
        "first_seen": "2026-05-29T19:34:24", "last_seen": "2026-05-29T19:34:24",
        "active": True, "posted_at": "2026-05-29T19:34:24",  # naive
        "description": "",
    }]).to_csv(data_dir / "postings" / "postings.csv", index=False)
    pd.DataFrame([{
        "posting_id": "p1", "run_id": "r", "extracted_at": "2026-05-29T19:34:24",
        "role_family": "infra", "seniority": "senior", "domain_tags": "[]",
        "required_skills": '[{"name": "Kubernetes", "required": true, "evidence": ""}]',
        "preferred_skills": "[]", "model": "x", "prompt_tokens": 1,
        "completion_tokens": 1, "latency_ms": 1.0,
    }]).to_csv(data_dir / "extractions" / "extractions.csv", index=False)

    report = t_mod.compute_trend(window_days=30, baseline_days=90, top_n=5)
    assert report.window_total_postings == 1

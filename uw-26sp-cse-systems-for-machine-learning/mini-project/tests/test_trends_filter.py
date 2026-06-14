"""Tests for the per-company trend filter."""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def temp_data(tmp_path: Path, monkeypatch):
    data = tmp_path / "data"
    for sub in ("postings", "extractions", "metrics", "reports"):
        (data / sub).mkdir(parents=True)
    monkeypatch.setenv("SKILLTREND_DATA_DIR", str(data))
    from skilltrend import settings as s_mod
    importlib.reload(s_mod)
    from skilltrend import storage as st_mod
    importlib.reload(st_mod)
    from skilltrend import trends as t_mod
    importlib.reload(t_mod)

    rows = []
    for i, (company, skill) in enumerate([
        ("Apple", "Swift"), ("Apple", "Objective-C"),
        ("NVIDIA", "CUDA"), ("NVIDIA", "Triton"),
        ("Anthropic", "Python"), ("Anthropic", "PyTorch"),
    ]):
        rows.append({
            "posting_id": f"p{i}", "url": f"u{i}", "company": company,
            "title": f"role-{i}", "location": "", "source": "test",
            "first_seen": "2026-05-29T19:00:00Z",
            "last_seen": "2026-05-29T19:00:00Z",
            "active": True, "posted_at": "2026-05-29T19:00:00Z",
            "description": "",
        })
    pd.DataFrame(rows).to_csv(data / "postings" / "postings.csv", index=False)

    ext_rows = []
    for i, (_, skill) in enumerate([
        ("Apple", "Swift"), ("Apple", "Objective-C"),
        ("NVIDIA", "CUDA"), ("NVIDIA", "Triton"),
        ("Anthropic", "Python"), ("Anthropic", "PyTorch"),
    ]):
        ext_rows.append({
            "posting_id": f"p{i}", "run_id": "r",
            "extracted_at": "2026-05-29T19:00:00Z",
            "role_family": "ai", "seniority": "senior", "domain_tags": "[]",
            "required_skills": f'[{{"name": "{skill}", "required": true, "evidence": ""}}]',
            "preferred_skills": "[]", "model": "x", "prompt_tokens": 1,
            "completion_tokens": 1, "latency_ms": 1.0,
        })
    pd.DataFrame(ext_rows).to_csv(data / "extractions" / "extractions.csv", index=False)
    yield data


def test_no_filter_sees_all_skills(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=10)
    assert r.window_total_postings == 6
    skills = {t.skill for t in r.rising}
    assert "Swift" in skills and "CUDA" in skills and "Python" in skills


def test_filter_to_one_company_excludes_others(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=10,
                      companies=["NVIDIA"])
    assert r.window_total_postings == 2
    skills = {t.skill for t in r.rising}
    assert "CUDA" in skills and "Triton" in skills
    assert "Swift" not in skills and "Python" not in skills


def test_filter_to_multiple_companies(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=10,
                      companies=["Apple", "Anthropic"])
    assert r.window_total_postings == 4
    skills = {t.skill for t in r.rising}
    assert "Swift" in skills and "Python" in skills
    assert "CUDA" not in skills


def test_filter_to_unknown_company_returns_empty(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=10,
                      companies=["DoesNotExist"])
    assert r.window_total_postings == 0
    assert r.rising == [] and r.declining == []

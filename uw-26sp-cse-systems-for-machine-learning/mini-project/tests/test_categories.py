"""Tests for the skill -> category lookup + trend category filter."""
from __future__ import annotations

import importlib

import pandas as pd
import pytest

from skilltrend.categories import (UNCATEGORIZED, all_categories,
                                   category_for)


@pytest.mark.parametrize("skill, expected", [
    ("Python", "Languages"),
    ("python", "Languages"),  # case-insensitive
    ("  PYTHON  ", "Languages"),  # whitespace insensitive
    ("PyTorch", "AI/ML Frameworks"),
    ("LangChain", "LLM / Agentic Stack"),
    ("Kubernetes", "Cloud & Infrastructure"),
    ("Snowflake", "Data & Analytics"),
    ("React", "Web & Frontend"),
    ("PostgreSQL", "Databases"),
    ("Salesforce", "Tools & Business Apps"),
])
def test_known_skill_classified(skill, expected):
    assert category_for(skill) == expected


def test_unknown_skill_is_uncategorized():
    assert category_for("RandomNewToolXYZ") == UNCATEGORIZED


def test_empty_skill_is_uncategorized():
    assert category_for("") == UNCATEGORIZED
    assert category_for(None) == UNCATEGORIZED  # type: ignore[arg-type]


def test_all_categories_includes_uncategorized():
    cats = all_categories()
    assert UNCATEGORIZED in cats
    assert "Languages" in cats
    assert "AI/ML Frameworks" in cats


# ----------------------------------------------------------------------- trend


@pytest.fixture
def temp_data(tmp_path, monkeypatch):
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

    pd.DataFrame([
        {"posting_id": "p1", "url": "u1", "company": "Acme", "title": "t",
         "location": "", "source": "test",
         "first_seen": "2026-05-29T00:00:00Z", "last_seen": "2026-05-29T00:00:00Z",
         "active": True, "posted_at": "2026-05-29T00:00:00Z", "description": ""},
        {"posting_id": "p2", "url": "u2", "company": "Acme", "title": "t",
         "location": "", "source": "test",
         "first_seen": "2026-05-29T00:00:00Z", "last_seen": "2026-05-29T00:00:00Z",
         "active": True, "posted_at": "2026-05-29T00:00:00Z", "description": ""},
    ]).to_csv(data / "postings" / "postings.csv", index=False)

    pd.DataFrame([
        {"posting_id": "p1", "run_id": "r", "extracted_at": "2026-05-29T00:00:00Z",
         "role_family": "ai", "seniority": "senior", "domain_tags": "[]",
         "required_skills":
            '[{"name":"Python","required":true,"evidence":""},'
            '{"name":"PyTorch","required":true,"evidence":""},'
            '{"name":"Kubernetes","required":true,"evidence":""}]',
         "preferred_skills": "[]", "model": "x", "prompt_tokens": 1,
         "completion_tokens": 1, "latency_ms": 1.0},
        {"posting_id": "p2", "run_id": "r", "extracted_at": "2026-05-29T00:00:00Z",
         "role_family": "ai", "seniority": "senior", "domain_tags": "[]",
         "required_skills":
            '[{"name":"React","required":true,"evidence":""},'
            '{"name":"PostgreSQL","required":true,"evidence":""}]',
         "preferred_skills": "[]", "model": "x", "prompt_tokens": 1,
         "completion_tokens": 1, "latency_ms": 1.0},
    ]).to_csv(data / "extractions" / "extractions.csv", index=False)
    yield data


def test_trend_includes_category_field(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=20)
    by_skill = {t.skill: t.category for t in r.rising}
    assert by_skill["Python"] == "Languages"
    assert by_skill["PyTorch"] == "AI/ML Frameworks"
    assert by_skill["Kubernetes"] == "Cloud & Infrastructure"


def test_trend_filter_to_single_category(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=20,
                      categories=["Languages"])
    skills = {t.skill for t in r.rising}
    assert "Python" in skills
    assert "PyTorch" not in skills  # AI/ML Frameworks excluded
    assert "Kubernetes" not in skills  # Cloud excluded


def test_trend_filter_to_multiple_categories(temp_data):
    from skilltrend.trends import compute_trend
    r = compute_trend(window_days=30, baseline_days=90, top_n=20,
                      categories=["Languages", "Databases"])
    skills = {t.skill for t in r.rising}
    assert "Python" in skills
    assert "PostgreSQL" in skills
    assert "PyTorch" not in skills

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_id(*parts: str) -> str:
    """Deterministic ID derived from inputs — same posting URL across scans
    always hashes to the same id, so we can detect repeats without dedup'ing
    away history."""
    raw = "|".join(p.strip().lower() for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Posting(BaseModel):
    """A raw job posting snapshot. Multiple rows for the same posting_id are
    allowed: each scan appends a new snapshot row so we can observe how a
    posting's visibility changes over time."""

    posting_id: str
    url: str
    company: str
    title: str
    location: str = ""
    source: str  # provider name
    first_seen: str  # ISO timestamp of earliest snapshot
    last_seen: str  # ISO timestamp of this snapshot
    active: bool = True
    posted_at: str = ""  # provider-reported posting date if exposed
    description: str = ""


class ExtractedSkill(BaseModel):
    name: str
    category: str = ""  # canonical taxonomy category if available
    required: bool = True
    evidence: str = ""


class Extraction(BaseModel):
    """LLM extraction output for one posting."""

    posting_id: str
    run_id: str
    extracted_at: str
    role_family: str = ""
    seniority: str = ""
    domain_tags: list[str] = Field(default_factory=list)
    required_skills: list[ExtractedSkill] = Field(default_factory=list)
    preferred_skills: list[ExtractedSkill] = Field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class RunMetric(BaseModel):
    """A single LLM extraction call's measurement, written as one JSONL row."""

    run_id: str
    posting_id: str
    mode: str  # "sequential" | "concurrent"
    workers: int
    model: str
    start_ts: float
    end_ts: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ok: bool
    error: str = ""


class RunSummary(BaseModel):
    """Aggregated metrics for a full extraction run."""

    run_id: str
    mode: str
    workers: int
    model: str
    started_at: str
    finished_at: str
    total_postings: int
    successful: int
    failed: int
    total_latency_s: float
    wall_clock_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    throughput_postings_per_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float

"""LLM-driven extraction worker.

One Posting -> one Extraction. The system prompt encodes the taxonomy
guidance once; the per-call prompt is the job description plus a short
header. This split matters for the systems analysis: stable taxonomy text
becomes a cache-friendly prefix on backends that support prompt caching."""
from __future__ import annotations

import json
import logging
from typing import Iterable

from ..llm import chat_json
from ..models import ExtractedSkill, Extraction, Posting, utcnow_iso

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You extract structured skill data from job postings. The
output is consumed by a tool that tells a job seeker WHICH SPECIFIC
technologies, frameworks, or tools they should learn — so generic umbrella
terms are useless. Be ruthlessly specific.

Return a JSON object with these keys:

- role_family: one of {ai, ml, mlops, backend, frontend, fullstack, data,
  infra, security, research, other}.
- seniority: one of {intern, junior, mid, senior, staff, principal, manager,
  unknown}.
- domain_tags: array of short tags, e.g. ["ai", "backend"].
- required_skills: array of {name, required: true, evidence}. "evidence" is a
  short verbatim quote (<=120 chars) from the posting.
- preferred_skills: same shape, for nice-to-have items.

EXTRACTION RULES — apply strictly:

1. ALWAYS pick the most SPECIFIC name. If the posting names a concrete
   technology alongside a generic umbrella, extract ONLY the specific one.
     "SQL (PostgreSQL preferred)"          -> "PostgreSQL"     (not "SQL")
     "LLM frameworks like LangChain"       -> "LangChain"      (not "LLM")
     "deep learning frameworks (PyTorch)"  -> "PyTorch"
     "cloud platforms (AWS or GCP)"        -> "AWS", "GCP"
     "containers (Docker, Kubernetes)"     -> "Docker", "Kubernetes"

2. BLOCKLIST. Never include these as standalone skills. Only allow them when
   the posting truly mentions no specific technology in the same context:
     AI, ML, LLM, GenAI, NLP, Computer Vision, Deep Learning, Machine Learning,
     Artificial Intelligence, MLOps, DevOps, SQL, NoSQL, Database, Cloud,
     SaaS, IaaS, PaaS, Backend, Frontend, Mobile, Web, API, Microservices,
     Big Data, Data, Software, Engineering, Distributed Systems,
     Containerization, Serverless, Observability, CI/CD, Automation,
     Infrastructure.

3. NEVER extract WORK ACTIVITIES — these describe what you DO, not what you
   KNOW. Skip:
     "production rollouts", "evaluation", "optimization", "testing",
     "deployment", "monitoring", "code review", "design", "architecture",
     "collaboration", "communication", "leadership", "mentoring".

4. PREFER concrete names of:
   - languages (Python, Go, TypeScript, Rust, ...)
   - frameworks/libraries (PyTorch, React, FastAPI, LangChain, vLLM, ...)
   - cloud services (AWS Lambda, BigQuery, Cloud Run, S3, ...)
   - databases (PostgreSQL, Redis, Snowflake, ...)
   - protocols/standards (gRPC, GraphQL, MCP, OAuth, ...)
   - methodologies (RAG, RLHF, fine-tuning, distributed training, ...)
   - platforms (Kubernetes, Airflow, Databricks, Spark, ...)

5. SUPPORT EVERY SKILL WITH EVIDENCE. If a skill is only implied or merely
   "would be nice for someone at this company", DO NOT include it. The
   "evidence" field must come directly from the posting.

Return strictly valid JSON. No commentary, no markdown."""


def _build_user_prompt(posting: Posting) -> str:
    desc = posting.description or ""
    if len(desc) > 8000:
        desc = desc[:8000] + "\n[...truncated...]"
    return (
        f"Company: {posting.company}\n"
        f"Title: {posting.title}\n"
        f"Location: {posting.location}\n\n"
        f"--- Job Description ---\n{desc}\n"
    )


def _coerce_skills(items: Iterable[dict], required: bool) -> list[ExtractedSkill]:
    out: list[ExtractedSkill] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        if not name:
            continue
        out.append(ExtractedSkill(
            name=name,
            category=str(it.get("category", "")),
            required=bool(it.get("required", required)),
            evidence=str(it.get("evidence", ""))[:240],
        ))
    return out


async def extract_one(posting: Posting, *, run_id: str) -> Extraction:
    """Run the extraction LLM call for a single posting. Returns an Extraction
    with measured latency/token counts. Errors raise — the pipeline catches
    them and records a failed metric."""
    user = _build_user_prompt(posting)
    resp = await chat_json(SYSTEM_PROMPT, user)
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError:
        log.warning("non-JSON response for posting %s; recording empty result", posting.posting_id)
        data = {}

    required_skills = _coerce_skills(data.get("required_skills", []), required=True)
    preferred_skills = _coerce_skills(data.get("preferred_skills", []), required=False)

    return Extraction(
        posting_id=posting.posting_id,
        run_id=run_id,
        extracted_at=utcnow_iso(),
        role_family=str(data.get("role_family", "")),
        seniority=str(data.get("seniority", "")),
        domain_tags=[str(t) for t in data.get("domain_tags", []) if t],
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        model=resp.model,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
        latency_ms=resp.latency_ms,
    )

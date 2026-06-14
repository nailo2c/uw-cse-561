"""Deterministic post-extraction filter.

Small models like gemini-2.5-flash-lite ignore "do not extract X" instructions
roughly 5-15% of the time. Rather than rely on prompt obedience, we
filter generic umbrella terms here so the trend output is actionable.

Edit this list freely — it's the policy layer between raw LLM output and
the trend reports.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Exact-match blocklist (case-insensitive, whitespace-collapsed).
# Keep entries to genuine umbrella terms — don't put real tools here.
_EXACT = {
    # ── tech umbrella categories
    "ai", "ml", "llm", "nlp", "genai", "computer vision",
    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "data analytics", "data engineering",
    "analytics", "statistics",
    # ── database / storage umbrella
    "sql", "nosql", "database", "databases", "rdbms", "data warehouse",
    "data lake", "data lakehouse", "lakehouse",
    # ── infra umbrella
    "cloud", "cloud computing", "saas", "iaas", "paas",
    "api", "apis", "rest", "rest api", "rest apis", "restful api",
    "microservices", "serverless", "containerization", "containers",
    "observability", "monitoring", "automation", "ci/cd", "cicd",
    "infrastructure", "iac", "infrastructure as code",
    "distributed systems", "big data", "streaming",
    # ── generic engineering categories
    "backend", "frontend", "fullstack", "full-stack", "full stack",
    "mobile", "web", "web development", "software",
    "software engineering", "engineering", "devops", "mlops", "dataops",
    "platform engineering", "systems engineering",
    # ── languages spoken by humans
    "english", "french", "german", "spanish", "japanese",
    "chinese", "mandarin", "korean", "italian", "portuguese",
    "language skills", "fluency",
    # ── soft / workplace skills
    "communication", "communication skills", "leadership", "collaboration",
    "negotiation", "mentoring", "mentorship", "coaching",
    "project management", "program management", "change management",
    "stakeholder management", "people management", "team leadership",
    "presentation", "presentation skills", "writing", "documentation",
    "problem solving", "problem-solving", "critical thinking",
    "time management", "organization", "attention to detail",
    # ── work activities (verbs, not skills)
    "evaluation", "optimization", "deployment", "code review",
    "testing", "design", "architecture", "production rollouts",
    "prototyping", "research", "experimentation",
    # ── business / commerce umbrellas
    "sales", "marketing", "growth", "operations", "strategy",
    "b2b sales", "b2c sales", "saas sales", "b2b saas sales",
    "enterprise sales", "inside sales", "sales operations",
    "customer success", "customer support", "account management",
    # ── ML/AI variant umbrellas the prompt blocklist doesn't catch directly
    "ai tools", "ai systems", "ai platforms", "ai workflows",
    "ai models", "ai engineering", "agentic ai", "ai agents",
    "llms", "generative ai", "foundation models",
    "language models", "large language models",
    # ── data variant umbrellas
    "data tools", "data platforms", "data pipelines",
    "etl", "elt", "data integration",
}

# Pattern-match blocklist for "X tools/systems/platforms" style noise.
_PATTERNS = [
    re.compile(r"^(ai|ml|llm|data|cloud|web|api)\s+(tool|system|platform|workflow|model|service|solution)s?$"),
]


@lru_cache(maxsize=4096)
def is_blocked(name: str) -> bool:
    if not name:
        return True
    key = " ".join(name.split()).strip().lower()
    if not key:
        return True
    if key in _EXACT:
        return True
    return any(p.match(key) for p in _PATTERNS)

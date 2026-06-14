"""Unit tests for the deterministic blocklist filter."""
from __future__ import annotations

import pytest

from skilltrend.blocklist import is_blocked


@pytest.mark.parametrize("name", [
    "AI", "ai", "ML", "LLM", "SQL", "NoSQL",
    "Cloud", "SaaS", "API", "Microservices",
    "Backend", "Frontend", "DevOps",
    "English", "French",
    "communication", "negotiation", "project management",
    "B2B sales", "SaaS sales",
    "ai tools", "AI Tools", "AI workflows", "data platforms",
    "Agentic AI", "Generative AI", "data analysis",
])
def test_blocked(name: str):
    assert is_blocked(name), f"expected to block: {name!r}"


@pytest.mark.parametrize("name", [
    "Python", "PyTorch", "TensorFlow", "LangChain", "vLLM",
    "Kubernetes", "Docker", "PostgreSQL", "Redis", "Snowflake",
    "BigQuery", "AWS Lambda", "Cloud Run",
    "RAG", "Fine-tuning", "RLHF", "Model Context Protocol",
    "Agentic workflows",  # canonical, distinct from "Agentic AI"
    "TypeScript", "React", "FastAPI",
])
def test_allowed(name: str):
    assert not is_blocked(name), f"expected to allow: {name!r}"


def test_whitespace_and_case_insensitive():
    assert is_blocked("  ai  ")
    assert is_blocked("Project   Management")
    assert is_blocked("AI tools")


def test_empty_is_blocked():
    assert is_blocked("")
    assert is_blocked("   ")

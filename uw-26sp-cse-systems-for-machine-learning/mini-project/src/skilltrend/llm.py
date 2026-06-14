"""OpenAI-compatible async LLM client.

The whole pipeline talks to LLMs only through this module so we can swap
between OpenAI, vLLM, Ollama, or a deterministic fake without touching the
agent code.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from .ratelimit import TokenBucketLimiter
from .settings import settings

log = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    rate_limit_wait_ms: float = 0.0
    raw: dict | None = None


_client: AsyncOpenAI | None = None
_limiter: TokenBucketLimiter | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


def get_limiter() -> TokenBucketLimiter:
    global _limiter
    if _limiter is None:
        _limiter = TokenBucketLimiter(rpm=settings.rpm)
    return _limiter


async def chat_json(system: str, user: str, *, model: str | None = None,
                    temperature: float = 0.0) -> LLMResponse:
    """Send a chat message expecting JSON output back.

    Uses `response_format={"type": "json_object"}` which is supported by
    OpenAI, vLLM (guided_json), and recent versions of Ollama. Backends
    without JSON-mode support will simply receive a strongly worded
    instruction in the system prompt instead — we still parse defensively.
    """
    model = model or settings.model

    if settings.fake_llm:
        return _fake_response(user, model)

    # Throttle BEFORE the API call so 429s don't happen client-side.
    # Wait time is reported separately from the inference latency so the
    # paper's latency tables don't get contaminated by client-side queueing.
    wait_s = await get_limiter().acquire()

    client = get_client()
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
    except Exception:
        # JSON mode unsupported on this backend — retry without it.
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system + "\nReturn JSON only."},
                      {"role": "user", "content": user}],
            temperature=temperature,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = resp.choices[0].message.content or "{}"
    usage = resp.usage
    return LLMResponse(
        text=text,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        latency_ms=latency_ms,
        model=model,
        rate_limit_wait_ms=wait_s * 1000.0,
    )


# ---------------------------------------------------------------------------
# Deterministic offline mode

_FAKE_SKILLS_BY_KEYWORD = {
    "python": "Python", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "kubernetes": "Kubernetes", "k8s": "Kubernetes", "docker": "Docker",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure", "react": "React",
    "typescript": "TypeScript", "go ": "Go", "golang": "Go", "rust": "Rust",
    "postgres": "PostgreSQL", "redis": "Redis", "kafka": "Kafka",
    "spark": "Spark", "airflow": "Airflow", "dbt": "dbt",
    "snowflake": "Snowflake", "bigquery": "BigQuery", "vllm": "vLLM",
    "rag": "RAG", "langchain": "LangChain", "agent": "Agentic workflows",
    "mcp": "Model Context Protocol", "fine-tun": "Fine-tuning",
    "inference": "Inference optimization",
}


def _fake_response(user: str, model: str) -> LLMResponse:
    low = user.lower()
    required: list[str] = []
    for needle, canonical in _FAKE_SKILLS_BY_KEYWORD.items():
        if needle in low and canonical not in required:
            required.append(canonical)
    payload = {
        "role_family": "software" if "engineer" in low else "unknown",
        "seniority": "senior" if "senior" in low else "mid",
        "domain_tags": ["ai"] if any(k in low for k in ("llm", "ml", "ai")) else ["backend"],
        "required_skills": [{"name": s, "required": True, "evidence": ""} for s in required],
        "preferred_skills": [],
    }
    text = json.dumps(payload)
    # Mimic non-zero token usage so metrics aren't degenerate.
    return LLMResponse(
        text=text,
        prompt_tokens=max(50, len(user.split())),
        completion_tokens=max(20, len(text.split())),
        latency_ms=15.0,
        model=f"{model}-fake",
    )

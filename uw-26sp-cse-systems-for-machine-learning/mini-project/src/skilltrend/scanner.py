"""Multi-provider scanner.

Iterates over companies declared in config/companies.yaml, fetches postings
from each ATS provider concurrently (one task per company), and persists the
results through the storage layer. Failures on one company never abort the
entire scan — they are logged and the scan keeps going.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import httpx

from .models import Posting
from .providers.amazon import AmazonProvider
from .providers.apple import AppleProvider
from .providers.ashby import AshbyProvider
from .providers.base import CompanyRef, Provider
from .providers.google import GoogleProvider
from .providers.greenhouse import GreenhouseProvider
from .providers.jobspy_provider import JobSpyProvider
from .providers.lever import LeverProvider
from .providers.meta import MetaProvider
from .providers.microsoft import MicrosoftProvider
from .providers.workday import WorkdayProvider
from .settings import settings
from .storage import upsert_postings

log = logging.getLogger(__name__)


PROVIDER_REGISTRY: dict[str, type[Provider]] = {
    "greenhouse": GreenhouseProvider,
    "lever": LeverProvider,
    "ashby": AshbyProvider,
    "workday": WorkdayProvider,
    "amazon": AmazonProvider,
    "apple": AppleProvider,
    "google": GoogleProvider,
    "microsoft": MicrosoftProvider,
    "meta": MetaProvider,
    "jobspy": JobSpyProvider,
}


@dataclass
class ScanResult:
    total_fetched: int
    added: int
    refreshed: int
    failures: list[tuple[str, str, str]]  # (provider, slug, error)


# (done, total, provider_name, slug, fetched_count, error_str_or_none)
ScanProgressCb = Callable[[int, int, str, str, int, str | None], None]


async def _scan_company(provider: Provider, company: CompanyRef, limit: int) -> list[Posting]:
    try:
        return await provider.fetch(company, limit=limit)
    except Exception as exc:  # noqa: BLE001 — surface but never abort scan
        log.warning("provider=%s slug=%s failed: %s", provider.name, company.slug, exc)
        raise


async def _scan_with_meta(provider_name: str, cref: CompanyRef,
                          provider: Provider, limit: int):
    """Wrap _scan_company so as_completed yields back the (provider, cref)
    context — we need that for progress reporting without a separate lookup."""
    try:
        result = await _scan_company(provider, cref, limit)
        return provider_name, cref, result, None
    except Exception as exc:  # noqa: BLE001
        return provider_name, cref, None, exc


def count_companies(cfg: dict[str, list]) -> int:
    return sum(len(c or []) for k, c in cfg.items() if k in PROVIDER_REGISTRY)


async def scan_all(limit_per_company: int | None = None,
                   on_progress: ScanProgressCb | None = None) -> ScanResult:
    cfg = settings.load_companies()
    limit = limit_per_company or settings.max_postings_per_company
    failures: list[tuple[str, str, str]] = []
    fetched: list[Posting] = []

    async with httpx.AsyncClient(headers={"User-Agent": "skilltrend/0.1"}) as client:
        tasks: list[asyncio.Task] = []
        for provider_name, companies in cfg.items():
            cls = PROVIDER_REGISTRY.get(provider_name)
            if cls is None:
                log.warning("unknown provider in config: %s", provider_name)
                continue
            provider = cls(client)
            for entry in companies or []:
                extra = {k: v for k, v in entry.items() if k not in ("slug", "name")}
                cref = CompanyRef(
                    slug=entry.get("slug", entry.get("name", "")),
                    name=entry.get("name", entry.get("slug", "")),
                    extra=extra,
                )
                tasks.append(asyncio.create_task(
                    _scan_with_meta(provider_name, cref, provider, limit)
                ))

        total = len(tasks)
        done = 0
        for fut in asyncio.as_completed(tasks):
            provider_name, cref, result, exc = await fut
            done += 1
            if exc is not None:
                failures.append((provider_name, cref.slug, str(exc)))
                if on_progress is not None:
                    on_progress(done, total, provider_name, cref.slug, 0, str(exc))
            else:
                fetched.extend(result)
                if on_progress is not None:
                    on_progress(done, total, provider_name, cref.slug, len(result), None)

    added, refreshed = upsert_postings(fetched)
    return ScanResult(total_fetched=len(fetched), added=added, refreshed=refreshed,
                      failures=failures)

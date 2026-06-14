from __future__ import annotations

import abc
from dataclasses import dataclass, field

import httpx

from ..models import Posting


@dataclass(slots=True)
class CompanyRef:
    slug: str
    name: str
    # Per-provider overrides from companies.yaml. Used by providers that need
    # more than just a slug (Workday: host+site; JobSpy: linkedin_company_id).
    extra: dict = field(default_factory=dict)


class Provider(abc.ABC):
    """Common interface for ATS-specific scrapers.

    A provider takes a company slug (the identifier in the ATS URL) and returns
    a list of Posting objects with the description text already inlined when
    the ATS exposes it. Providers should not deduplicate — that is the job of
    the storage layer."""

    name: str = "base"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    @abc.abstractmethod
    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        ...

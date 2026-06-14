"""JobSpy provider — aggregator fallback for companies not on a public ATS.

Reaches FAANG-class companies (Apple, Google, Meta, Tesla) that don't expose
a public JSON API by scraping LinkedIn / Indeed via the python-jobspy
library. Output is normalized into the same Posting model as our ATS
providers so the downstream pipeline doesn't change.

Trade-offs vs the ATS providers:
- LinkedIn rate-limits aggressively after ~10 pages from one IP. For larger
  scrapes, set up proxies via env (JobSpy reads them from the `proxies` kwarg
  -- not wired here; this provider is intentionally low-volume).
- Job descriptions on LinkedIn are not always full. The provider passes
  linkedin_fetch_description=True to maximize completeness, at the cost of
  one extra request per posting.
- This is gray area under LinkedIn ToS. Acceptable for academic/research
  use, NOT for commercial deployment.

companies.yaml entries:

    jobspy:
      - slug: apple-linkedin
        name: Apple
        sites: [linkedin]
        search_term: software engineer
        linkedin_company_ids: [162479]
        location: United States
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider

log = logging.getLogger(__name__)

try:
    from jobspy import scrape_jobs as _scrape_jobs
    JOBSPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    JOBSPY_AVAILABLE = False


def _to_iso(value) -> str:
    if value is None:
        return ""
    try:
        return datetime.fromisoformat(str(value)).date().isoformat()
    except (TypeError, ValueError):
        return ""


class JobSpyProvider(Provider):
    name = "jobspy"

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        if not JOBSPY_AVAILABLE:
            raise RuntimeError(
                "python-jobspy not installed. Run: pip install python-jobspy"
            )

        sites = company.extra.get("sites") or ["linkedin"]
        search_term = company.extra.get("search_term", company.name)
        location = company.extra.get("location", "United States")
        linkedin_ids = company.extra.get("linkedin_company_ids")

        kwargs = dict(
            site_name=sites,
            search_term=search_term,
            location=location,
            results_wanted=limit,
            linkedin_fetch_description=True,  # full description for skill extraction
        )
        if linkedin_ids:
            kwargs["linkedin_company_ids"] = linkedin_ids

        # JobSpy is synchronous — offload to a thread so we don't block the
        # event loop while LinkedIn / Indeed scraping happens.
        df = await asyncio.to_thread(_scrape_jobs, **kwargs)
        if df is None or len(df) == 0:
            return []

        now = utcnow_iso()
        out: list[Posting] = []
        for _, row in df.iterrows():
            url = str(row.get("job_url") or "")
            title = str(row.get("title") or "")
            posting_id = stable_id(self.name, str(row.get("id", "")), url)
            description = str(row.get("description") or "")
            posted = _to_iso(row.get("date_posted"))
            out.append(Posting(
                posting_id=posting_id,
                url=url,
                company=company.name,
                title=title,
                location=str(row.get("location") or ""),
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at=posted,
                description=description,
            ))
        return out

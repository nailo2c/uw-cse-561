"""Apple Jobs provider.

Apple's public careers site exposes a JSON search API, but it requires the
short-lived CSRF token and cookies issued by `/api/v1/CSRFToken` first.
The search response includes job summaries inline; those summaries are not as
rich as ATS detail pages, but they are stable and do not require browser
automation.
"""
from __future__ import annotations

import asyncio
import html
import re

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider
from .filters import allowed_by_terms, config_terms

BASE = "https://jobs.apple.com"
CSRF_URL = f"{BASE}/api/v1/CSRFToken"
SEARCH_URL = f"{BASE}/api/v1/search"
PAGE_SIZE = 20
REQUEST_DELAY_S = 0.3

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": BASE,
    "Referer": f"{BASE}/en-us/search",
    "browserlocale": "en-us",
    "locale": "EN_US",
}


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _location_text(job: dict) -> str:
    locations = job.get("locations") or []
    parts = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        pieces = [
            loc.get("city"),
            loc.get("stateProvince"),
            loc.get("countryName") or loc.get("country"),
        ]
        value = ", ".join(str(p) for p in pieces if p)
        if value:
            parts.append(value)
    return "; ".join(parts)


class AppleProvider(Provider):
    name = "apple"

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        csrf = await self.client.get(CSRF_URL, headers=HEADERS, timeout=30.0)
        csrf.raise_for_status()
        csrf_token = csrf.headers.get("x-apple-csrf-token")
        if not csrf_token:
            raise RuntimeError("apple: CSRF token missing")

        headers = {**HEADERS, "x-apple-csrf-token": csrf_token}
        terms = company.extra.get("search_terms") or [company.extra.get("search_term", "")]
        if isinstance(terms, str):
            terms = [terms]
        title_include = config_terms(company.extra, "title_include")
        title_exclude = config_terms(company.extra, "title_exclude")
        location_include = config_terms(company.extra, "location_include")
        location_exclude = config_terms(company.extra, "location_exclude")
        locale = company.extra.get("locale", "en-us")
        sort = company.extra.get("sort", "newest")
        filters = company.extra.get("filters") or {}
        max_pages = int(company.extra.get("max_pages", 5))

        now = utcnow_iso()
        out: list[Posting] = []
        seen_ids: set[str] = set()

        for term in terms:
            page = 1
            total_records = None
            while len(out) < limit and page <= max_pages:
                payload = {
                    "query": term,
                    "filters": filters,
                    "page": page,
                    "locale": locale,
                    "sort": sort,
                    "format": {"longDate": "MMMM D, YYYY", "mediumDate": "MMM D, YYYY"},
                }
                r = await self.client.post(
                    SEARCH_URL, json=payload, headers=headers, timeout=30.0
                )
                r.raise_for_status()
                envelope = r.json()
                data = envelope.get("res", envelope)
                jobs = data.get("searchResults", []) or []
                total_records = data.get("totalRecords", total_records)
                if not jobs:
                    break

                for job in jobs:
                    if len(out) >= limit:
                        break
                    position_id = str(
                        job.get("positionId") or job.get("id") or job.get("reqId") or ""
                    )
                    if not position_id or position_id in seen_ids:
                        continue
                    summary = _strip_html(job.get("jobSummary", ""))
                    team = (
                        (job.get("team") or {}).get("teamName")
                        if isinstance(job.get("team"), dict)
                        else ""
                    )
                    title = job.get("postingTitle", "")
                    location = _location_text(job)
                    haystack = f"{title} {team}".lower()
                    if not allowed_by_terms(haystack, title_include, title_exclude):
                        continue
                    if not allowed_by_terms(location, location_include, location_exclude):
                        continue
                    seen_ids.add(position_id)
                    title_slug = job.get("transformedPostingTitle") or ""
                    url = (
                        f"{BASE}/en-us/details/{position_id}/{title_slug}"
                        if position_id and title_slug
                        else f"{BASE}/en-us/details/{position_id}"
                    )
                    description = "\n\n".join(
                        p for p in [summary, f"Team: {team}" if team else ""] if p
                    )
                    out.append(Posting(
                        posting_id=stable_id(self.name, position_id, url),
                        url=url,
                        company=company.name,
                        title=title,
                        location=location,
                        source=self.name,
                        first_seen=now,
                        last_seen=now,
                        active=True,
                        posted_at=job.get("postDateInGMT", "") or job.get("postingDate", ""),
                        description=description,
                    ))

                if total_records is not None and page * PAGE_SIZE >= int(total_records):
                    break
                page += 1
                if len(out) < limit:
                    await asyncio.sleep(REQUEST_DELAY_S)

        return out

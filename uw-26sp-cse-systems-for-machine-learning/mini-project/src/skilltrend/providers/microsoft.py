"""Microsoft Careers provider.

Microsoft's current public careers flow uses an Eightfold-backed JSON search
endpoint. Search rows do not include the full job description, so this provider
fetches the public detail page and reads the JobPosting JSON-LD when available.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider
from .filters import allowed_by_terms, config_terms

log = logging.getLogger(__name__)

BASE = "https://apply.careers.microsoft.com"
SEARCH_URL = f"{BASE}/api/pcsx/search"
PAGE_SIZE = 10
DETAIL_CONCURRENCY = 5

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "skilltrend/0.1",
}


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _posted_ts(value) -> str:
    try:
        if value:
            return datetime.fromtimestamp(int(value), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
    return ""


def _location_text(position: dict) -> str:
    locations = position.get("locations") or []
    if isinstance(locations, list):
        return "; ".join(str(loc) for loc in locations if loc)
    return str(locations or "")


def _extract_jsonld(html_text: str) -> dict:
    soup = BeautifulSoup(html_text, "html.parser")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return item
    return {}


class MicrosoftProvider(Provider):
    name = "microsoft"
    detail_concurrency = DETAIL_CONCURRENCY

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        query = company.extra.get("search_term", "")
        location = company.extra.get("location", "")
        title_include = config_terms(company.extra, "title_include")
        title_exclude = config_terms(company.extra, "title_exclude")
        location_include = config_terms(company.extra, "location_include")
        location_exclude = config_terms(company.extra, "location_exclude")
        max_pages = int(company.extra.get("max_pages", 10))
        start = 0
        pages = 0
        positions: list[dict] = []

        while len(positions) < limit and pages < max_pages:
            params = {
                "domain": company.extra.get("domain", "microsoft.com"),
                "query": query,
                "location": location,
                "start": start,
                "sort_by": company.extra.get("sort_by", "timestamp"),
            }
            r = await self.client.get(
                SEARCH_URL, params=params, headers=HEADERS, timeout=30.0
            )
            r.raise_for_status()
            chunk = (((r.json() or {}).get("data") or {}).get("positions") or [])
            if not chunk:
                break
            for position in chunk:
                if not allowed_by_terms(
                    str(position.get("name", "")), title_include, title_exclude
                ):
                    continue
                if not allowed_by_terms(
                    _location_text(position), location_include, location_exclude
                ):
                    continue
                positions.append(position)
                if len(positions) >= limit:
                    break
            start += PAGE_SIZE
            pages += 1
            if len(chunk) < PAGE_SIZE:
                break

        positions = positions[:limit]
        now = utcnow_iso()
        sem = asyncio.Semaphore(self.detail_concurrency)

        async def _to_posting(position: dict) -> Posting:
            raw_url = position.get("positionUrl") or ""
            url = f"{BASE}{raw_url}" if raw_url.startswith("/") else raw_url
            detail = {}
            description = ""
            if url:
                async with sem:
                    try:
                        dr = await self.client.get(
                            url, headers={"Accept": "text/html"}, timeout=30.0
                        )
                        dr.raise_for_status()
                        detail = _extract_jsonld(dr.text)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("microsoft detail failed for %s: %s", position.get("id"), exc)
            if detail.get("description"):
                description = _strip_html(str(detail.get("description", "")))
            title = detail.get("title") or position.get("name", "")
            posted_at = detail.get("datePosted") or _posted_ts(position.get("postedTs"))
            return Posting(
                posting_id=stable_id(self.name, str(position.get("id", "")), url),
                url=url,
                company=company.name,
                title=str(title or ""),
                location=_location_text(position),
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at=str(posted_at or ""),
                description=description,
            )

        return await asyncio.gather(*[_to_posting(p) for p in positions])

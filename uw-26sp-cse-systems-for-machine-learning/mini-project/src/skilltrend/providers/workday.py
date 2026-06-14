"""Workday CXS provider.

Workday-hosted careers sites all expose the same JSON API pattern:

    POST https://{host}/wday/cxs/{tenant}/{site}/jobs    -> list
    GET  https://{host}/wday/cxs/{tenant}/{site}{path}   -> detail

Each entry in companies.yaml must spell out three fields because tenant != host
isn't always derivable (e.g. salesforce uses wd12, nvidia uses wd5):

    workday:
      - slug: nvidia
        name: NVIDIA
        host: nvidia.wd5.myworkdayjobs.com
        tenant: nvidia
        site: NVIDIAExternalCareerSite

Each posting requires one extra HTTP detail call to pull the full description.
The N detail calls per company run concurrently bounded by a Semaphore so we
don't hammer Workday with 50 parallel sockets per company."""
from __future__ import annotations

import asyncio
import html
import logging
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider
from .filters import allowed_by_terms, config_terms

log = logging.getLogger(__name__)

LIST_URL = "https://{host}/wday/cxs/{tenant}/{site}/jobs"
DETAIL_URL = "https://{host}/wday/cxs/{tenant}/{site}{path}"


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _parse_posted_on(value: str) -> str:
    """Workday's `postedOn` is human-readable ("Posted Today", "Posted 30+
    Days Ago"). Convert relative phrases into ISO dates so trend windows can
    use them. Falls back to the empty string when we can't infer a date."""
    if not value:
        return ""
    v = value.strip().lower()
    now = datetime.now(timezone.utc)
    if "today" in v or "yesterday" in v or "just posted" in v:
        return now.date().isoformat()
    m = re.search(r"(\d+)\+?\s*day", v)
    if m:
        days = int(m.group(1))
        from datetime import timedelta
        return (now - timedelta(days=days)).date().isoformat()
    return ""  # unparseable — trends.py will fall back to last_seen


class WorkdayProvider(Provider):
    name = "workday"
    detail_concurrency = 5  # per-company in-flight detail requests

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        host = company.extra.get("host")
        tenant = company.extra.get("tenant", company.slug)
        site = company.extra.get("site")
        if not host or not site:
            raise ValueError(f"workday/{company.slug}: missing host or site in config")

        list_url = LIST_URL.format(host=host, tenant=tenant, site=site)
        search_text = company.extra.get("search_term", company.extra.get("search_text", ""))
        title_include = config_terms(company.extra, "title_include")
        title_exclude = config_terms(company.extra, "title_exclude")
        location_include = config_terms(company.extra, "location_include")
        location_exclude = config_terms(company.extra, "location_exclude")
        max_pages = int(company.extra.get("max_pages", 10))
        payload = {"limit": min(max(limit, 20), 20), "offset": 0, "searchText": search_text}
        postings: list[dict] = []
        # Workday caps a single list call at 20 — paginate until we have `limit`
        offset = 0
        pages = 0
        while len(postings) < limit and pages < max_pages:
            payload["offset"] = offset
            r = await self.client.post(list_url, json=payload, timeout=30.0,
                                       headers={"Accept": "application/json"})
            r.raise_for_status()
            chunk = r.json().get("jobPostings", []) or []
            if not chunk:
                break
            for stub in chunk:
                title = str(stub.get("title", ""))
                location = str(stub.get("locationsText", ""))
                if not allowed_by_terms(title, title_include, title_exclude):
                    continue
                if not allowed_by_terms(location, location_include, location_exclude):
                    continue
                postings.append(stub)
                if len(postings) >= limit:
                    break
            offset += len(chunk)
            pages += 1
            if len(chunk) < payload["limit"]:
                break
        postings = postings[:limit]

        sem = asyncio.Semaphore(self.detail_concurrency)

        async def _detail(stub: dict) -> Posting | None:
            ext_path = stub.get("externalPath", "")
            if not ext_path:
                return None
            url = DETAIL_URL.format(host=host, tenant=tenant, site=site, path=ext_path)
            async with sem:
                try:
                    dr = await self.client.get(url, timeout=30.0,
                                                headers={"Accept": "application/json"})
                    dr.raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    log.warning("workday/%s detail failed: %s", company.slug, exc)
                    return None
            info = dr.json().get("jobPostingInfo", {})
            now = utcnow_iso()
            description = _strip_html(info.get("jobDescription", ""))
            posted_at = (info.get("startDate")
                         or _parse_posted_on(info.get("postedOn", ""))
                         or "")
            return Posting(
                posting_id=stable_id(self.name, info.get("jobReqId", ""),
                                     info.get("externalUrl", "")),
                url=info.get("externalUrl", ""),
                company=company.name,
                title=info.get("title", stub.get("title", "")),
                location=info.get("location", stub.get("locationsText", "")) or "",
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at=posted_at,
                description=description,
            )

        results = await asyncio.gather(*[_detail(s) for s in postings])
        return [p for p in results if p is not None]

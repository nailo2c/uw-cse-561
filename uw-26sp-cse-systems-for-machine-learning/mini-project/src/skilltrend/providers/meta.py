"""Meta Careers provider.

Meta's careers site is a Comet app. The usable path today is:

1. GET `/jobsearch/` to read the public `lsd` token and `_js_datr` cookie value.
2. POST `/graphql` with the current job-search `doc_id`.
3. Fetch each job detail page and extract the embedded
   `xcp_requisition_job_description` payload for full description text.

The `doc_id` is intentionally configurable because Meta rotates these IDs.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import re

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider
from .filters import allowed_by_terms, config_terms

log = logging.getLogger(__name__)

BASE = "https://www.metacareers.com"
JOBS_URL = f"{BASE}/jobsearch/"
GRAPHQL_URL = f"{BASE}/graphql"
DEFAULT_DOC_ID = "9114524511922157"
DETAIL_CONCURRENCY = 5
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_bootstrap_tokens(page: str) -> tuple[str, str]:
    lsd = ""
    datr = ""
    m = re.search(r'\["LSD",\[\],\{"token":"([^"]+)"', page)
    if m:
        lsd = m.group(1)
    m = re.search(r'"_js_datr":\{"value":"([^"]+)"', page)
    if m:
        datr = m.group(1)
    return lsd, datr


def _raw_decode_object_after(text: str, marker: str) -> dict:
    idx = text.find(marker)
    if idx < 0:
        return {}
    start = text.find("{", idx + len(marker))
    if start < 0:
        return {}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _description_from_detail_payload(payload: dict) -> str:
    parts: list[str] = []
    raw_desc = payload.get("description")
    if raw_desc:
        try:
            desc_obj = json.loads(raw_desc)
            raw_desc = desc_obj.get("__html", raw_desc)
        except (TypeError, json.JSONDecodeError):
            pass
        text = _strip_html(str(raw_desc))
        if text:
            parts.append(text)
    for label, key in (
        ("Minimum qualifications", "minimum_qualifications"),
        ("Preferred qualifications", "preferred_qualifications"),
        ("Responsibilities", "responsibilities"),
    ):
        values = []
        for item in payload.get(key) or []:
            if isinstance(item, dict) and item.get("item"):
                values.append(str(item["item"]))
            elif isinstance(item, str):
                values.append(item)
        if values:
            parts.append(
                label + ":\n" + "\n".join(f"- {_strip_html(v)}" for v in values)
            )
    return "\n\n".join(parts)


class MetaProvider(Provider):
    name = "meta"
    detail_concurrency = DETAIL_CONCURRENCY

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        bootstrap = await self.client.get(
            JOBS_URL,
            headers={"Accept": "text/html", **HEADERS},
            timeout=30.0,
            follow_redirects=True,
        )
        bootstrap.raise_for_status()
        lsd, datr = _extract_bootstrap_tokens(bootstrap.text)
        if not lsd or not datr:
            raise RuntimeError("meta: bootstrap tokens missing")

        title_include = config_terms(company.extra, "title_include")
        title_exclude = config_terms(company.extra, "title_exclude")
        location_include = config_terms(company.extra, "location_include")
        location_exclude = config_terms(company.extra, "location_exclude")
        results_per_page = company.extra.get("results_per_page")

        variables = {
            "search_input": {
                "q": company.extra.get("search_term", ""),
                "divisions": [],
                "offices": company.extra.get("offices", ["North America"]),
                "roles": [],
                "leadership_levels": company.extra.get(
                    "leadership_levels", ["Individual Contributor"]
                ),
                "saved_jobs": [],
                "saved_searches": [],
                "sub_teams": [],
                "teams": company.extra.get("teams", []),
                "is_leadership": False,
                "is_remote_only": False,
                "sort_by_new": True,
                "page": 1,
                "results_per_page": results_per_page,
            }
        }
        r = await self.client.post(
            GRAPHQL_URL,
            data={
                "lsd": lsd,
                "variables": json.dumps(variables, separators=(",", ":")),
                "doc_id": str(company.extra.get("doc_id", DEFAULT_DOC_ID)),
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Cookie": f"datr={datr};",
                **HEADERS,
            },
            timeout=30.0,
        )
        r.raise_for_status()
        search_jobs = (((r.json() or {}).get("data") or {}).get("job_search") or [])
        jobs = []
        for job in search_jobs:
            title = str(job.get("title", ""))
            location = ", ".join(str(x) for x in (job.get("locations") or []) if x)
            if not allowed_by_terms(title, title_include, title_exclude):
                continue
            if not allowed_by_terms(location, location_include, location_exclude):
                continue
            jobs.append(job)
            if len(jobs) >= limit:
                break
        now = utcnow_iso()
        sem = asyncio.Semaphore(self.detail_concurrency)

        async def _to_posting(job: dict) -> Posting:
            job_id = str(job.get("id") or "")
            url = f"{BASE}/profile/job_details/{job_id}" if job_id else BASE
            detail_url = f"{BASE}/profile/job_details/{job_id}/" if job_id else BASE
            description = ""
            title = job.get("title", "")
            location = ", ".join(str(x) for x in (job.get("locations") or []) if x)
            async with sem:
                try:
                    dr = await self.client.get(
                        detail_url,
                        headers={"Accept": "text/html", **HEADERS},
                        timeout=30.0,
                        follow_redirects=True,
                    )
                    dr.raise_for_status()
                    payload = _raw_decode_object_after(
                        dr.text, '"xcp_requisition_job_description":'
                    )
                    if payload:
                        title = payload.get("title") or title
                        location = (
                            ", ".join(str(x) for x in (payload.get("locations") or []) if x)
                            or location
                        )
                        description = _description_from_detail_payload(payload)
                except Exception as exc:  # noqa: BLE001
                    log.warning("meta detail failed for %s: %s", job_id, exc)
            return Posting(
                posting_id=stable_id(self.name, job_id, url),
                url=url,
                company=company.name,
                title=str(title or ""),
                location=location,
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at="",
                description=description,
            )

        return await asyncio.gather(*[_to_posting(j) for j in jobs])

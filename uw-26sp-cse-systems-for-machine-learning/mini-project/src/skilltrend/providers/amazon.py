"""Amazon Jobs provider.

amazon.jobs has a single public endpoint that returns rich JSON in one shot
(no per-job detail call needed):

    GET https://www.amazon.jobs/en/search.json?result_limit=N&offset=X

The companies.yaml entry only needs a slug for naming consistency; the
provider always hits the same global Amazon endpoint:

    amazon:
      - slug: amazon
        name: Amazon
"""
from __future__ import annotations

from datetime import datetime

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider
from .filters import allowed_by_terms, config_terms

BASE = "https://www.amazon.jobs"
SEARCH = f"{BASE}/en/search.json"


def _parse_posted_date(value: str) -> str:
    """Amazon posts dates like "November 4, 2025". Convert to ISO date."""
    if not value:
        return ""
    try:
        return datetime.strptime(value.strip(), "%B %d, %Y").date().isoformat()
    except ValueError:
        return value  # store raw; trends._parse_ts will reject and fall back


class AmazonProvider(Provider):
    name = "amazon"
    page_size = 100  # amazon caps per-request at 100

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        out: list[Posting] = []
        now = utcnow_iso()
        offset = 0
        search_term = company.extra.get("search_term")
        categories = company.extra.get("categories") or []
        country_codes = company.extra.get("country_codes") or []
        title_include = config_terms(company.extra, "title_include")
        title_exclude = config_terms(company.extra, "title_exclude")
        location_include = config_terms(company.extra, "location_include")
        location_exclude = config_terms(company.extra, "location_exclude")
        max_pages = int(company.extra.get("max_pages", 10))
        pages = 0

        while len(out) < limit:
            params = {
                "result_limit": min(self.page_size, max(limit - len(out), limit)),
                "offset": offset,
                "sort": "recent",
            }
            if search_term:
                params["base_query"] = search_term
            for category in categories:
                params.setdefault("category[]", []).append(category)
            for country_code in country_codes:
                params.setdefault("country[]", []).append(country_code)
            r = await self.client.get(SEARCH, params=params, timeout=30.0)
            r.raise_for_status()
            data = r.json()
            jobs = data.get("jobs", []) or []
            if not jobs:
                break
            for j in jobs:
                title = j.get("title", "")
                location = j.get("location", "") or j.get("normalized_location", "")
                if not allowed_by_terms(title, title_include, title_exclude):
                    continue
                if not allowed_by_terms(location, location_include, location_exclude):
                    continue
                path = j.get("job_path", "")
                url = f"{BASE}{path}" if path.startswith("/") else path
                out.append(Posting(
                    posting_id=stable_id(self.name, j.get("id", ""), url),
                    url=url,
                    company=company.name,
                    title=title,
                    location=location,
                    source=self.name,
                    first_seen=now,
                    last_seen=now,
                    active=True,
                    posted_at=_parse_posted_date(j.get("posted_date", "")),
                    description="\n\n".join(filter(None, [
                        j.get("description", ""),
                        ("BASIC QUALIFICATIONS\n" + j.get("basic_qualifications", "")
                         if j.get("basic_qualifications") else ""),
                        ("PREFERRED QUALIFICATIONS\n" + j.get("preferred_qualifications", "")
                         if j.get("preferred_qualifications") else ""),
                    ])).strip(),
                ))
                if len(out) >= limit:
                    break
            offset += len(jobs)
            pages += 1
            if len(jobs) < params["result_limit"]:
                break
            if pages >= max_pages:
                break
        return out[:limit]

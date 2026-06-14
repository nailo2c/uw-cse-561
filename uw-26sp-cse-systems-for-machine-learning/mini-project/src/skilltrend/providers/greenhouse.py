from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider

BASE = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class GreenhouseProvider(Provider):
    name = "greenhouse"

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        url = BASE.format(slug=company.slug)
        # content=true asks Greenhouse to include the full HTML job description.
        r = await self.client.get(url, params={"content": "true"}, timeout=30.0)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])[:limit]
        now = utcnow_iso()
        out: list[Posting] = []
        for j in jobs:
            posting_id = stable_id(self.name, str(j.get("id", "")), j.get("absolute_url", ""))
            description = _strip_html(j.get("content", ""))
            location_obj = j.get("location") or {}
            out.append(Posting(
                posting_id=posting_id,
                url=j.get("absolute_url", ""),
                company=company.name,
                title=j.get("title", ""),
                location=location_obj.get("name", "") if isinstance(location_obj, dict) else "",
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at=j.get("updated_at", "") or j.get("first_published", ""),
                description=description,
            ))
        return out

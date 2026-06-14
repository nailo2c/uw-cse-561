from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider

BASE = "https://api.ashbyhq.com/posting-api/job-board/{slug}"


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class AshbyProvider(Provider):
    name = "ashby"

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        url = BASE.format(slug=company.slug)
        # Ashby exposes both html and plain description; prefer plain.
        r = await self.client.get(url, timeout=30.0)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
        # Ashby returns listed + unlisted; only keep what the public board shows.
        jobs = [j for j in jobs if j.get("isListed", True)][:limit]
        now = utcnow_iso()
        out: list[Posting] = []
        for j in jobs:
            posting_id = stable_id(self.name, str(j.get("id", "")), j.get("jobUrl", ""))
            description = j.get("descriptionPlain") or _strip_html(j.get("descriptionHtml", ""))
            out.append(Posting(
                posting_id=posting_id,
                url=j.get("jobUrl", ""),
                company=company.name,
                title=j.get("title", ""),
                location=j.get("location", "") or "",
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at=j.get("publishedAt", ""),
                description=description,
            ))
        return out

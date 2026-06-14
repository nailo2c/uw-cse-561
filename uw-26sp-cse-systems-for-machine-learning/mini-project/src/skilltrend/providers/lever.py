from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider

BASE = "https://api.lever.co/v0/postings/{slug}"


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class LeverProvider(Provider):
    name = "lever"

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        url = BASE.format(slug=company.slug)
        r = await self.client.get(url, params={"mode": "json"}, timeout=30.0)
        r.raise_for_status()
        jobs = r.json()[:limit]
        now = utcnow_iso()
        out: list[Posting] = []
        for j in jobs:
            posting_id = stable_id(self.name, j.get("id", ""), j.get("hostedUrl", ""))
            categories = j.get("categories") or {}
            description_parts = [j.get("descriptionPlain") or _strip_html(j.get("description", ""))]
            for section in j.get("lists", []) or []:
                description_parts.append(section.get("text", ""))
            description_parts.append(j.get("additionalPlain") or "")
            description = "\n\n".join(p for p in description_parts if p).strip()
            out.append(Posting(
                posting_id=posting_id,
                url=j.get("hostedUrl", ""),
                company=company.name,
                title=j.get("text", ""),
                location=categories.get("location", ""),
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at=str(j.get("createdAt", "")),
                description=description,
            ))
        return out

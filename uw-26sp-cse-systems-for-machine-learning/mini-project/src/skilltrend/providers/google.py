"""Google Careers provider.

The JSON API used by older scrapers currently returns 404, but the public
careers HTML still contains stable job-result links and detail pages with
visible qualification sections. This provider intentionally parses only the
public HTML and does not require browser automation.
"""
from __future__ import annotations

import asyncio
import html
import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..models import Posting, stable_id, utcnow_iso
from .base import CompanyRef, Provider
from .filters import allowed_by_terms, config_terms

log = logging.getLogger(__name__)

BASE = "https://www.google.com/about/careers/applications"
SEARCH_URL = f"{BASE}/jobs/results/"
DETAIL_CONCURRENCY = 5

HEADERS = {
    "Accept": "text/html,application/xhtml+xml",
    "User-Agent": "skilltrend/0.1",
}


def _strip_html(raw: str) -> str:
    text = BeautifulSoup(html.unescape(raw or ""), "html.parser").get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _job_id_from_url(url: str) -> str:
    m = re.search(r"/jobs/results/(\d+)", url)
    return m.group(1) if m else url


def _absolute_url(href: str) -> str:
    if href.startswith("/about/careers/applications/"):
        return urljoin("https://www.google.com", href)
    if href.startswith("/jobs/"):
        return f"{BASE}{href}"
    return urljoin(f"{BASE}/", href)


def _parse_search(html_text: str) -> tuple[list[dict], str]:
    soup = BeautifulSoup(html_text, "html.parser")
    jobs: list[dict] = []
    seen: set[str] = set()
    next_url = ""
    for a in soup.find_all("a", href=True):
        label = a.get("aria-label", "")
        href = str(a["href"])
        url = _absolute_url(href)
        path = urlparse(url).path
        if "next page" in label.lower():
            next_url = url
            continue
        if not path.startswith("/about/careers/applications/jobs/results/"):
            continue
        if "previous page" in label.lower():
            continue
        if url in seen:
            continue
        seen.add(url)
        title_node = a.find(["h2", "h3"])
        title = (
            title_node.get_text(" ", strip=True)
            if title_node
            else label.replace("Learn more about", "").strip()
        )
        loc_node = a.find("p")
        location = loc_node.get_text(" ", strip=True) if loc_node else ""
        if title:
            jobs.append({"title": title, "url": url, "location": location})
    return jobs, next_url


def _detail_metadata(html_text: str) -> tuple[str, str]:
    soup = BeautifulSoup(html_text, "html.parser")
    location = ""
    loc_node = soup.select_one(".sPeqm .r0wTof") or soup.select_one(".r0wTof")
    if loc_node is not None:
        location = loc_node.get_text(" ", strip=True)
    detail = soup.find("div", class_="KwJkGe")
    if detail is not None:
        return _strip_html(str(detail)), location
    main = soup.find("main")
    if main is not None:
        return _strip_html(str(main)), location
    meta = soup.find("meta", attrs={"name": "description"})
    return _strip_html(meta.get("content", "") if meta else ""), location


class GoogleProvider(Provider):
    name = "google"
    detail_concurrency = DETAIL_CONCURRENCY

    async def fetch(self, company: CompanyRef, limit: int) -> list[Posting]:
        params = {
            "q": company.extra.get("search_term", ""),
            "location": company.extra.get("location", "United States"),
            "sort_by": company.extra.get("sort_by", "date"),
        }
        title_include = config_terms(company.extra, "title_include")
        title_exclude = config_terms(company.extra, "title_exclude")
        location_include = config_terms(company.extra, "location_include")
        location_exclude = config_terms(company.extra, "location_exclude")
        max_pages = int(company.extra.get("max_pages", 5))
        listings: list[dict] = []
        next_url = SEARCH_URL
        pages = 0

        while next_url and len(listings) < limit and pages < max_pages:
            if next_url == SEARCH_URL:
                r = await self.client.get(next_url, params=params, headers=HEADERS, timeout=30.0)
            else:
                r = await self.client.get(next_url, headers=HEADERS, timeout=30.0)
            r.raise_for_status()
            chunk, next_url = _parse_search(r.text)
            pages += 1
            if not chunk:
                break
            for item in chunk:
                if not allowed_by_terms(item["title"], title_include, title_exclude):
                    continue
                if not allowed_by_terms(
                    item.get("location", ""), location_include, location_exclude
                ):
                    continue
                listings.append(item)
                if len(listings) >= limit:
                    break

        listings = listings[:limit]
        now = utcnow_iso()
        sem = asyncio.Semaphore(self.detail_concurrency)

        async def _to_posting(item: dict) -> Posting:
            description = ""
            location = item.get("location", "")
            async with sem:
                try:
                    dr = await self.client.get(item["url"], headers=HEADERS, timeout=30.0)
                    dr.raise_for_status()
                    description, detail_location = _detail_metadata(dr.text)
                    location = detail_location or location
                except Exception as exc:  # noqa: BLE001
                    log.warning("google detail failed for %s: %s", item["url"], exc)
            return Posting(
                posting_id=stable_id(self.name, _job_id_from_url(item["url"]), item["url"]),
                url=item["url"],
                company=company.name,
                title=item["title"],
                location=location,
                source=self.name,
                first_seen=now,
                last_seen=now,
                active=True,
                posted_at="",
                description=description,
            )

        return await asyncio.gather(*[_to_posting(i) for i in listings])

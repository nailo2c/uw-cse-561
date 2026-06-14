from __future__ import annotations

import httpx
import pandas as pd
import pytest

from skilltrend.providers import jobspy_provider as jobspy_module
from skilltrend.providers.amazon import AmazonProvider
from skilltrend.providers.apple import AppleProvider
from skilltrend.providers.base import CompanyRef
from skilltrend.providers.google import GoogleProvider
from skilltrend.providers.jobspy_provider import JobSpyProvider
from skilltrend.providers.meta import MetaProvider
from skilltrend.providers.microsoft import MicrosoftProvider
from skilltrend.providers.workday import WorkdayProvider


def test_scanner_registry_includes_company_providers():
    from skilltrend.scanner import PROVIDER_REGISTRY

    assert PROVIDER_REGISTRY["apple"] is AppleProvider
    assert PROVIDER_REGISTRY["amazon"] is AmazonProvider
    assert PROVIDER_REGISTRY["google"] is GoogleProvider
    assert PROVIDER_REGISTRY["microsoft"] is MicrosoftProvider
    assert PROVIDER_REGISTRY["meta"] is MetaProvider
    assert PROVIDER_REGISTRY["workday"] is WorkdayProvider
    assert PROVIDER_REGISTRY["jobspy"] is JobSpyProvider


@pytest.mark.asyncio
async def test_amazon_provider_applies_search_and_local_filters():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["base_query"] == "software engineer"
        return httpx.Response(200, json={
            "jobs": [
                {
                    "id": "a1",
                    "title": "Software Development Engineer",
                    "location": "US, WA, Seattle",
                    "job_path": "/jobs/a1/software-development-engineer",
                    "posted_date": "May 30, 2026",
                    "description": "Build Python services.",
                    "basic_qualifications": "Python",
                    "preferred_qualifications": "Kubernetes",
                },
                {
                    "id": "a2",
                    "title": "Software Development Engineer",
                    "location": "MX, DIF, Mexico City",
                    "job_path": "/jobs/a2/software-development-engineer",
                    "posted_date": "May 30, 2026",
                    "description": "Should be filtered by location.",
                },
                {
                    "id": "a3",
                    "title": "Financial Analyst",
                    "location": "US, WA, Seattle",
                    "job_path": "/jobs/a3/financial-analyst",
                    "posted_date": "May 30, 2026",
                    "description": "Should be filtered by title.",
                },
            ],
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await AmazonProvider(client).fetch(
            CompanyRef(
                slug="amazon",
                name="Amazon",
                extra={
                    "search_term": "software engineer",
                    "title_include": ["software", "engineer"],
                    "location_include": ["US,"],
                },
            ),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].title == "Software Development Engineer"
    assert jobs[0].location == "US, WA, Seattle"
    assert "Kubernetes" in jobs[0].description


@pytest.mark.asyncio
async def test_apple_provider_uses_csrf_and_maps_search_results():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/CSRFToken":
            return httpx.Response(200, headers={"x-apple-csrf-token": "tok"})
        if request.url.path == "/api/v1/search":
            assert request.headers["x-apple-csrf-token"] == "tok"
            return httpx.Response(200, json={
                "res": {
                    "totalRecords": 1,
                    "searchResults": [{
                        "positionId": "2001",
                        "postingTitle": "ML Systems Engineer",
                        "transformedPostingTitle": "ml-systems-engineer",
                        "jobSummary": "<p>Build PyTorch services.</p>",
                        "postDateInGMT": "2026-05-30",
                        "locations": [{
                            "city": "Cupertino",
                            "stateProvince": "CA",
                            "countryName": "United States",
                        }],
                        "team": {"teamName": "Machine Learning and AI"},
                    }],
                },
            })
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await AppleProvider(client).fetch(
            CompanyRef(slug="apple", name="Apple", extra={"search_term": "ml"}),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].source == "apple"
    assert jobs[0].title == "ML Systems Engineer"
    assert "PyTorch" in jobs[0].description
    assert "Cupertino, CA, United States" == jobs[0].location


@pytest.mark.asyncio
async def test_microsoft_provider_fetches_detail_jsonld():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/pcsx/search":
            return httpx.Response(200, json={
                "data": {
                    "positions": [{
                        "id": 123,
                        "name": "Software Engineer",
                        "positionUrl": "/careers/job/123",
                        "postedTs": 1780156153,
                        "locations": ["United States, Washington, Redmond"],
                    }],
                },
            })
        if request.url.path == "/careers/job/123":
            return httpx.Response(200, text="""
                <html><script type="application/ld+json">
                {"@context":"http://schema.org","@type":"JobPosting",
                 "title":"Software Engineer",
                 "datePosted":"2026-05-30T15:49:13",
                 "description":"<p>Build Azure and Kubernetes systems.</p>"}
                </script></html>
            """)
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await MicrosoftProvider(client).fetch(
            CompanyRef(slug="microsoft", name="Microsoft", extra={"search_term": "software"}),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].source == "microsoft"
    assert jobs[0].url == "https://apply.careers.microsoft.com/careers/job/123"
    assert "Azure" in jobs[0].description
    assert jobs[0].posted_at.startswith("2026-05-30")


@pytest.mark.asyncio
async def test_microsoft_provider_filters_non_software_titles_before_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/pcsx/search":
            return httpx.Response(200, json={
                "data": {
                    "positions": [
                        {
                            "id": 111,
                            "name": "Principal Signoff CAD Engineer",
                            "positionUrl": "/careers/job/111",
                            "locations": ["United States, Washington, Redmond"],
                        },
                        {
                            "id": 222,
                            "name": "Principal Software Engineer",
                            "positionUrl": "/careers/job/222",
                            "locations": ["United States, Washington, Redmond"],
                        },
                    ],
                },
            })
        if request.url.path == "/careers/job/111":
            raise AssertionError("filtered CAD job should not fetch detail")
        if request.url.path == "/careers/job/222":
            return httpx.Response(200, text="""
                <html><script type="application/ld+json">
                {"@context":"http://schema.org","@type":"JobPosting",
                 "title":"Principal Software Engineer",
                 "description":"<p>Build distributed systems.</p>"}
                </script></html>
            """)
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await MicrosoftProvider(client).fetch(
            CompanyRef(
                slug="microsoft",
                name="Microsoft",
                extra={
                    "search_term": "software",
                    "title_include": ["software"],
                    "title_exclude": ["cad"],
                },
            ),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].title == "Principal Software Engineer"


@pytest.mark.asyncio
async def test_meta_provider_bootstraps_graphql_and_extracts_detail_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/jobsearch/":
            return httpx.Response(200, text=(
                '["LSD",[],{"token":"lsd-token"},123]'
                '"_js_datr":{"value":"datr-token","expiration_for_js":1}'
            ))
        if request.url.path == "/graphql":
            assert "datr=datr-token" in request.headers["cookie"]
            return httpx.Response(200, json={
                "data": {
                    "job_search": [{
                        "id": "248",
                        "title": "Software Engineer, Systems ML",
                        "locations": ["Menlo Park, CA"],
                    }],
                },
            })
        if request.url.path == "/profile/job_details/248/":
            return httpx.Response(200, text=r'''
                <script>
                "xcp_requisition_job_description":{
                  "id":"248",
                  "title":"Software Engineer, Systems ML",
                  "locations":["Menlo Park, CA","Seattle, WA"],
                  "description":"{\"__html\":\"<span>Build PyTorch inference systems.</span>\"}",
                  "minimum_qualifications":[{"item":"Python experience"}],
                  "preferred_qualifications":[{"item":"Kubernetes experience"}]
                }
                </script>
            ''')
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await MetaProvider(client).fetch(
            CompanyRef(slug="meta", name="Meta", extra={"search_term": "software"}),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].source == "meta"
    assert jobs[0].location == "Menlo Park, CA, Seattle, WA"
    assert "PyTorch inference" in jobs[0].description
    assert "Kubernetes experience" in jobs[0].description


@pytest.mark.asyncio
async def test_meta_provider_filters_non_software_titles_before_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/jobsearch/":
            return httpx.Response(200, text=(
                '["LSD",[],{"token":"lsd-token"},123]'
                '"_js_datr":{"value":"datr-token","expiration_for_js":1}'
            ))
        if request.url.path == "/graphql":
            return httpx.Response(200, json={
                "data": {
                    "job_search": [
                        {
                            "id": "100",
                            "title": "Mechanical Design Engineer",
                            "locations": ["Redmond, WA"],
                        },
                        {
                            "id": "200",
                            "title": "Software Engineer, Infrastructure",
                            "locations": ["Menlo Park, CA"],
                        },
                    ],
                },
            })
        if request.url.path == "/profile/job_details/100":
            raise AssertionError("filtered mechanical job should not fetch detail")
        if request.url.path == "/profile/job_details/200/":
            return httpx.Response(200, text=r'''
                <script>
                "xcp_requisition_job_description":{
                  "id":"200",
                  "title":"Software Engineer, Infrastructure",
                  "locations":["Menlo Park, CA"],
                  "description":"{\"__html\":\"<span>Build Python services.</span>\"}"
                }
                </script>
            ''')
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await MetaProvider(client).fetch(
            CompanyRef(
                slug="meta",
                name="Meta",
                extra={
                    "search_term": "software",
                    "title_include": ["software"],
                    "title_exclude": ["mechanical"],
                },
            ),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].title == "Software Engineer, Infrastructure"


@pytest.mark.asyncio
async def test_google_provider_parses_search_links_and_detail_text():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs/results/"):
            return httpx.Response(200, text="""
                <a class="WpHeLc" href="/jobs/results/987-ml-platform-engineer"
                   aria-label="Learn more about ML Platform Engineer">
                  <h3>ML Platform Engineer</h3>
                  <p>Sunnyvale, CA, USA</p>
                </a>
            """)
        if request.url.path.endswith("/jobs/results/987-ml-platform-engineer"):
            return httpx.Response(200, text="""
                <html><div class="KwJkGe">
                  <h3>Minimum qualifications:</h3>
                  <ul><li>Experience with Python and TensorFlow.</li></ul>
                  <h3>Preferred qualifications:</h3>
                  <ul><li>Experience with Kubernetes.</li></ul>
                </div></html>
            """)
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await GoogleProvider(client).fetch(
            CompanyRef(slug="google", name="Google", extra={"search_term": "ml"}),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].source == "google"
    assert jobs[0].title == "ML Platform Engineer"
    assert jobs[0].location == "Sunnyvale, CA, USA"
    assert "TensorFlow" in jobs[0].description


@pytest.mark.asyncio
async def test_google_provider_applies_title_filters():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs/results/"):
            return httpx.Response(200, text="""
                <a class="WpHeLc" href="/jobs/results/123-data-center-technician"
                   aria-label="Learn more about Data Center Technician Operations III">
                  <h3>Data Center Technician Operations III</h3>
                  <p>Sunnyvale, CA, USA</p>
                </a>
                <a class="WpHeLc" href="/jobs/results/987-software-engineer"
                   aria-label="Learn more about Software Engineer III">
                  <h3>Software Engineer III</h3>
                  <p>Sunnyvale, CA, USA</p>
                </a>
            """)
        if request.url.path.endswith("/jobs/results/123-data-center-technician"):
            raise AssertionError("filtered technician job should not fetch detail")
        if request.url.path.endswith("/jobs/results/987-software-engineer"):
            return httpx.Response(200, text="""
                <html><div class="KwJkGe">
                  <h3>Minimum qualifications:</h3>
                  <ul><li>Experience with Python.</li></ul>
                </div></html>
            """)
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await GoogleProvider(client).fetch(
            CompanyRef(
                slug="google",
                name="Google",
                extra={
                    "search_term": "software",
                    "title_include": ["software"],
                    "title_exclude": ["technician"],
                },
            ),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].title == "Software Engineer III"


@pytest.mark.asyncio
async def test_workday_provider_applies_title_filters_before_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wday/cxs/acme/Site/jobs":
            payload = request.read().decode()
            assert "software engineer" in payload
            return httpx.Response(200, json={
                "jobPostings": [
                    {
                        "title": "Mechanical Engineer",
                        "locationsText": "US, WA, Seattle",
                        "externalPath": "/job/mechanical-engineer",
                    },
                    {
                        "title": "Software Engineer",
                        "locationsText": "US, WA, Seattle",
                        "externalPath": "/job/software-engineer",
                    },
                ],
            })
        if request.url.path == "/wday/cxs/acme/Site/job/mechanical-engineer":
            raise AssertionError("filtered mechanical job should not fetch detail")
        if request.url.path == "/wday/cxs/acme/Site/job/software-engineer":
            return httpx.Response(200, json={
                "jobPostingInfo": {
                    "jobReqId": "sw-1",
                    "externalUrl": "https://jobs.example/software-engineer",
                    "title": "Software Engineer",
                    "location": "US, WA, Seattle",
                    "startDate": "2026-05-30",
                    "jobDescription": "<p>Build Python services.</p>",
                },
            })
        raise AssertionError(f"unexpected request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        jobs = await WorkdayProvider(client).fetch(
            CompanyRef(
                slug="acme",
                name="Acme",
                extra={
                    "host": "jobs.example",
                    "tenant": "acme",
                    "site": "Site",
                    "search_term": "software engineer",
                    "title_include": ["software"],
                    "title_exclude": ["mechanical"],
                },
            ),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].title == "Software Engineer"


@pytest.mark.asyncio
async def test_jobspy_provider_maps_tesla_linkedin_fallback(monkeypatch):
    def fake_scrape_jobs(**kwargs):
        assert kwargs["site_name"] == ["linkedin"]
        assert kwargs["linkedin_company_ids"] == [15564]
        return pd.DataFrame([{
            "id": "tesla-1",
            "job_url": "https://linkedin.example/jobs/tesla-1",
            "title": "Software Engineer, Autonomy",
            "location": "Palo Alto, CA",
            "description": "Build Python and PyTorch tooling for autonomy systems.",
            "date_posted": "2026-05-30",
        }])

    monkeypatch.setattr(jobspy_module, "JOBSPY_AVAILABLE", True)
    monkeypatch.setattr(jobspy_module, "_scrape_jobs", fake_scrape_jobs)

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: None)) as client:
        jobs = await JobSpyProvider(client).fetch(
            CompanyRef(
                slug="tesla-linkedin",
                name="Tesla",
                extra={
                    "sites": ["linkedin"],
                    "search_term": "software engineer",
                    "linkedin_company_ids": [15564],
                    "location": "United States",
                },
            ),
            limit=1,
        )

    assert len(jobs) == 1
    assert jobs[0].company == "Tesla"
    assert jobs[0].source == "jobspy"
    assert "PyTorch" in jobs[0].description

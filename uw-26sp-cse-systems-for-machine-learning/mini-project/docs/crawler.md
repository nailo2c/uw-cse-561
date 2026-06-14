# Crawler Architecture

This document explains how skilltrend collects job postings — what each
provider does, what endpoint it hits, what semantics its `posted_at` carries,
and how to add new companies.

## Overall flow

```text
config/companies.yaml
        │
        ▼
src/skilltrend/scanner.py   ── single async orchestrator
        │
        │   one task per (provider, company) pair, all in parallel
        ▼
src/skilltrend/providers/
        ├── greenhouse.py    public ATS JSON API
        ├── lever.py         public ATS JSON API
        ├── ashby.py         public ATS JSON API
        ├── workday.py       public Workday CXS JSON API   (1 list + N detail calls)
        ├── amazon.py        amazon.jobs custom JSON API
        ├── apple.py         jobs.apple.com JSON API + CSRF bootstrap
        ├── google.py        careers.google.com public HTML parser
        ├── microsoft.py     apply.careers.microsoft.com JSON API + detail JSON-LD
        ├── meta.py          metacareers GraphQL + detail embedded JSON
        └── jobspy_provider.py  python-jobspy → LinkedIn (aggregator fallback)
        │
        ▼
src/skilltrend/storage.py   ── append/refresh data/postings/postings.csv
```

Every provider returns the **same `Posting` model**, so the downstream agent
extractor / normaliser / trend code is provider-agnostic. Adding a provider
is just dropping a new file into `providers/` and registering it in
`scanner.PROVIDER_REGISTRY`.

## Provider reference

### 1. Greenhouse — `providers/greenhouse.py`

- **Endpoint**: `https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true`
- **Auth**: none
- **Cost / rate limit**: free, no published limit
- **`posted_at`**: ISO 8601 (`updated_at` or `first_published`)
- **Notes**: single request per company, full description inlined.

```yaml
greenhouse:
  - slug: anthropic        # = path segment in the URL
    name: Anthropic
```

Discover new companies: try `curl -sI https://boards-api.greenhouse.io/v1/boards/<guess>/jobs` — 200 means hit, 404 means not on Greenhouse.

### 2. Lever — `providers/lever.py`

- **Endpoint**: `https://api.lever.co/v0/postings/{slug}?mode=json`
- **Auth**: none
- **Cost / rate limit**: free, no published limit
- **`posted_at`**: Unix epoch ms (e.g. `1753687796431`) — parsed by `trends._parse_ts`
- **Notes**: single request per company, full description inlined.

```yaml
lever:
  - slug: mistral
    name: Mistral AI
```

### 3. Ashby — `providers/ashby.py`

- **Endpoint**: `https://api.ashbyhq.com/posting-api/job-board/{slug}`
- **Auth**: none (this is the public **Job Board API**, not the authenticated **ATS API**)
- **Cost / rate limit**: free, no published limit
- **`posted_at`**: ISO 8601 (`publishedAt`)
- **Notes**: single request per company. Response is large (~10 MB for OpenAI's
  716 jobs). The provider filters `isListed=true` so unpublished drafts don't
  leak in.

```yaml
ashby:
  - slug: openai
    name: OpenAI
```

### 4. Workday — `providers/workday.py`

- **List endpoint**: `POST https://{host}/wday/cxs/{tenant}/{site}/jobs`
- **Detail endpoint**: `GET https://{host}/wday/cxs/{tenant}/{site}{externalPath}`
- **Auth**: none
- **Cost / rate limit**: free; we cap **5 concurrent detail calls per company** to be polite
- **`posted_at`**: detail response's `startDate` (ISO) if present, else parsed from `postedOn` strings like "Posted Today" / "Posted 30+ Days Ago" (converted to ISO date)
- **Notes**: 1+N HTTP calls per company (list + one detail per posting). Slower than ATS providers; expect ~5–10s per company for 50 postings. `search_term`, `title_include`, `title_exclude`, `location_include`, and `location_exclude` can narrow noisy Workday boards before detail fetches.

```yaml
workday:
  - slug: nvidia
    name: NVIDIA
    host: nvidia.wd5.myworkdayjobs.com
    tenant: nvidia
    site: NVIDIAExternalCareerSite
```

**Adding a new Workday company** — Workday URLs vary in three dimensions
(`host`, `tenant`, `site`) and none are derivable from the brand name. Find the values by:

1. Open the company's public careers page (e.g. `https://careers.intel.com`)
2. Click any job opening; the URL will redirect to the Workday subdomain
3. Read off the parts: `https://intel.wd1.myworkdayjobs.com/External/job/...`
   - `host` = `intel.wd1.myworkdayjobs.com`
   - `tenant` = `intel` (first path segment after `/wday/cxs/`)
   - `site` = `External` (second path segment)
4. Verify:
   ```bash
   curl -X POST "https://intel.wd1.myworkdayjobs.com/wday/cxs/intel/External/jobs" \
        -H 'Content-Type: application/json' \
        -d '{"limit":1,"offset":0,"searchText":""}' \
        -o /dev/null -w "HTTP %{http_code}\n"
   ```
   200 → add it. 422 → wrong `site` name (Workday returns 422 on unknown sites).

Confirmed working hosts/tenants/sites for popular Mag-7-tier companies:

| Company | host | tenant | site |
|---|---|---|---|
| NVIDIA | `nvidia.wd5.myworkdayjobs.com` | `nvidia` | `NVIDIAExternalCareerSite` |
| Salesforce | `salesforce.wd12.myworkdayjobs.com` | `salesforce` | `External_Career_Site` |
| Intel | `intel.wd1.myworkdayjobs.com` | `intel` | `External` |

Some Workday-hosted companies expose non-obvious `site` names. Those remain
manual additions: find the host / tenant / site in browser DevTools, then add
the three fields to `companies.yaml`.

### 5. Amazon — `providers/amazon.py`

- **Endpoint**: `GET https://www.amazon.jobs/en/search.json?result_limit=N&offset=X`
- **Auth**: none
- **Cost / rate limit**: free, no published limit; we paginate with 100/page
- **`posted_at`**: parsed from human-readable "November 4, 2025" to ISO date
- **Notes**: single endpoint (no per-job detail call needed — `description`, `basic_qualifications`, `preferred_qualifications` all inline). 10,000+ jobs available globally; cap with `SKILLTREND_MAX_POSTINGS_PER_COMPANY` or `--limit`. Use `search_term`, `country_codes`, and title/location include/exclude filters to avoid broad Amazon roles.

```yaml
amazon:
  - slug: amazon
    name: Amazon
```

### 6. Company-specific Mag-7 providers

These providers exist because several Mag-7 companies do not use standard
Greenhouse / Lever / Ashby / Workday boards:

| Provider | Endpoint / source | Detail strategy | Main risk |
|---|---|---|---|
| Apple | `jobs.apple.com/api/v1/CSRFToken` then `POST /api/v1/search` | search response includes `jobSummary` | summary is shorter than a full ATS description |
| Google | `careers.google.com/jobs/results/` HTML | fetch each public detail page and parse qualification text | HTML selectors can break on redesign |
| Microsoft | `apply.careers.microsoft.com/api/pcsx/search` | fetch `positionUrl`, parse JobPosting JSON-LD | detail page HTML may change |
| Meta | `/jobsearch` bootstrap tokens then `POST /graphql` | fetch detail page and parse `xcp_requisition_job_description` | GraphQL `doc_id` can rotate |

```yaml
apple:
  - slug: apple
    name: Apple
    search_terms: [siri, machine learning, software platform, cloud, security engineer]
    title_include: [developer, scientist, architect, machine learning, ml, ai, siri, software, platform, security, data]
    title_exclude: [product manager, program manager, sales, marketing, recruiter, mechanical, electrical]
    max_pages: 3

google:
  - slug: google
    name: Google
    search_term: software engineer
    location: United States
    title_include: [software, developer, site reliability, machine learning, ai, ml, cloud, infrastructure, platform]
    title_exclude: [technician, product manager, program manager, rtl design]

microsoft:
  - slug: microsoft
    name: Microsoft
    search_term: software engineer
    title_exclude: [hardware systems, signoff cad, product manager, program manager]

meta:
  - slug: meta
    name: Meta
    search_term: software engineer
    offices: [North America]
    title_include: [software, production engineer, data engineer, research engineer, machine learning, infrastructure]
    title_exclude: [mechanical, electrical, critical facility, robotics hardware]
```

### 7. JobSpy (LinkedIn aggregator) — `providers/jobspy_provider.py`

- **Underlying library**: [`python-jobspy`](https://github.com/speedyapply/JobSpy) — wraps LinkedIn / Indeed / Glassdoor / Google / ZipRecruiter
- **Auth**: none required; proxies recommended for sustained scraping
- **Cost / rate limit**: LinkedIn rate-limits aggressively after ~10 pages from one IP. Keep `results_wanted` small (≤50/company) without proxies
- **`posted_at`**: ISO date from JobSpy's `date_posted` field
- **Notes**: synchronous library; we wrap in `asyncio.to_thread` so it doesn't block the event loop. Output is normalised into the same `Posting` model.

```yaml
jobspy:
  - slug: tesla-linkedin
    name: Tesla
    sites: [linkedin]
    search_term: software engineer
    linkedin_company_ids: [15564]
    location: United States
```

**Why this provider exists**: Tesla's public endpoint is Akamai-protected
against non-browser clients, and any company-specific provider can break when
the company changes its site. JobSpy → LinkedIn is the low-maintenance fallback
when a direct provider is blocked or stale.

**Gray-area warning**: this scrapes LinkedIn, which is grey-zone under their
ToS. Acceptable for academic research; **do not deploy commercially**.

**Finding `linkedin_company_ids`**:
1. Visit `https://linkedin.com/company/<slug>` in a browser
2. View page source, search for `"company":{"urn":"urn:li:fsd_company:`
3. The integer that follows is the company ID

Common Mag-7 IDs:

| Company | linkedin_company_id |
|---|---:|
| Apple | 162479 |
| Google (Alphabet) | 1441 |
| Microsoft | 1035 |
| Meta | 10667 |
| Amazon | 1586 |
| Tesla | 15564 |
| Netflix | 165245 |
| NVIDIA | 3608 |

The Tesla JobSpy entry is enabled by default to complete Mag-7 coverage.
Comment out the `jobspy:` section if you want to avoid LinkedIn scraping.

## Trade-offs at a glance

| Provider | Auth | RPS / cost | Detail cost | `posted_at` quality | Mag-7 coverage |
|---|---|---|---|---|---|
| Greenhouse | none | free | 0 (inline) | ISO | none |
| Lever | none | free | 0 (inline) | epoch ms (parsed) | none |
| Ashby | none | free | 0 (inline) | ISO | none |
| Workday | none | free | **N detail calls** | mixed (relative→ISO) | NVIDIA, Salesforce, Intel cleanly |
| Amazon | none | free | 0 (inline) | parsed ISO | Amazon only |
| Apple | CSRF cookie | free | 0 (summary inline) | ISO/date string | Apple |
| Google | none | free | **N detail HTML calls** | falls back to scan time | Google |
| Microsoft | none | free | **N detail HTML calls** | ISO/epoch parsed | Microsoft |
| Meta | bootstrap token | free | **N detail HTML calls** | falls back to scan time | Meta |
| JobSpy | none | LinkedIn rate-limits | 0 (inline) | ISO | Tesla + fallback for all Mag-7 (gray area) |

## Design rationale: why one CSV across providers

All providers write to the same `data/postings/postings.csv` keyed by
`posting_id = sha1(provider | external_id | url)[:16]`. This means:

- Re-scanning is idempotent — same posting from same provider always hashes
  the same; storage layer refreshes `last_seen` instead of duplicating
- Trend analysis is single-table — no JOIN gymnastics across provider-
  specific tables
- Adding a 7th, 8th, ... provider doesn't require any schema migration
- The same data flows through extraction / normalisation / blocklist
  filtering regardless of source

The tradeoff: providers with materially different semantics (e.g. JobSpy's
LinkedIn job descriptions can be paywalled / truncated) just live as
"slightly noisier rows" in the same table. The extraction pipeline handles
this naturally because the LLM already deals with arbitrary description text.

## What's *not* supported, and why

This section records the remaining unsupported paths after adding direct
providers for Apple, Google, Microsoft, and Meta. The recommended fallback is
the existing JobSpy → LinkedIn provider.

### Research summary (probe results, 2026-05-30)

We probed each company's careers endpoint with realistic browser headers
to see if a public JSON API was reachable:

| Company | Endpoint tried | Result | Root cause |
|---|---|---|---|
| Apple legacy | `POST jobs.apple.com/api/role/search` | **400/301** | Replaced by the working `/api/v1/CSRFToken` + `/api/v1/search` flow |
| Google API | `careers.google.com/api/v3/{search,search-jobs}/` | **404** | API path changed or removed; provider now parses public HTML |
| Tesla | `tesla.com/cua-api/{jobs,apps/careers/state}` | **403 Akamai bot-shield** | Active bot detection; rejects non-browser traffic |
| Microsoft Workday | `microsoft.wd5.myworkdayjobs.com/wday/cxs/microsoft/{8 site names tried}/jobs` | **422 on all** | Microsoft careers now works through Eightfold, not Workday |

Bottom line: the only Mag-7 company still without a clean HTTP-only provider is
Tesla. Apple / Google / Microsoft / Meta are reachable, but their providers are
less stable than ATS providers because they depend on company-specific web
contracts.

### GitHub projects we evaluated

| Project | Stars | Method | Status | Decision |
|---|---:|---|---|---|
| [ever-jobs/ever-jobs](https://github.com/ever-jobs/ever-jobs) | 54 | company-specific providers | active in 2026 | ✅ Borrowed endpoint patterns for Apple / Microsoft / Tesla |
| [kbhujbal/go-get-jobs](https://github.com/kbhujbal/go-get-jobs) | 38 | Go scrapers | active in 2026 | ✅ Borrowed Google HTML and Meta GraphQL patterns |
| [thayton/apple-job-scraper](https://github.com/thayton/apple-job-scraper) | <10 | unknown | 5 commits, blog-companion code | ❌ Hobby, no maintenance evidence |
| [alimahmoud7/google-jobs-scraper](https://github.com/alimahmoud7/google-jobs-scraper) | 40 | Selenium + ChromeDriver | 16 commits | ❌ HTML-dependent, brittle to redesigns |
| [anon767/maangcrawler](https://github.com/anon767/maangcrawler) | 5 | unknown (Flask UI) | 15 commits | ❌ Too immature |
| Tesla HTTP-only scrapers | — | Tesla CUA API | brittle | ❌ Endpoint exists but frequently hits Akamai challenge |
| [speedyapply/JobSpy](https://github.com/speedyapply/JobSpy) | 3.5k | LinkedIn / Indeed APIs | Active, v1.1.79 (2026-03) | ✅ **Already integrated as `providers/jobspy_provider.py`** |

### Why we *don't* add Playwright/Selenium scrapers by default

A direct browser-automation provider would, in our environment:

- Add **~500 MB** to the Docker image (Chromium binary)
- Take **5–18 minutes per company** per scan (vs ~5 s for an ATS provider)
- Break on every site redesign — empirically ~once per 6–12 months
- Cost roughly **1 engineer-week per company** to write and harden
- Yield job descriptions that are often **paywalled or truncated** without
  authenticated sessions

For an academic prototype optimising for measurable inference-engine
behaviour, that cost ratio is wrong. Tesla can still be represented through
the JobSpy → LinkedIn path without pulling a browser into the default runtime.

### Re-checking whether anything has opened up

If you want to re-evaluate later, the probe script lives at
`tools/probe_endpoints.sh`. Run it once a quarter; if any endpoint flips
from 4xx to 200, revisit this section.

### Other categories explicitly out of scope

- **Aggregator sites beyond LinkedIn / Indeed** (Glassdoor scraping
  directly, Hacker News "Who's Hiring" threads, niche Slack channels) —
  small-corpus, hard to normalise.
- **Tesla direct scraping without a real browser session** — blocked by Akamai
  often enough that the provider would be noisy in unattended scans.
- **Workday companies with hidden `site` slugs** — solvable per-company with
  browser-DevTools work; documented in the "Adding a new Workday company"
  subsection.

## Operational guidance

- **Cron cadence**: daily `skilltrend scan` is fine for trend purposes; ATS
  boards rarely change inside an hour. Workday and Amazon detail calls make
  this slower than ATS-only scans — budget ~30s per Workday company at
  limit=50, ~10s for Amazon at limit=50.
- **Per-company cap**: `SKILLTREND_MAX_POSTINGS_PER_COMPANY` (default 50) caps
  every provider equally. For Amazon (10k+ jobs globally), consider raising it
  to 100–200 when doing a "full" scrape.
- **Failure tolerance**: scanner reports failed companies at the end of each
  scan but never aborts. A wrong Workday site name or a temporarily 5xx ATS
  endpoint loses that company's data for the run; the next run will retry.
- **Rate limit etiquette**: ATS providers are tolerant; Workday's per-company
  Semaphore (5 concurrent) keeps detail calls polite. JobSpy/LinkedIn is the
  only provider with real risk of getting your IP banned — keep
  `results_wanted` ≤ 50 without proxies.

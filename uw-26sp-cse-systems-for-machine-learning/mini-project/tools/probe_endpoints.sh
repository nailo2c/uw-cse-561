#!/usr/bin/env bash
# Probe the public careers endpoints for the FAANG-tier companies we don't
# currently support. Re-run quarterly; if any of these flips from 4xx to 2xx,
# revisit docs/crawler.md and consider writing a provider.
#
# Usage: bash tools/probe_endpoints.sh
set -u

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

probe() {
  local label=$1
  local method=$2
  local url=$3
  local data=${4:-}
  local extra_headers=${5:-}

  if [ "$method" = "POST" ]; then
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$url" \
      -H "User-Agent: $UA" \
      -H "Accept: application/json" \
      -H "Content-Type: application/json" \
      $extra_headers \
      -d "$data")
  else
    code=$(curl -sL -o /dev/null -w "%{http_code}" "$url" \
      -H "User-Agent: $UA" \
      -H "Accept: application/json" \
      $extra_headers)
  fi
  if [ "$code" = "200" ]; then
    echo "  ✅ $label -> HTTP $code  ← endpoint opened up, investigate"
  else
    echo "  ❌ $label -> HTTP $code"
  fi
}

echo "=== Mag-7 careers endpoints ==="

probe "Apple"     POST "https://jobs.apple.com/api/role/search?lang=en-us" \
                  '{"query":"","filters":{},"page":1}'

probe "Google v3" GET  "https://careers.google.com/api/v3/search/?company=Google&page=1&page_size=1"

probe "Tesla"     GET  "https://www.tesla.com/cua-api/apps/careers/state?country=US&start=0&limit=1"

probe "Meta v1"   GET  "https://www.metacareers.com/v1/jobs"

echo ""
echo "=== Microsoft Workday (we know the tenant; site slug is the unknown) ==="
for site in External Microsoft MSFTcareers Worldwide Careers ExternalCareerSite External_Career_Site Careers_Site; do
  probe "  microsoft/$site" POST \
    "https://microsoft.wd5.myworkdayjobs.com/wday/cxs/microsoft/$site/jobs" \
    '{"limit":1,"offset":0,"searchText":""}'
done

echo ""
echo "Done. If anything flipped to 200, update docs/crawler.md."

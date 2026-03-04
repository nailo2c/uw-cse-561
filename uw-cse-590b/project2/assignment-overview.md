# Assignment Overview

You will write a full, professional-quality post-mortem for a simulated production incident in a cloud application.

The goal is to demonstrate your ability to:

- Analyze a distributed systems failure
- Apply SRE and Well-Architected principles
- Describe cause, effect, and mitigations clearly
- Use a blameless, fact-driven approach
- Propose realistic architectural improvements
- Produce a credible, industry-standard post-mortem document

This project mimics what is required in real cloud engineering roles.

---

## 1️⃣ Scenario: Incident on the "Acme Inventory Service"

You are on the cloud applications team for **Acme Retail**, an e-commerce company with the following architecture:

- **Inventory Service** (Go microservice)
- Backed by **PostgreSQL (RDS)**
- **Redis** used as a read-through cache
- **API Gateway** fronting the service
- Downstream dependency: **Supplier Pricing API**
- Deployed via **Kubernetes** (autoscaled)
- Observability: **Prometheus + Grafana + OpenTelemetry** traces

On **March 18**, the Inventory Service experienced a major incident.

Below is the raw, unprocessed incident data (intentionally messy, incomplete, and partially contradictory). Your job is to build the real story.

---

## 2️⃣ Raw Incident Inputs

### A. Alert Stream (Slack excerpts)

| Time  | Alert / Message                                                    |
| ----- | ------------------------------------------------------------------ |
| 09:07 | Alert: "InventoryService p95 latency > 4s"                        |
| 09:09 | Alert: "Redis errors increasing (CONNECTION_TIMEOUT)"              |
| 09:11 | Alert: "DB CPU 98% (critical)"                                     |
| 09:14 | Alert: "InventoryService error rate > 22%"                         |
| 09:15 | On-call acknowledges                                               |
| 09:21 | "Seeing many queries missing cache — spike from 5% to 72% miss rate" |
| 09:31 | "Nodes at max CPU, autoscaler added 2 pods but still overloaded"   |
| 09:46 | "Supplier Pricing API latency > 5s"                                |
| 10:02 | "DB connection pool saturated (200/200)"                           |
| 10:19 | Failover initiated manually                                        |
| 10:27 | System stabilizing                                                 |
| 10:41 | Alerts clear                                                       |

### B. Observed Customer Impact

- Some customers saw **"Out of Stock"** for items that were actually in stock.
- Shopping cart page frequently timed out.
- API Gateway returning **504s**.
- Business reports: **18% drop in checkout completion** during outage window.

### C. Screenshots (summarized text form)

- Redis latency spiked from **3ms → 200ms**, then many requests timed out.
- Cache miss rate increased dramatically for 15 minutes.
- PostgreSQL CPU flatlined at **100%** between 09:11 and 10:25.
- Query heatmap shows top offender:
  ```sql
  SELECT stock FROM inventory WHERE item_id = ?
  ```
- InventoryService CPU also spiked, mostly waiting on DB calls.

### D. System Activity Logs (summaries)

- A new feature flag enabling **"Real-Time Supplier Pricing Sync"** was turned on at **09:05**.
- The feature performs a **synchronous call** to Supplier Pricing API on every inventory lookup when pricing is >12 hours old.
- Supplier Pricing API experienced a **regional slowdown** beginning around 09:28.
- Cache TTL for item stock was recently reduced from **6 hours → 15 minutes**.
- DBA confirms an **index on `inventory(item_id)` was accidentally dropped** during a schema migration two days prior.

### E. Partial Slack Conversation

> "Is this a thundering herd?"
>
> "Why are so many cache misses suddenly happening?"
>
> "Did we change Redis connection limits recently?"
>
> "Why is DB using sequential scan?"
>
> "If Supplier Pricing API is slow, do we block inventory?"
>
> "Can we turn off the feature flag?"
>
> "Failover helped but not fully."

### F. Mitigation Actions Observed

- On-call manually toggled off the feature flag (**09:52**)
- Redis cluster was restarted at **10:00** (did not help much)
- DB failover at **10:19** improved CPU usage
- Increased Redis connection pool
- Temporarily increased cache TTL to **1 hour**

---

## 3️⃣ Your Required Deliverables

You must produce a professional post-mortem document (**4–8 pages**) including:

### 1. Executive Summary (non-technical)

- What happened
- Impact to customers
- Duration and severity
- High-level root cause
- Short overview of the fix

> Should be 4–6 sentences max.

### 2. Detailed Customer Impact Section

Quantify wherever possible:

- Requests failing
- Latency spikes
- Business metrics (checkout drop)
- Regions affected
- SLO violations
- Duration per symptom

### 3. Precise Incident Timeline (with minute-level detail)

Based on raw data, construct a factually consistent timeline. Should include:

- Alert triggers
- Human actions
- Automated responses
- Observed changes in system state
- Deployment/feature-flag actions
- Mitigating steps

### 4. Root Cause Analysis

Must identify and differentiate:

#### A. Trigger

The first event that led to system failure.

#### B. Contributing Factors

Architectural or operational weaknesses that made the system more fragile.

#### C. True Root Cause(s)

Underlying, systemic issues that allowed the trigger to cause an outage.

#### D. Why It Was Not Detected Earlier

Weaknesses in:

- Monitoring
- Alerts
- Testing
- Rollout procedure
- Capacity planning
- Observability

> This section should be evidence-backed, not speculative.

### 5. Technical Deep Dive

Include relevant details, such as:

- How the dropped index affected query performance
- Why synchronous pricing calls created cascading latency
- Cache-miss explosion and Redis connection pool saturation
- How PostgreSQL responded to load
- The effects of TTL changes on traffic patterns
- Whether this was a thundering herd scenario
- Why autoscaling didn't help

> Include diagrams if helpful (optional).

### 6. Mitigation & Short-Term Fixes

List exactly what mitigations were taken, when, and by whom.

### 7. Long-Term Preventive Actions

Must include:

- Clear owner
- Deadline
- Measurable outcome
- Dependencies
- Operational or architectural changes

**Examples:**

- Add back and verify DB index
- Add caching layer fallback
- Implement circuit breaker for Supplier Pricing API
- Increase Redis observability dashboards
- Add feature-flag canarying process
- Revisit cache TTL strategy

### 8. Lessons Learned

Focus on:

- Architectural practices
- Cloud design patterns (cache-aside, bulkhead, circuit breaker)
- Reliability engineering principles
- What tradeoffs were misjudged

### 9. Validation Plan

How the team will verify that fixes work:

- Load testing
- Chaos experiments
- Canary rollouts
- Targeted monitoring

---

## 4️⃣ Grading Rubric

| Category                    | Points | Criteria                                                        |
| --------------------------- | ------ | --------------------------------------------------------------- |
| Executive Summary           | 2      | Clear, concise, non-technical, accurate                         |
| Customer Impact             | 2      | Quantified, specific, aligned to SLOs/SLIs                     |
| Timeline                    | 3      | Complete, detailed, consistent with logs                        |
| Root Cause Analysis         | 5      | Blameless, evidence-based, correctly separates trigger/contributing/root |
| Technical Depth             | 3      | Shows systems thinking, cloud design patterns applied           |
| Preventive Actions          | 3      | Actionable, owned, realistic, tied to architecture              |
| Clarity and Professionalism | 2      | Structure, writing quality, diagrams, readability               |
| **Total**                   | **20** |                                                                 |

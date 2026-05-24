# Systems for ML - Project Proposal: Agentic Skill-Demand Trend Analyzer

+ Student: Aaron Chen
+ NetID: aaronyc

## 1. The Application

I propose to build an agentic application for tracking how technical skill demand changes in the job market over time. The goal is not to automate job applications. Instead, the system would behave like a labor-market analysis tool: it would collect job postings, extract skill requirements from job descriptions, and report which skills are increasing or decreasing in demand over configurable time windows such as one month, three months, six months, and one year.

The application would be structured as a market analytics pipeline rather than a candidate-job matching tool. Instead of asking "Should I apply to this role?", the application asks "What skills are appearing more often in recent job postings, and which ones are becoming less common?"

The core workflow will be:

```text
Job sources
  -> provider-based scanner
  -> raw job posting snapshot store
  -> LLM-based skill extraction workers
  -> skill normalization and taxonomy mapping
  -> time-window trend analysis
  -> report, dashboard, or API output
```

The agentic design pattern is likely to combine planning and decomposition, dynamic tool orchestration, and multi-agent collaboration:

- A planner agent can decompose the market-analysis request into source collection, job-description extraction, skill extraction, normalization, and trend analysis steps.
- Source-specific collector tools can fetch postings from public ATS providers or job-board APIs through a shared provider interface.
- Multiple extraction workers can process job descriptions in parallel and return structured skill records.
- A normalization/review step may map equivalent surface forms to a canonical skill name and group related concepts when the evidence supports doing so.
- A trend analyst agent can generate final summaries, including rising skills, declining skills, representative postings, and confidence caveats.

The initial prototype would likely focus on software engineering and AI-related roles, such as AI engineer, ML engineer, backend engineer, platform engineer, MLOps/LLMOps engineer, data engineer, and full-stack engineer. The data scope would be explicitly bounded so that results are interpretable rather than presented as a complete view of the entire labor market.

Because many job boards do not expose complete historical postings, the prototype would treat historical analysis as a function of the snapshot database. It should support one-month, three-month, six-month, and one-year windows over collected or backfilled records, but the final writeup would clearly distinguish between real collected snapshots, postings with exposed dates, and any replayed benchmark traces used for controlled evaluation.

## 2. The Execution Plan

The prototype is expected to be a local application with a small data pipeline, an agentic processing layer, and an evaluation harness.

### Data Collection

The scanner may use a provider-based design:

- Each job source can be implemented as a provider module with a common interface.
- Initial providers would target public ATS endpoints or job-board APIs where possible.
- Each scan should produce normalized raw records containing fields such as `url`, `company`, `title`, `location`, `source`, `first_seen`, `last_seen`, `active_status`, and raw job-description text when available.
- The system should preserve repeated snapshots rather than deduplicating everything away, because trend analysis depends on observing changes over time.

For benchmarking, I plan to save a fixed dataset of job postings instead of relying only on live job-board results. Live postings can disappear, change, or move during development, which would make performance comparisons unreliable. A fixed corpus of around 500-1,000 postings would let me rerun the same workload when comparing different agent designs, concurrency settings, caching strategies, and inference backends.

### Agent Logic

The main agent could operate over a task such as:

```text
Analyze AI and software engineering skill demand over the last 90 days.
Show rising and declining skills, and explain the evidence.
```

The planner can decide which subtasks to run:

1. Load or refresh the job snapshot corpus.
2. Select postings within the requested time window and comparison baseline.
3. Dispatch job-description extraction or skill extraction workers.
4. Normalize extracted skills against a taxonomy.
5. Compute trend metrics.
6. Generate a final report with caveats and representative examples.

The extraction worker would call an LLM with a structured-output prompt. For each job description, it may extract:

- Required technical skills
- Preferred/nice-to-have skills
- Tools, frameworks, platforms, and cloud providers
- Role family
- Seniority level
- Domain tags such as AI, backend, data, infra, security, or frontend
- Evidence spans from the job description

If included, normalization would reduce noisy strings into canonical entities only when doing so is justified by the chosen taxonomy. Possible examples include:

```text
{ "React.js", "ReactJS", "React" } -> "React"
{ "AWS Lambda", "Lambda" } -> "AWS Lambda"
{ "LLM agents", "AI agents", "agentic workflows" } -> "AI agents"
```

I will compare at least two execution strategies:

1. A single-agent sequential baseline that processes postings one at a time.
2. A multi-worker agentic pipeline that processes postings concurrently and then runs a normalization/review step.

If time permits, I will also evaluate a reflection/self-correction variant where the model checks whether each extracted skill is explicitly supported by evidence in the job description.

### Inference Setup

Because the application relies on LLM calls for skill extraction, the prototype needs an inference backend that can be measured. I plan to write the agent against an OpenAI-compatible LLM endpoint so that the backend can be swapped during development. One candidate backend is vLLM on Cloud TPU, following the course guidance, but the application should not depend on one specific provider or serving stack.

This setup is needed for the systems portion of the project. I want to measure how the agent workflow stresses the inference backend, including request throughput, token throughput, p50/p95 latency, queueing delay, and the effect of worker concurrency. Since job descriptions are relatively long and the extracted JSON outputs are short, I expect the workload to be input-heavy, making prefill latency and batching behavior especially important.

### Metrics and Evaluation

At the application level, I will measure:

- End-to-end latency for a full analysis request
- Per-posting extraction latency
- Throughput in postings/minute
- Number of LLM calls per posting
- Token usage per posting and per report
- Skill extraction quality on a manually labeled subset
- Normalization consistency across equivalent skill names
- Trend stability under repeated runs

At the inference/system level, I will measure:

- vLLM request throughput
- Input tokens/sec and output tokens/sec
- p50/p95 latency per extraction request
- Queueing delay under different concurrency levels
- Effect of batch size and parallel workers
- Difference between the sequential baseline and multi-worker pipeline

The correctness evaluation will use a small manually labeled gold set of job descriptions. I will label expected skills and compare the system's extracted canonical skills against that set using precision, recall, and F1. I will also inspect representative errors, such as hallucinated skills, missed implicit requirements, and over-normalization of distinct skills.

## 3. Hypothesis

My main hypothesis is that the primary bottleneck will be LLM inference on long job descriptions, especially the prefill phase. Each posting contains a large input prompt consisting of the system instruction, taxonomy guidance, and the job description, but the desired output is usually a compact structured record. This makes the workload different from open-ended chat: it is dominated by repeated long-input extraction requests.

I expect the single-agent sequential baseline to have poor end-to-end latency because it will issue many dependent LLM calls. The multi-worker design should improve application-level throughput by processing postings concurrently, but only up to the point where vLLM queueing and prefill saturation dominate.

The likely optimization opportunities are:

- Batch multiple extraction requests through vLLM to improve accelerator utilization.
- Reduce prompt size by separating stable taxonomy instructions from per-posting content where possible.
- Use a smaller model for extraction and reserve a larger model only for final trend interpretation.
- Cache extraction results by job URL and content hash so repeated trend queries do not reprocess unchanged postings.
- Replace some LLM calls with deterministic preprocessing, such as regex-based metadata extraction for title, location, and source.
- Use hierarchical aggregation: extract skills per posting once, then answer many time-window queries from the structured database without re-calling the model.
- Tune worker concurrency to minimize end-to-end latency without causing excessive queueing delay.

I also expect a quality/latency tradeoff. A smaller model may extract common skills quickly but may do worse on emerging AI-related terminology, such as agentic workflows, RAG evaluation, LLM observability, MCP, or inference optimization. A larger model may improve normalization and reduce hallucinated skills, but increase latency and lower throughput. The project will characterize this tradeoff and identify which parts of the workflow actually need a stronger model.

By the end of the project, I expect to have a functional prototype that can ingest a repeatable corpus of job postings, extract and normalize skills with an agentic workflow, compute time-window trend statistics, and produce a report of rising and declining skills. The systems analysis will connect the agent design pattern to measurable inference costs and evaluate how batching, concurrency, caching, and prompt reduction affect the user-facing latency of the application.

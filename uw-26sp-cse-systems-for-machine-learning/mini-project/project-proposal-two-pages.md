# Systems for ML - Project Proposal: Agentic Skill-Demand Trend Analyzer

+ Student: Aaron Chen
+ NetID: aaronyc

## 1. The Application

I propose to build an agentic application that tracks how technical skill demand changes in the job market over time. The goal is not to automate job applications. Instead, the system would behave like a labor-market analysis tool: it would collect job postings, extract skill requirements from job descriptions, and report which skills are increasing or decreasing in demand over configurable time windows such as one month, three months, six months, and one year.

The application would be structured as a market analytics pipeline rather than a candidate-job matching tool. Instead of asking "Should I apply to this role?", the application asks "What skills are appearing more often in recent job postings, and which ones are becoming less common?"

The core workflow would be:

```text
Job sources -> posting snapshot store -> LLM skill extraction workers
            -> optional skill normalization -> trend analysis -> report/dashboard
```

The agentic pattern is likely to combine planning and decomposition, dynamic tool orchestration, and multi-agent collaboration. A planner agent can break a market-analysis request into subtasks such as loading job snapshots, selecting the requested time window, dispatching extraction workers, computing trend metrics, and generating a final report. Source-specific collector tools can fetch postings from public ATS providers or job-board APIs. Multiple extraction workers can process job descriptions in parallel and return structured skill records. If included, a normalization/review step would aggregate equivalent skill mentions under canonical names when justified by the chosen taxonomy, so that counts are not split across spelling variants or synonyms.

The initial prototype would likely focus on software engineering and AI-related roles, such as AI engineer, ML engineer, backend engineer, platform engineer, MLOps/LLMOps engineer, data engineer, and full-stack engineer. The data scope would be explicitly bounded so that results are interpretable rather than presented as a complete view of the entire labor market.

## 2. The Execution Plan

The prototype is expected to be a local application with three parts: a small data collection pipeline, an agentic processing layer, and an evaluation harness.

For data collection, the scanner may use a provider-based design. Each job source can be implemented as a provider module with a common interface, targeting public ATS endpoints or job-board APIs when available. Each scan should produce raw records containing fields such as `url`, `company`, `title`, `location`, `source`, `first_seen`, `last_seen`, `active_status`, and job-description text when available. The system should preserve repeated snapshots rather than deduplicating everything away, because trend analysis depends on observing changes over time.

For benchmarking, I plan to save a fixed dataset of job postings instead of relying only on live job-board results. Live postings can disappear, change, or move during development, which would make performance comparisons unreliable. A fixed corpus of around 500-1,000 postings would let me rerun the same workload when comparing different agent designs, concurrency settings, caching strategies, and inference backends.

After the system has collected raw job-description text, an extraction worker would use an LLM with a structured-output prompt to convert each unstructured JD into structured fields such as required skills, preferred skills, tools/frameworks/platforms, role family, seniority, domain tags, and evidence spans from the JD. Most of the trend computation itself can then be done by deterministic code over the structured records.

For a typical analysis request, the agentic workflow could run through these stages:

1. Load or refresh the job snapshot corpus.
2. Select postings within the requested time window and comparison baseline.
3. Dispatch job-description extraction or skill extraction workers.
4. Optionally normalize equivalent skill mentions under a chosen taxonomy.
5. Compute trend metrics.
6. Generate a final report with caveats and representative examples.

I will compare at least two execution strategies: a sequential baseline that processes postings one at a time, and a parallel multi-worker pipeline that processes postings concurrently. If time permits, I will also test a reflection/self-correction variant that adds an extra model call to verify extracted skills against evidence in the JD, allowing me to study the quality-latency tradeoff.

Because the application relies on LLM calls for skill extraction, the prototype needs an inference backend that can be measured. I plan to write the agent against an OpenAI-compatible LLM endpoint so that the backend can be swapped during development. One candidate backend is vLLM on Cloud TPU, following the course guidance, but the application should not depend on one specific provider or serving stack.

At the application level, I will measure end-to-end latency, per-posting extraction latency, postings processed per minute, number of LLM calls per posting, token usage, and extraction quality on a manually labeled subset. At the inference/system level, I will measure request throughput, token throughput, p50/p95 latency, queueing delay, and the effect of worker concurrency.

## 3. Hypothesis

My main hypothesis is that the primary bottleneck will be LLM inference on long job descriptions, especially the prefill phase. Each posting contains a relatively large input prompt consisting of the extraction instructions, taxonomy guidance, and the job description, while the desired output is usually a compact structured record. This makes the workload input-heavy rather than decode-heavy.

I expect the sequential baseline to have poor end-to-end latency because it processes postings one at a time. A multi-worker agentic pipeline should improve throughput by processing multiple postings concurrently, but only up to the point where the inference backend becomes saturated and queueing delay dominates. The project will therefore study not only whether parallelism helps, but where it stops helping.

Likely optimization opportunities include batching extraction requests, reducing prompt size, caching extraction results by job URL and content hash, replacing simple metadata extraction with deterministic code, using a smaller model for extraction and a larger model only for final trend explanation, and tuning worker concurrency to balance throughput against queueing delay.

I also expect a quality-latency tradeoff. A smaller model may extract common skills quickly but may miss or misclassify emerging AI-related terminology such as agentic workflows, RAG evaluation, LLM observability, or inference optimization. A larger model may improve extraction and normalization quality, but at higher latency and lower throughput. The final project will characterize this tradeoff and connect the agent design pattern to measurable inference costs.

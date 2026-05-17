# Project 2: Cloud Application Post-Mortem Analysis

### Objectives
Students will:
- Analyze a complex distributed system failure.
- Apply SRE and Well-Architected principles to identify root causes.
- Propose architectural improvements and resilience patterns.
- Produce an industry-standard, blameless post-mortem report.

### Scenario
An incident occurred on the **Acme Inventory Service** involving database regressions, cache-miss amplification, and dependency failures.
- **Detailed Scenario:** See [Assignment Overview](./assignment-overview.md) for raw incident data, alert streams, and customer impact details.

### Tasks
1. **Incident Timeline:** Correlate alerts, human actions, and state changes.
2. **Root Cause Analysis (RCA):** Differentiate between the trigger and systemic root causes.
3. **Technical Deep Dive:** Analyze query plan regressions and cascading latency.
4. **Preventive Actions:** Map architectural and operational improvements to measurable outcomes.

### Deliverables
- **[Post-Mortem Report](./post-mortem.md)**: A comprehensive 4–8 page analysis.
- **Validation Plan**: Strategies for verifying fixes through load testing and canary rollouts.

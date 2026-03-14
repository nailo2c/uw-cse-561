# Event-Driven Inventory & Order Processing System

Design a cloud application for a mid-size retailer that must support:

*   Order intake
*   Stock validation
*   Inventory updates
*   Email/SMS notifications
*   Analytics on order trends

## Constraints

*   Must handle unpredictable spikes (e.g., holiday events)
*   Requires at-least-once processing
*   Needs to support async workflows

---

# **Required Deliverables**

## 1. Architecture Design Document (Primary Artifact)

Students must create a **12–20 page architecture document** covering:

### a. Executive Summary

Business goals, success criteria, constraints, risks.

### b. High-Level Architecture

*   Component diagram
*   Data flow diagram
*   Sequence diagrams for key workflows
*   Description of technology choices and tradeoffs

### c. Mapping to Well-Architected Pillars

Must justify decisions across:

*   Operational Excellence
*   Security
*   Reliability
*   Performance Efficiency
*   Cost Optimization
*   Sustainability (optional but encouraged)

### d. Cloud Design Patterns Used

Must include at least **four** of the following patterns (with explanation):

*   Event Sourcing
*   CQRS
*   Cache-Aside
*   Circuit Breaker
*   Queue-Based Load Leveling
*   Strangler Fig
*   Sidecar
*   Retry + Backoff
*   Bulkhead
*   Saga / Orchestration

### e. Data Management Strategy

Include:

*   Consistency model (strong/eventual) and justification
*   Storage types and roles
*   Replication and durability strategy

### f. Reliability & Resilience Plan

Must include:

*   Failure mode analysis
*   Chaos scenarios
*   Recovery strategies
*   RPO/RTO definitions

### g. Cost Model & Performance Model

Include a *lightweight* spreadsheet estimating:

*   Compute costs
*   Storage costs
*   Data transfer costs
*   Scaling behavior under load
*   Bottleneck analysis

### h. Security Model

Threat model including:

*   Identity & access control
*   Data encryption
*   Network boundaries
*   Secrets management

### i. Operations & Observability Plan

Include:

*   Dashboards
*   Log aggregation
*   Metrics
*   Alerts
*   Synthetic checks
*   Runbooks and escalation paths

## 2. Implementation (Minimum Functionality)

Students must "deploy" a **minimum working version** of their system including:

*   Basic API endpoint or event pipeline
*   Functioning deployment (containerized or serverless)
*   Infrastructure-as-Code for at least *some* components
*   Basic logging/metrics
*   Simple load test or synthetic checks

The **goal is not to build a full system** — the goal is to demonstrate that the design can be instantiated and that architectural decisions are grounded in reality. **DO NOT DEPLOY. JUST VERIFY IT BUILDS.**

## 3. Performance & Cost Evaluation

*   **We're not doing this because we're not deploying. But this would be part of a normal design.**

Students must conduct a small experiment:

*   Run a load test (small scale, e.g., 50–200 RPS)
*   Measure latency, error rates, throughput
*   Identify a bottleneck and propose a fix
*   Provide a cost-based scenario analysis (e.g., low/medium/high traffic)

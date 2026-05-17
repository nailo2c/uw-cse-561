# Performance & Cost Evaluation

This document outlines the findings from the local load testing experiment and provides a cost-based scenario analysis for the proposed cloud architecture, fulfilling the requirements for Part 3 of the project.

## 1. Performance Evaluation (Local Experiment)

To validate the resilience of the event-driven architecture, a local load test was conducted against the containerized Order API. 

*For the complete, detailed interactive report, please refer to the included `locust_report.html` file.*

### Experiment Setup
*   **Testing Tool:** Locust
*   **Target:** Local containerized FastAPI application (Order Intake Service)
*   **Parameters:** 200 Concurrent Users, Spawn Rate of 50 users/second.
*   **Duration:** ~3 minutes and 43 seconds

### Measured Metrics
The system demonstrated exceptional stability and performance under sudden load:
*   **Throughput (RPS):** The API sustained an average of **~483 Requests Per Second (RPS)** during the peak of the test.
*   **Latency:**
    *   **Median Response Time:** **2 ms**
    *   **95th Percentile (p95):** **18 ms**
*   **Error Rate:** **0%**. Out of nearly 110,000 requests, zero requests failed.

### Architectural Validation
These metrics strongly validate the **Queue-Based Load Leveling** pattern. Because the API immediately returns an `HTTP 202 Accepted` response and offloads the heavy lifting (publishing to the queue/database) to a background asynchronous task, the frontend remains completely unblocked. The users experience near-instantaneous response times (2ms median) regardless of the background processing queue depth.

### Bottleneck Identification & Proposed Fix
*   **Identified Bottleneck (Local):** While the asynchronous design protects the latency, the ultimate bottleneck in a single-container deployment under extreme load (e.g., pushing beyond 1000 RPS) becomes the CPU. Python's `asyncio` event loop is single-threaded. When the sheer volume of incoming JSON parsing and queue-publishing tasks saturates the single CPU core, throughput is capped, and p99 latency will eventually degrade.
*   **Proposed Fix (Cloud Architecture):** In our actual AWS deployment, this compute bottleneck is resolved via **Horizontal Scaling**. The Order API is deployed as an AWS Fargate service behind an Application Load Balancer (ALB). We would configure Auto Scaling based on CPU Utilization (e.g., target 70%). As traffic spikes, AWS automatically provisions additional Fargate tasks to distribute the compute load. Crucially, scaling the API layer does *not* overwhelm the downstream DynamoDB database, as the intermediate SQS queue acts as a shock absorber.

---

## 2. Cost Evaluation & Scenario Analysis

A lightweight, service-by-service cost estimate was constructed based on AWS serverless and managed services. 

*For the complete line-item breakdown, please refer to the included `cost-estimation.csv` file.*

### Baseline Cost
The estimated monthly baseline cost to run this architecture in a highly available configuration (including Multi-AZ ElastiCache and NAT Gateways for private subnet egress) is approximately **$453.20 / month**. By utilizing Serverless compute (Fargate, Lambda) and On-Demand database capacity (DynamoDB), we avoid the massive upfront costs of over-provisioning idle EC2 instances.

### Cost-Based Scenario Analysis

| Scenario | Expected Traffic | Architectural Behavior & Cost Impact | Mitigation / Bottleneck Strategy |
| :--- | :--- | :--- | :--- |
| **Low / Normal Traffic** | 10 - 50 RPS | The system operates at its baseline capacity. Fargate tasks remain at the minimum count (e.g., 2 instances). Costs remain close to the baseline estimate. | None required. System handles load trivially. |
| **High Traffic (Sales Event)** | 500 - 1000 RPS | Incoming requests surge. **AWS Fargate** automatically scales out (adding more API and Worker tasks), incurring higher compute costs by the minute. **DynamoDB (On-Demand)** seamlessly absorbs the write spike, charging strictly per write request. SQS queues orders temporarily if workers fall behind. | The cost increases linearly with the traffic duration. The architecture operates exactly as intended, prioritizing 100% order capture over strict cost-saving during revenue-generating events. |
| **Extreme Spike (Viral Event)** | 5000+ RPS | Traffic exceeds normal sales expectations. SQS queue depth grows significantly as the workers cannot scale fast enough or DynamoDB hits soft account limits. Compute and database costs spike sharply. | The API Gateway must implement **Rate Limiting (Throttling)** to protect against malicious DDoS or out-of-control bills. Valid orders are queued securely in SQS, delaying fulfillment slightly but ensuring zero data loss and preventing complete account resource exhaustion. |

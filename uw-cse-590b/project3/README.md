# CSE 590B Project 3: Event-Driven Order Processing System

Cloud architecture, containerized MVP, and performance evaluation for an event-driven order processing system.

## Repository Guide

### Part 1: Architecture Design
* `architecture-design-document.pdf`: Primary design document (sections a-i).
* `high_level_architecture.png`: System architecture diagram.

### Part 2: Implementation
* `app/main.py`: FastAPI async order intake.
* `terraform/main.tf`: Infrastructure as Code (DynamoDB, SQS).
* `Dockerfile`: API containerization.
* `how-to-run.md`: Instructions for local execution and testing.

### Part 3: Performance & Cost
* `performance-cost-evaluation.pdf`: Load test results and cost analysis.
* `locust_report.html`: Interactive Locust performance report.
* `cost-estimation.csv`: Itemized AWS cost estimates.

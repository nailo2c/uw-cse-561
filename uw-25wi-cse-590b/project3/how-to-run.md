# How to Run & Verify

This guide outlines how to build and verify the MVP implementation locally. No actual cloud resources are deployed.

## 1. Run the Order API Container
From the project root:
```bash
# Build the Docker image
docker build -t order-api .

# Run the container
docker run -d -p 8000:8000 --name order-api order-api
```

### Verify API & Structured Logging
```bash
# Send a test order
curl -X POST http://localhost:8000/orders \
     -H "Content-Type: application/json" \
     -d '{"user_id": "usr_123", "item_id": "item_A", "quantity": 1}'

# Check logs for Trace-ID and mock async processing
docker logs order-api
```

## 2. Run the Load Test (Performance Evaluation)
```bash
# Install Locust
pip install locust

# Start the load test script
cd scripts
locust -f load_test.py
```
Open `http://localhost:8089` in your browser.
*   **Number of users:** 200
*   **Spawn rate:** 50
*   **Host:** `http://localhost:8000`
Click **Start swarming** to observe latency and throughput.

## 3. Verify Infrastructure-as-Code (Terraform)
```bash
cd terraform
terraform init
terraform plan
```
*(Uses mock credentials to verify syntax without deploying).*

## Cleanup
```bash
docker stop order-api && docker rm order-api
```

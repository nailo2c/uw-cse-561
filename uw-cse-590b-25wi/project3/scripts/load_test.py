from locust import HttpUser, task, between
import uuid
import random

class OrderAPIUser(HttpUser):
    # Wait time between tasks for a single user (simulating realistic behavior)
    wait_time = between(0.1, 0.5)

    @task(3)
    def create_order(self):
        """
        Simulate a user creating a new order.
        This hits our core asynchronous endpoint.
        """
        payload = {
            "user_id": f"usr_{uuid.uuid4().hex[:6]}",
            "item_id": random.choice(["item_A", "item_B", "item_C", "item_D"]),
            "quantity": random.randint(1, 5)
        }
        
        # We expect a 202 Accepted response
        with self.client.post("/orders", json=payload, catch_response=True) as response:
            if response.status_code == 202:
                response.success()
            else:
                response.failure(f"Expected 202, got {response.status_code}")

    @task(1)
    def check_health(self):
        """
        Simulate internal load balancer health checks.
        """
        self.client.get("/health")

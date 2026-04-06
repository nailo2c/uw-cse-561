import contextvars
import logging
import time
import uuid

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from pydantic import BaseModel

# =============================================================================
# Logging Setup
# =============================================================================

# Use ContextVar for safe async tracing
trace_id_var = contextvars.ContextVar("trace_id", default="system")

# Custom formatter to inject trace_id from ContextVar
class TraceIdFilter(logging.Filter):
    def filter(self, record):
        # Always prioritize 'trace_id' passed via 'extra=' in logger.info(..., extra={"trace_id": ...})
        if not hasattr(record, 'trace_id'):
            record.trace_id = trace_id_var.get()
        return True

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "trace_id": "%(trace_id)s", "message": "%(message)s"}'
)
logger = logging.getLogger("order-api")
logger.addFilter(TraceIdFilter())

# Add filter to root logger
logging.getLogger().addFilter(TraceIdFilter())

# =============================================================================
# Models
# =============================================================================

class OrderRequest(BaseModel):
    user_id: str
    item_id: str
    quantity: int

# =============================================================================
# Application Setup & Helper Functions
# =============================================================================

app = FastAPI(title="Event-Driven Order API")

# Mock function to simulate publishing to SQS
def publish_to_sqs(order_data: dict, trace_id: str):
    # Simulate network latency
    time.sleep(0.01) 
    
    # In a real app, we use boto3 here.
    # Pass the explicit trace_id via the 'extra' dict to override context/system defaults
    logger.info(f"[Mock SQS] Published event for order {order_data['order_id']}", extra={"trace_id": trace_id})

# =============================================================================
# Middleware & Endpoints
# =============================================================================

# Middleware to inject trace_id for observability
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    # Check if client sent a trace ID, otherwise generate one
    trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    
    # Set it for the current async context
    token = trace_id_var.set(trace_id)
    
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    
    trace_id_var.reset(token)
    return response


@app.post("/orders", status_code=202)
async def create_order(order: OrderRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Core Order Intake Endpoint.
    Demonstrates Queue-Based Load Leveling and Eventual Consistency.
    Accepts request, immediately returns 202, processes in background.
    """
    order_id = f"ord_{uuid.uuid4().hex[:8]}"
    
    # Retrieve the trace_id injected by the middleware
    trace_id = getattr(request.state, "trace_id", "fallback_id")
    
    order_data = {
        "order_id": order_id,
        "user_id": order.user_id,
        "item_id": order.item_id,
        "quantity": order.quantity,
        "status": "PENDING"
    }
    
    logger.info(f"Received order intake request for item {order.item_id}")
    
    # Hand off the heavy lifting (publishing to queue/db) to a background task
    background_tasks.add_task(publish_to_sqs, order_data, trace_id)
    
    return {"status": "accepted", "order_id": order_id, "message": "Order is being processed asynchronously."}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    # A mock metrics endpoint demonstrating basic observability
    return {
        "active_connections": 1,
        "queue_depth_estimate": 0,
        "status": "ok"
    }

# =============================================================================
# Execution
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

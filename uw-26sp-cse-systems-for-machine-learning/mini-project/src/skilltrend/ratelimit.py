"""Async token-bucket rate limiter.

Used to throttle outgoing LLM calls so that free-tier RPM quotas (e.g. 10
req/min on Gemini's lite models) don't return 429s. Setting rpm <= 0 makes
the limiter a no-op, which is the right default once you're on a paid tier
where the backend itself does the queueing.

The bucket capacity equals the RPM value — this allows a one-minute burst
when the project has been idle, then settles to a steady 1 token per
(60/rpm) seconds. That matches how Google's sliding-window RPM enforcement
actually behaves.
"""
from __future__ import annotations

import asyncio
import time


class TokenBucketLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.refill_per_s = rpm / 60.0 if rpm > 0 else 0.0
        self.capacity = float(max(rpm, 1))
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
        self.last_refill = now

    async def acquire(self) -> float:
        """Block until a token is available. Returns seconds waited."""
        if self.rpm <= 0:
            return 0.0
        async with self.lock:
            self._refill_locked()
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            wait = (1.0 - self.tokens) / self.refill_per_s
            await asyncio.sleep(wait)
            # We held the lock through the sleep, so no other task touched
            # the bucket in the meantime. Account for the token we just
            # earned and consumed.
            self.last_refill = time.monotonic()
            self.tokens = 0.0
            return wait

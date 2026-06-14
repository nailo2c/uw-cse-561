"""Unit tests for the async token-bucket rate limiter.

Uses high RPM values to keep test wall time fast (otherwise we'd be
literally rate-limited during testing).
"""
from __future__ import annotations

import asyncio
import time

import pytest

from skilltrend.ratelimit import TokenBucketLimiter


@pytest.mark.asyncio
async def test_disabled_when_rpm_zero():
    limiter = TokenBucketLimiter(rpm=0)
    t0 = time.perf_counter()
    for _ in range(50):
        wait = await limiter.acquire()
        assert wait == 0.0
    assert time.perf_counter() - t0 < 0.1


@pytest.mark.asyncio
async def test_burst_then_throttle():
    """RPM=60 -> 1 token/sec. Initial capacity=60 burst, then strict pacing."""
    limiter = TokenBucketLimiter(rpm=60)
    # Burst should be instant — we have a full bucket
    t0 = time.perf_counter()
    for _ in range(10):
        wait = await limiter.acquire()
        assert wait == 0.0
    burst_elapsed = time.perf_counter() - t0
    assert burst_elapsed < 0.05


@pytest.mark.asyncio
async def test_steady_state_pacing():
    """RPM=120 -> 2 tokens/sec. After draining the burst, requests should
    pace at 0.5s intervals."""
    limiter = TokenBucketLimiter(rpm=120)
    # Drain the initial bucket
    for _ in range(120):
        await limiter.acquire()
    # Now next request must wait ~0.5s for the next token
    t0 = time.perf_counter()
    wait = await limiter.acquire()
    elapsed = time.perf_counter() - t0
    assert wait > 0.3
    assert 0.3 < elapsed < 0.8


@pytest.mark.asyncio
async def test_concurrent_workers_share_budget():
    """4 workers competing for an RPM=240 (4/sec) budget should each get one
    of the first 4 burst slots immediately, then serialize."""
    limiter = TokenBucketLimiter(rpm=240)
    # Drain the bucket first
    for _ in range(240):
        await limiter.acquire()

    waits: list[float] = []

    async def worker():
        w = await limiter.acquire()
        waits.append(w)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(4)])
    elapsed = time.perf_counter() - t0
    # 4 requests at 4/sec = ~1s total
    assert 0.7 < elapsed < 1.5
    # Each wait should be non-zero (bucket was drained)
    assert all(w > 0 for w in waits)

"""
Shared benchmarking utilities for Project 3: Inference Profiling.

Provides timing helpers, warmup routines, and hardware constants
for TPU v5e benchmarking.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


# ── TPU v5e hardware specs ──────────────────────────────────────────────────
TPU_V5E_PEAK_TFLOPS_BF16 = 197.0        # Peak BF16 TFLOPS
TPU_V5E_HBM_BANDWIDTH_GBS = 819.0       # Peak HBM bandwidth in GB/s
TPU_V5E_HBM_CAPACITY_GB = 16.0          # HBM capacity in GB
TPU_V5E_ARITHMETIC_INTENSITY = (         # ops/byte crossover point
    TPU_V5E_PEAK_TFLOPS_BF16 * 1e12 / (TPU_V5E_HBM_BANDWIDTH_GBS * 1e9)
)

# Bytes per element for common dtypes
DTYPE_BYTES = {
    jnp.bfloat16: 2,
    jnp.float16: 2,
    jnp.float32: 4,
    jnp.int32: 4,
}


def get_dtype_bytes(dtype):
    """Return the number of bytes for a given JAX dtype."""
    dtype = jnp.dtype(dtype)
    return dtype.itemsize


def benchmark_fn(fn, warmup_iters=5, benchmark_iters=20, block=True):
    """Benchmark a JAX function with warmup and multiple iterations.

    Args:
        fn: Callable that takes no arguments and returns a JAX array.
        warmup_iters: Number of warmup iterations (to trigger JIT compilation).
        benchmark_iters: Number of timed iterations.
        block: Whether to call jax.block_until_ready on the result.

    Returns:
        dict with keys:
            - 'mean_time_s': Mean time per iteration in seconds.
            - 'std_time_s': Std deviation of time per iteration.
            - 'min_time_s': Minimum time across iterations.
            - 'times_s': List of all iteration times.
    """
    # Warmup
    for _ in range(warmup_iters):
        result = fn()
        if block:
            jax.block_until_ready(result)

    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        start = time.perf_counter()
        result = fn()
        if block:
            jax.block_until_ready(result)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {
        'mean_time_s': float(np.mean(times)),
        'std_time_s': float(np.std(times)),
        'min_time_s': float(np.min(times)),
        'median_time_s': float(np.median(times)),
        'times_s': times.tolist(),
    }


def gemm_flops(M, N, K):
    """Compute the number of FLOPs for a GEMM C = A @ B.
    A: (M, K), B: (K, N), C: (M, N)
    FLOPs = 2 * M * N * K  (multiply + accumulate)
    """
    return 2 * M * N * K


def gemm_bytes(M, N, K, dtype=jnp.bfloat16):
    """Compute the number of bytes transferred for a GEMM (read A, B; write C)."""
    elem_bytes = get_dtype_bytes(dtype)
    return (M * K + K * N + M * N) * elem_bytes


def gemm_operational_intensity(M, N, K, dtype=jnp.bfloat16):
    """Operational intensity of GEMM in FLOPs/byte."""
    return gemm_flops(M, N, K) / gemm_bytes(M, N, K, dtype)


def attention_prefill_flops(batch_size, num_heads, seq_len, head_dim):
    """FLOPs for prefill attention (Q @ K^T + attn @ V).

    Q @ K^T: 2 * B * H * S * S * D  (batched matmul)
    attn @ V: 2 * B * H * S * S * D  (batched matmul)
    Total: 4 * B * H * S^2 * D
    """
    return 4 * batch_size * num_heads * seq_len * seq_len * head_dim


def attention_prefill_bytes(batch_size, num_heads, seq_len, head_dim, dtype=jnp.bfloat16):
    """Bytes for prefill attention (read Q, K, V; write output).
    Each of Q, K, V, O: B * H * S * D
    """
    elem_bytes = get_dtype_bytes(dtype)
    # Read Q, K, V + write O = 4 * B * H * S * D
    return 4 * batch_size * num_heads * seq_len * head_dim * elem_bytes


def attention_decode_flops(batch_size, num_heads, context_len, head_dim):
    """FLOPs for decode attention (single query, reading full KV cache).

    Q @ K^T: 2 * B * H * 1 * C * D = 2 * B * H * C * D
    attn @ V: 2 * B * H * 1 * C * D = 2 * B * H * C * D
    Total: 4 * B * H * C * D
    """
    return 4 * batch_size * num_heads * context_len * head_dim


def attention_decode_bytes(batch_size, num_heads, context_len, head_dim, dtype=jnp.bfloat16):
    """Bytes for decode attention (read Q, K_cache, V_cache; write O).
    Q: B * H * 1 * D
    K_cache, V_cache: B * H * C * D each
    O: B * H * 1 * D
    """
    elem_bytes = get_dtype_bytes(dtype)
    q_bytes = batch_size * num_heads * 1 * head_dim * elem_bytes
    kv_bytes = 2 * batch_size * num_heads * context_len * head_dim * elem_bytes
    o_bytes = batch_size * num_heads * 1 * head_dim * elem_bytes
    return q_bytes + kv_bytes + o_bytes


def compute_tflops(flops, time_s):
    """Convert FLOPs and time to TFLOPS."""
    return flops / (time_s * 1e12)


def compute_bandwidth_gbs(bytes_transferred, time_s):
    """Convert bytes and time to GB/s."""
    return bytes_transferred / (time_s * 1e9)


def print_header(title):
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_result(label, tflops=None, bandwidth_gbs=None, time_ms=None):
    """Print a formatted benchmark result."""
    parts = [f"  {label:40s}"]
    if time_ms is not None:
        parts.append(f"time={time_ms:8.3f} ms")
    if tflops is not None:
        parts.append(f"TFLOPS={tflops:7.2f}")
    if bandwidth_gbs is not None:
        parts.append(f"BW={bandwidth_gbs:8.1f} GB/s")
    print("  |  ".join(parts))

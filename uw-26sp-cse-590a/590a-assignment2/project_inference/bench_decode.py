"""
Section 2, Q2.2: Decode Attention Benchmarking on TPU v5e.

Benchmarks single-layer decode attention (single query token reading KV cache)
using XLA attention. Measures memory bandwidth utilization (GB/s).

Decode attention is inherently memory-bound because it reads the entire
KV cache but only computes a single query's worth of attention.

Bandwidth note: measured GB/s may peak and then decline at very large batch
sizes. This happens because once batch × context grows large enough, the
operation transitions from fully memory-bound toward partially compute-bound —
XLA tiles the batched matmul differently, and total time grows faster than
the (proportionally larger) byte count, reducing apparent bandwidth.

Usage:
    python3.11 project_inference/bench_decode.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanogpt"))

import math
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_utils import (
    benchmark_fn,
    attention_decode_flops,
    attention_decode_bytes,
    compute_tflops,
    compute_bandwidth_gbs,
    print_header,
    print_result,
)
from plot_utils import setup_plot_style, save_plot, plot_bandwidth_vs_param, COLORS
from roofline import MODELS


DTYPE = jnp.bfloat16
WARMUP_ITERS = 10
BENCH_ITERS = 50


def make_decode_inputs(batch_size, context_len, num_heads, head_dim, dtype=DTYPE):
    """Create random Q (single token), K_cache, V_cache for decode."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    # Q: single query token (B, 1, H, D)
    q = jax.random.normal(k1, (batch_size, 1, num_heads, head_dim), dtype=dtype)
    # KV cache: (B, context_len, H, D)
    k_cache = jax.random.normal(k2, (batch_size, context_len, num_heads, head_dim), dtype=dtype)
    v_cache = jax.random.normal(k3, (batch_size, context_len, num_heads, head_dim), dtype=dtype)
    return q, k_cache, v_cache


def bench_xla_decode(batch_size, context_len, num_heads, head_dim):
    """Benchmark XLA dot_product_attention for decode (single query).

    q, k_cache, v_cache are passed as JIT arguments (not closed over) so XLA
    cannot constant-fold the inputs — matching real inference where the KV cache
    is a dynamic, growing buffer.

    No causal mask: with a single query position causal masking is trivially
    satisfied (every key position is "before" the only query position).
    """
    q, k_cache, v_cache = make_decode_inputs(batch_size, context_len, num_heads, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    @jax.jit
    def run(q, k_cache, v_cache):
        return jax.nn.dot_product_attention(
            q, k_cache, v_cache, scale=scale, implementation="xla"
        )

    result = benchmark_fn(lambda: run(q, k_cache, v_cache), warmup_iters=WARMUP_ITERS, benchmark_iters=BENCH_ITERS)

    flops = attention_decode_flops(batch_size, num_heads, context_len, head_dim)
    bytes_ = attention_decode_bytes(batch_size, num_heads, context_len, head_dim)

    tflops = compute_tflops(flops, result['median_time_s'])
    bw_gbs = compute_bandwidth_gbs(bytes_, result['median_time_s'])
    return tflops, bw_gbs, result['median_time_s']


def run_context_len_sweep():
    """Experiment 1: Vary context length, batch_size=1."""
    context_lens = [2**i for i in range(7, 14)]  # 128 to 8192
    all_results = {}

    for model_name, model_cfg in MODELS.items():
        num_heads = model_cfg["q_heads"]
        head_dim = model_cfg["head_dim"]
        print_header(f"Decode — {model_name} model, batch=1, varying context_len")

        xla_bw = []
        xla_tflops = []

        for c in context_lens:
            tflops, bw_gbs, time_s = bench_xla_decode(1, c, num_heads, head_dim)
            xla_bw.append(bw_gbs)
            xla_tflops.append(tflops)
            print_result(
                f"XLA      ctx_len={c:>5}",
                tflops=tflops,
                bandwidth_gbs=bw_gbs,
                time_ms=time_s * 1000,
            )

        all_results[model_name] = {"xla_bw": xla_bw, "xla_tflops": xla_tflops}

    return context_lens, all_results


def run_batch_sweep():
    """Experiment 2: Vary batch size, context_len=1024.

    Uses powers of 2 from 1 to 128. At large batch the operation transitions
    from memory-bound toward compute-bound, so measured GB/s may peak and
    decline — this is expected, not a measurement artifact.
    """
    batch_sizes = [2**i for i in range(0, 8)]  # 1 to 128
    context_len = 1024
    all_results = {}

    for model_name, model_cfg in MODELS.items():
        num_heads = model_cfg["q_heads"]
        head_dim = model_cfg["head_dim"]
        print_header(f"Decode — {model_name} model, ctx_len={context_len}, varying batch")

        xla_bw = []
        xla_tflops = []

        for b in batch_sizes:
            tflops, bw_gbs, time_s = bench_xla_decode(b, context_len, num_heads, head_dim)
            xla_bw.append(bw_gbs)
            xla_tflops.append(tflops)
            print_result(
                f"XLA      batch={b:>3}",
                tflops=tflops,
                bandwidth_gbs=bw_gbs,
                time_ms=time_s * 1000,
            )

        all_results[model_name] = {"xla_bw": xla_bw, "xla_tflops": xla_tflops}

    return batch_sizes, all_results


def plot_context_len_results(context_lens, all_results):
    """Plot bandwidth vs context_len for each model."""
    setup_plot_style()
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, results) in zip(axes, all_results.items()):
        plot_bandwidth_vs_param(
            ax, context_lens, {"xla": results["xla_bw"]},
            xlabel="Context length (log₂)",
            title=f"Decode Attention — {model_name}",
            log_x=True,
        )

    fig.suptitle("Decode Attention: Bandwidth Utilization vs Context Length (batch=1)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "decode_bw_vs_ctxlen")


def plot_batch_results(batch_sizes, all_results):
    """Plot bandwidth vs batch_size for each model."""
    setup_plot_style()
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, results) in zip(axes, all_results.items()):
        plot_bandwidth_vs_param(
            ax, batch_sizes, {"xla": results["xla_bw"]},
            xlabel="Batch size (log₂)",
            title=f"Decode Attention — {model_name}",
            log_x=True,
        )

    fig.suptitle("Decode Attention: Bandwidth Utilization vs Batch Size (ctx=1024)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "decode_bw_vs_batch")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    # Experiment 1: Vary context length
    context_lens, ctx_results = run_context_len_sweep()
    plot_context_len_results(context_lens, ctx_results)

    # Experiment 2: Vary batch size
    batch_sizes, batch_results = run_batch_sweep()
    plot_batch_results(batch_sizes, batch_results)

    print("\nDone! Check project_inference/plots/ for decode attention plots.")

"""
Section 2, Q2.1: Prefill Attention Benchmarking on TPU v5e.

Benchmarks single-layer prefill attention using XLA attention
(jax.nn.dot_product_attention). Measures compute utilization (TFLOPS)
across varying sequence lengths and batch sizes.

Note: seq_len is capped at 4096 (2^12). At seq_len=8192 the XLA
implementation materializes the full S×S attention matrix, causing
severe memory pressure and anomalously low measured TFLOPS.

Usage:
    python3.11 project_inference/bench_prefill.py
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
    attention_prefill_flops,
    compute_tflops,
    print_header,
    print_result,
)
from plot_utils import setup_plot_style, save_plot, plot_tflops_vs_param, COLORS
from roofline import MODELS


DTYPE = jnp.bfloat16
WARMUP_ITERS = 5
BENCH_ITERS = 20


def make_prefill_inputs(batch_size, seq_len, num_heads, head_dim, dtype=DTYPE):
    """Create random Q, K, V tensors for prefill."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=dtype)
    return q, k, v


def bench_xla_prefill(batch_size, seq_len, num_heads, head_dim):
    """Benchmark XLA dot_product_attention for prefill.

    q, k, v are passed as JIT arguments (not closed over) so XLA cannot
    constant-fold the inputs — matching real inference behavior where Q/K/V
    are dynamic tensors arriving from the previous layer.
    """
    q, k, v = make_prefill_inputs(batch_size, seq_len, num_heads, head_dim)
    scale = 1.0 / math.sqrt(head_dim)

    @jax.jit
    def run(q, k, v):
        return jax.nn.dot_product_attention(
            q, k, v, scale=scale, is_causal=True, implementation="xla"
        )

    result = benchmark_fn(lambda: run(q, k, v), warmup_iters=WARMUP_ITERS, benchmark_iters=BENCH_ITERS)
    flops = attention_prefill_flops(batch_size, num_heads, seq_len, head_dim)
    tflops = compute_tflops(flops, result['median_time_s'])
    return tflops, result['median_time_s']


def run_seq_len_sweep():
    """Experiment 1: Vary sequence length, batch_size=1.

    Capped at 4096 (2^12): at 8192 XLA materializes the full S×S attention
    matrix, causing heavy memory pressure and near-zero measured TFLOPS.
    """
    seq_lens = [2**i for i in range(7, 12)]  # 128 to 2048
    all_results = {}

    for model_name, model_cfg in MODELS.items():
        num_heads = model_cfg["q_heads"]
        head_dim = model_cfg["head_dim"]
        print_header(f"Prefill — {model_name} model, batch=1, varying seq_len")

        xla_tflops = []

        for s in seq_lens:
            print(f"\n  seq_len = {s}")
            tf_xla, t_xla = bench_xla_prefill(1, s, num_heads, head_dim)
            xla_tflops.append(tf_xla)
            print_result(f"XLA      seq_len={s}", tflops=tf_xla, time_ms=t_xla * 1000)

        all_results[model_name] = {"xla": xla_tflops}

    return seq_lens, all_results


def run_batch_sweep():
    """Experiment 2: Vary batch size, seq_len=1024."""
    batch_sizes = [2**i for i in range(0, 5)]  # 1 to 16
    seq_len = 1024
    all_results = {}

    for model_name, model_cfg in MODELS.items():
        num_heads = model_cfg["q_heads"]
        head_dim = model_cfg["head_dim"]
        print_header(f"Prefill — {model_name} model, seq_len={seq_len}, varying batch")

        xla_tflops = []

        for b in batch_sizes:
            print(f"\n  batch_size = {b}")
            tf_xla, t_xla = bench_xla_prefill(b, seq_len, num_heads, head_dim)
            xla_tflops.append(tf_xla)
            print_result(f"XLA      batch={b}", tflops=tf_xla, time_ms=t_xla * 1000)

        all_results[model_name] = {"xla": xla_tflops}

    return batch_sizes, all_results


def plot_seq_len_results(seq_lens, all_results):
    """Plot TFLOPS vs seq_len for each model config."""
    setup_plot_style()
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, results) in zip(axes, all_results.items()):
        plot_tflops_vs_param(
            ax, seq_lens, results,
            xlabel="Sequence length (log₂)",
            title=f"Prefill Attention — {model_name}",
            log_x=True,
        )

    fig.suptitle("Prefill Attention: TFLOPS vs Sequence Length (batch=1)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "prefill_tflops_vs_seqlen")


def plot_batch_results(batch_sizes, all_results):
    """Plot TFLOPS vs batch_size for each model config."""
    setup_plot_style()
    num_models = len(all_results)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, results) in zip(axes, all_results.items()):
        plot_tflops_vs_param(
            ax, batch_sizes, results,
            xlabel="Batch size (log₂)",
            title=f"Prefill Attention — {model_name}",
            log_x=True,
        )

    fig.suptitle("Prefill Attention: TFLOPS vs Batch Size (seq_len=1024)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_plot(fig, "prefill_tflops_vs_batch")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    # Experiment 1: Vary seq_len
    seq_lens, seq_results = run_seq_len_sweep()
    plot_seq_len_results(seq_lens, seq_results)

    # Experiment 2: Vary batch_size
    batch_sizes, batch_results = run_batch_sweep()
    plot_batch_results(batch_sizes, batch_results)

    print("\nDone! Check project_inference/plots/ for prefill attention plots.")

"""
Section 3: End-to-End Inference Profiling on TPU v5e.

Benchmarks the full inference pipeline (prefill + autoregressive decode)
using the nanoGPTJAX model and KV cache. Measures:
  - Prefill latency vs prompt length (Q3.1)
  - Decode latency & throughput vs decode length and batch size (Q3.2)
  - Memory footprint analysis (Q3.3)

Requires a trained checkpoint. Set the checkpoint path in the config YAML.

Usage:
    python3.11 project_inference/bench_e2e.py --config configs/small.yaml
    python3.11 project_inference/bench_e2e.py --config configs/small.yaml --profile
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanogpt"))

import time
import math
import argparse
import dataclasses
from pathlib import Path
from functools import partial
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GPT, forward_v2, count_params, precompute_frequencies
from kvcache import KVCache, prepare_chunk, count_left_padding
from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import Config, load_config_from_yaml
from utils import DP_AXIS_NAME

from jax.sharding import Mesh

from bench_utils import (
    TPU_V5E_PEAK_TFLOPS_BF16,
    TPU_V5E_HBM_BANDWIDTH_GBS,
    benchmark_fn,
    print_header,
    get_dtype_bytes,
)
from plot_utils import setup_plot_style, save_plot


# ── Inference functions (from inference.py, adapted for benchmarking) ───────

@partial(jax.jit, static_argnames=("head_dim", "pad_id"))
def prefill(params, input_ids, segment_ids, cache, head_dim, pad_id):
    left_pad_counts = count_left_padding(input_ids, pad_id=pad_id)
    uninitialized_iter = -jnp.ones_like(cache.iter)
    cache = dataclasses.replace(cache, starts=left_pad_counts, iter=uninitialized_iter)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    last_token_logits = logits[:, -1, :]
    return last_token_logits, cache


def decode_step(params, input_ids, cache, head_dim):
    segment_ids = jnp.ones_like(input_ids, dtype=jnp.int32)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    return logits[:, -1, :], cache


@partial(jax.jit, static_argnames=("head_dim", "max_new_tokens"))
def generate(params, cache, last_token, head_dim, max_new_tokens):
    """Autoregressive decode loop using jax.lax.scan."""
    generated_tokens = jnp.zeros(
        (last_token.shape[0], max_new_tokens), dtype=jnp.int32
    )
    generated_tokens = generated_tokens.at[:, 0].set(
        jnp.argmax(last_token, axis=-1).astype(jnp.int32) if last_token.ndim > 1
        else last_token.squeeze(-1)
    )

    def decode_body(carry, t):
        cache, token, gen_tokens = carry
        logits, cache = decode_step(params, token, cache, head_dim)
        next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        gen_tokens = gen_tokens.at[:, t].set(next_token)
        return (cache, next_token[:, None], gen_tokens), None

    first_token = jnp.argmax(last_token, axis=-1).astype(jnp.int32)[:, None]
    (cache, _, generated_tokens), _ = jax.lax.scan(
        decode_body,
        (cache, first_token, generated_tokens),
        jnp.arange(1, max_new_tokens),
    )
    return generated_tokens


# ── Benchmarking helpers ────────────────────────────────────────────────────

def create_dummy_input(batch_size, seq_len, vocab_size, pad_id):
    """Create a dummy left-padded input for benchmarking."""
    key = jax.random.PRNGKey(0)
    # Generate random token IDs (avoiding pad_id)
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size - 10)
    segment_ids = jnp.ones_like(tokens, dtype=jnp.int32)
    return tokens, segment_ids


def load_model(config_path):
    """Load model and config."""
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=DP_AXIS_NAME)
    cfg = load_config_from_yaml(config_path, mesh=mesh)

    print("Building model...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    model_sharding = GPT.shardings(cfg.mesh, cfg.model)
    print(f"Model parameters: {count_params(model):,}")

    # Try to load checkpoint
    ckpt_path = str(Path(cfg.ckpt_cfg.load_params_ckpt_path).resolve())
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        model = load_weights_from_checkpoint_with_validation(
            ckpt_path, model, model_sharding
        )
        print("Checkpoint loaded!")
    else:
        print(f"WARNING: Checkpoint not found at {ckpt_path}")
        print("Using randomly initialized weights (results are still valid for benchmarking).")

    return model, cfg, mesh


# ── Q3.1: Prefill Latency ──────────────────────────────────────────────────

def bench_prefill_latency(model, cfg, mesh):
    """Measure prefill latency vs prompt length."""
    prefill_lens = [2**i for i in range(7, 13)]  # 128 to 4096
    # Cap at model's max sequence length
    prefill_lens = [p for p in prefill_lens if p <= cfg.model.seqlen]

    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1  # Use last token as pad
    batch_size = len(jax.devices())  # Match device count

    print_header("Q3.1: Prefill Latency vs Prompt Length")
    results = []

    for p_len in prefill_lens:
        input_ids, segment_ids = create_dummy_input(
            batch_size, p_len, cfg.model.vocab_size, pad_id
        )
        cache = KVCache.init(jax.random.PRNGKey(1), cfg.mesh, batch_size, cfg)

        def run_prefill():
            with jax.set_mesh(cfg.mesh):
                return prefill(model, input_ids, segment_ids, cache, head_dim, pad_id=pad_id)

        timing = benchmark_fn(run_prefill, warmup_iters=3, benchmark_iters=10)
        time_ms = timing['median_time_s'] * 1000
        results.append({
            'prefill_len': p_len,
            'time_ms': time_ms,
            'tokens_per_sec': (batch_size * p_len) / timing['median_time_s'],
        })
        print(f"  prefill_len={p_len:>5}  |  time={time_ms:8.2f} ms  |  "
              f"throughput={results[-1]['tokens_per_sec']:>10,.0f} tok/s")

    return prefill_lens, results


def bench_prefill_with_profile(model, cfg, mesh, profile_lens=None, batch_size=None, kv_seqlen=None):
    """Run prefill with JAX profiler to capture trace."""
    if profile_lens is None:
        profile_lens = [128, 4096]
    profile_lens = [p for p in profile_lens if p <= cfg.model.seqlen]
    if batch_size is None:
        batch_size = len(jax.devices())

    # Allow overriding the KV cache seqlen (e.g. Q3.3 asks for seqlen=1024)
    if kv_seqlen is not None:
        import dataclasses as dc
        patched_model = dc.replace(cfg.model, seqlen=kv_seqlen)
        cfg = dc.replace(cfg, model=patched_model)

    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1
    profile_dir = str(Path("project_inference/profiles").resolve())
    os.makedirs(profile_dir, exist_ok=True)

    for p_len in profile_lens:
        print(f"\n  Profiling prefill with prompt_len={p_len}, batch_size={batch_size}, kv_seqlen={cfg.model.seqlen}...")
        input_ids, segment_ids = create_dummy_input(
            batch_size, p_len, cfg.model.vocab_size, pad_id
        )
        cache = KVCache.init(jax.random.PRNGKey(1), cfg.mesh, batch_size, cfg)

        # Warmup
        with jax.set_mesh(cfg.mesh):
            result = prefill(model, input_ids, segment_ids, cache, head_dim, pad_id=pad_id)
            jax.block_until_ready(result)

        # Profile
        trace_dir = os.path.join(profile_dir, f"prefill_len{p_len}_batch{batch_size}_kv{cfg.model.seqlen}")
        jax.profiler.start_trace(trace_dir)
        with jax.set_mesh(cfg.mesh):
            result = prefill(model, input_ids, segment_ids, cache, head_dim, pad_id=pad_id)
            jax.block_until_ready(result)
        jax.profiler.stop_trace()
        print(f"  Trace saved to: {trace_dir}")


# ── Q3.2: Decode Latency & Throughput ───────────────────────────────────────

def bench_decode_latency(model, cfg, mesh):
    """Measure decode latency vs decode length with fixed prefill."""
    decode_lens = [2**i for i in range(5, 11)]  # 32 to 1024
    prefill_len = 128
    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1
    batch_size = len(jax.devices())

    # Cap decode_lens to fit in the KV cache
    max_decode = cfg.model.seqlen - prefill_len
    decode_lens = [d for d in decode_lens if d <= max_decode]

    print_header("Q3.2a: Decode Latency vs Decode Length (prefill=128, batch=1)")
    results = []

    for d_len in decode_lens:
        input_ids, segment_ids = create_dummy_input(
            batch_size, prefill_len, cfg.model.vocab_size, pad_id
        )
        cache = KVCache.init(jax.random.PRNGKey(1), cfg.mesh, batch_size, cfg)

        # Do prefill once
        with jax.set_mesh(cfg.mesh):
            last_logits, cache = prefill(
                model, input_ids, segment_ids, cache, head_dim, pad_id=pad_id
            )
            jax.block_until_ready(last_logits)

        # Benchmark decode
        def run_decode():
            with jax.set_mesh(cfg.mesh):
                return generate(model, cache, last_logits, head_dim, max_new_tokens=d_len)

        timing = benchmark_fn(run_decode, warmup_iters=2, benchmark_iters=5)
        total_time_ms = timing['median_time_s'] * 1000
        per_token_ms = total_time_ms / d_len
        tokens_per_sec = (batch_size * d_len) / timing['median_time_s']

        results.append({
            'decode_len': d_len,
            'total_time_ms': total_time_ms,
            'per_token_ms': per_token_ms,
            'tokens_per_sec': tokens_per_sec,
        })
        print(f"  decode_len={d_len:>5}  |  total={total_time_ms:8.2f} ms  |  "
              f"per_token={per_token_ms:6.2f} ms  |  throughput={tokens_per_sec:>8,.0f} tok/s")

    return decode_lens, results


def bench_batch_throughput(model, cfg, mesh):
    """Measure throughput vs batch size with fixed prefill and decode lengths."""
    batch_sizes = [2**i for i in range(0, 6)]  # 1 to 32
    prefill_len = 128
    decode_len = 128
    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1

    # Filter batch sizes that might OOM
    batch_sizes = [b for b in batch_sizes if b <= 32]

    print_header("Q3.2b: Throughput vs Batch Size (prefill=128, decode=128)")
    results = []

    for b in batch_sizes:
        input_ids, segment_ids = create_dummy_input(
            b, prefill_len, cfg.model.vocab_size, pad_id
        )
        cache = KVCache.init(jax.random.PRNGKey(1), cfg.mesh, b, cfg)

        def run_full():
            with jax.set_mesh(cfg.mesh):
                last_logits, new_cache = prefill(
                    model, input_ids, segment_ids, cache, head_dim, pad_id=pad_id
                )
                return generate(model, new_cache, last_logits, head_dim, max_new_tokens=decode_len)

        timing = benchmark_fn(run_full, warmup_iters=2, benchmark_iters=5)
        total_tokens = b * (prefill_len + decode_len)
        throughput = total_tokens / timing['median_time_s']
        time_ms = timing['median_time_s'] * 1000

        results.append({
            'batch_size': b,
            'time_ms': time_ms,
            'throughput': throughput,
        })
        print(f"  batch={b:>3}  |  time={time_ms:8.2f} ms  |  "
              f"throughput={throughput:>10,.0f} tok/s")

    return batch_sizes, results


# ── Q3.3: Memory Footprint ─────────────────────────────────────────────────

def analyze_memory(cfg):
    """Compute theoretical memory usage."""
    print_header("Q3.3: Memory Footprint Analysis")
    model_cfg = cfg.model
    elem_bytes = get_dtype_bytes(model_cfg.dtype)

    # Model weights
    d = model_cfg.d_emb
    h = model_cfg.mlp_hidden_dim
    q_heads = model_cfg.attn.q_heads
    kv_heads = model_cfg.attn.kv_heads
    hd = model_cfg.attn.head_dim
    n_layers = model_cfg.num_layers
    vocab = model_cfg.vocab_size
    seqlen = model_cfg.seqlen

    # Per-layer weights
    qkv_params = d * (q_heads + 2 * kv_heads) * hd
    out_params = q_heads * hd * d
    mlp_params = d * h + h * d  # fc1 + fc2
    layer_params = qkv_params + out_params + mlp_params

    # Total weights
    embed_params = vocab * d
    lm_head_params = d * vocab
    total_params = embed_params + n_layers * layer_params + lm_head_params
    weight_bytes = total_params * elem_bytes

    # KV cache per token per layer: 2 * kv_heads * head_dim * elem_bytes
    kv_per_token_per_layer = 2 * kv_heads * hd * elem_bytes
    kv_per_token = kv_per_token_per_layer * n_layers

    # Q3.3 asks specifically about seqlen=1024
    kv_seqlen = 1024

    print(f"  Model config:")
    print(f"    d_emb={d}, mlp_hidden={h}, layers={n_layers}")
    print(f"    q_heads={q_heads}, kv_heads={kv_heads}, head_dim={hd}")
    print(f"    vocab_size={vocab}, dtype={model_cfg.dtype}")
    print(f"\n  Weight memory:")
    print(f"    Total params:     {total_params:>12,}")
    print(f"    Weight memory:    {weight_bytes / 1e6:>12.2f} MB")
    print(f"\n  KV cache memory (seqlen={kv_seqlen}):")
    print(f"    Per token/layer:  {kv_per_token_per_layer:>12} bytes")
    print(f"    Per token total:  {kv_per_token:>12} bytes")

    for batch_size in [1, 4, 8, 16]:
        kv_total = batch_size * kv_seqlen * kv_per_token
        print(f"    batch={batch_size:>2}, seqlen={kv_seqlen:>5}: "
              f"{kv_total / 1e6:>10.2f} MB  "
              f"({'> weights!' if kv_total > weight_bytes else 'OK'})")

    total_b1 = weight_bytes + 1 * kv_seqlen * kv_per_token
    print(f"\n  Total (weights + KV, batch=1): {total_b1 / 1e6:.2f} MB")
    print(f"  TPU v5e HBM capacity: {16 * 1024:.0f} MB")
    print(f"  Utilization (batch=1): {total_b1 / (16 * 1024 * 1e6) * 100:.1f}%")


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_prefill_latency(prefill_lens, results):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    times = [r['time_ms'] for r in results]
    throughputs = [r['tokens_per_sec'] for r in results]

    ax1.plot(prefill_lens, times, 'o-', color='#1f77b4', linewidth=2, markersize=7)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Prefill Length (log₂)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Prefill Latency")
    ax1.grid(True, alpha=0.3)

    ax2.plot(prefill_lens, throughputs, 's-', color='#ff7f0e', linewidth=2, markersize=7)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Prefill Length (log₂)")
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Prefill Throughput")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Q3.1: Prefill Performance vs Prompt Length", fontsize=15)
    fig.tight_layout()
    save_plot(fig, "e2e_prefill_latency")


def plot_decode_latency(decode_lens, results):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    total_times = [r['total_time_ms'] for r in results]
    per_token = [r['per_token_ms'] for r in results]

    ax1.plot(decode_lens, total_times, 'o-', color='#1f77b4', linewidth=2, markersize=7)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Decode Length (log₂)")
    ax1.set_ylabel("Total Decode Time (ms)")
    ax1.set_title("Total Decode Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(decode_lens, per_token, 's-', color='#ff7f0e', linewidth=2, markersize=7)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Decode Length (log₂)")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title("Per-Token Decode Latency")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Q3.2a: Decode Performance vs Decode Length", fontsize=15)
    fig.tight_layout()
    save_plot(fig, "e2e_decode_latency")


def plot_batch_throughput(batch_sizes, results):
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    times = [r['time_ms'] for r in results]
    throughputs = [r['throughput'] for r in results]

    ax1.plot(batch_sizes, times, 'o-', color='#1f77b4', linewidth=2, markersize=7)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Batch Size (log₂)")
    ax1.set_ylabel("Total Time (ms)")
    ax1.set_title("End-to-End Time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(batch_sizes, throughputs, 's-', color='#2ca02c', linewidth=2, markersize=7)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Batch Size (log₂)")
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Total Throughput")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Q3.2b: End-to-End Performance vs Batch Size", fontsize=15)
    fig.tight_layout()
    save_plot(fig, "e2e_batch_throughput")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end inference benchmarking")
    parser.add_argument("--config", type=str, default="configs/small.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--profile", action="store_true",
                        help="Enable JAX profiler traces for detailed breakdowns")
    parser.add_argument("--skip-decode", action="store_true",
                        help="Skip decode benchmarks (prefill-only)")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")

    model, cfg, mesh = load_model(args.config)

    # Q3.1: Prefill latency
    prefill_lens, prefill_results = bench_prefill_latency(model, cfg, mesh)
    plot_prefill_latency(prefill_lens, prefill_results)

    if args.profile:
        print_header("Profiling prefill with JAX profiler (Q3.1 trace viewer)")
        for p_len in [128, 4096]:
            bench_prefill_with_profile(model, cfg, mesh, profile_lens=[p_len], kv_seqlen=p_len)

    if not args.skip_decode:
        # Q3.2a: Decode latency
        decode_lens, decode_results = bench_decode_latency(model, cfg, mesh)
        plot_decode_latency(decode_lens, decode_results)

        # Q3.2b: Batch throughput
        batch_sizes, batch_results = bench_batch_throughput(model, cfg, mesh)
        plot_batch_throughput(batch_sizes, batch_results)

    # Q3.3: Memory analysis
    analyze_memory(cfg)

    if args.profile:
        print_header("Profiling memory footprint by batch size (Q3.3 memory viewer)")
        for bs in [1, 8]:
            bench_prefill_with_profile(model, cfg, mesh, profile_lens=[128], batch_size=bs, kv_seqlen=1024)

    print("\n" + "=" * 70)
    print("  All benchmarks complete!")
    print("  Plots saved to: project_inference/plots/")
    if args.profile:
        print("  Profiles saved to: project_inference/profiles/")
    print("=" * 70)

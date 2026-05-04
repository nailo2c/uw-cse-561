import argparse
import csv
import dataclasses
import os
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nanogpt"))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import load_config_from_yaml
from kvcache import KVCache, count_left_padding
from model import GPT, count_params, forward_v2
from plot_utils import save_plot, setup_plot_style
from utils import DP_AXIS_NAME

from jax.sharding import Mesh


@partial(jax.jit, static_argnames=("head_dim", "pad_id"))
def full_prefill(params, input_ids, segment_ids, cache, head_dim, pad_id):
    """One-shot prefill over the whole prompt."""
    left_pad_counts = count_left_padding(input_ids, pad_id=pad_id)
    cache = dataclasses.replace(
        cache,
        starts=left_pad_counts,
        iter=-jnp.ones_like(cache.iter),
    )
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    return logits[:, -1, :], cache


@partial(jax.jit, static_argnames=("head_dim",))
def prefill_chunk(params, input_ids, segment_ids, cache, head_dim):
    """Prefill one contiguous chunk and append its KV entries to the cache."""
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    return logits[:, -1, :], cache


def benchmark_fn(fn, warmup_iters, benchmark_iters):
    for _ in range(warmup_iters):
        result = fn()
        jax.block_until_ready(result)

    times = []
    for _ in range(benchmark_iters):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    times = np.asarray(times)
    return {
        "median_time_s": float(np.median(times)),
        "mean_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "times_s": times.tolist(),
    }


def create_dummy_input(batch_size, seq_len, vocab_size, seed):
    key = jax.random.PRNGKey(seed)
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size - 10)
    segment_ids = jnp.ones_like(tokens, dtype=jnp.int32)
    return tokens, segment_ids


def parse_token_budgets(raw, prefill_len):
    if raw:
        budgets = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        budgets = [64, 128, 256, 512, 1024, 2048, 4096]

    budgets = sorted({b for b in budgets if b > 0 and b <= prefill_len})
    if prefill_len not in budgets:
        budgets.append(prefill_len)
    return budgets


def load_model(config_path):
    device = np.array([jax.devices()[0]])
    mesh = Mesh(device, axis_names=DP_AXIS_NAME)
    cfg = load_config_from_yaml(config_path, mesh=mesh)

    print("Building model...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    model_sharding = GPT.shardings(cfg.mesh, cfg.model)
    print(f"  Parameters: {count_params(model):,}")

    ckpt_path = str(Path(cfg.ckpt_cfg.load_params_ckpt_path).resolve())
    if os.path.exists(ckpt_path):
        print(f"  Loading checkpoint: {ckpt_path}")
        model = load_weights_from_checkpoint_with_validation(
            ckpt_path, model, model_sharding
        )
        print("  Checkpoint loaded.")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt_path}.")
        print("  Using random weights. Timings are still valid for benchmarking.")

    return model, cfg, mesh


def run_full_prefill(model, cfg, input_ids, segment_ids, cache_template):
    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1
    with jax.set_mesh(cfg.mesh):
        return full_prefill(
            model, input_ids, segment_ids, cache_template, head_dim, pad_id
        )


def run_chunked_prefill(model, cfg, input_ids, segment_ids, cache_template, budget):
    head_dim = cfg.model.attn.head_dim
    cache = cache_template
    last_logits = None
    with jax.set_mesh(cfg.mesh):
        for start in range(0, input_ids.shape[1], budget):
            end = min(start + budget, input_ids.shape[1])
            chunk_ids = input_ids[:, start:end]
            chunk_segments = segment_ids[:, start:end]
            last_logits, cache = prefill_chunk(
                model, chunk_ids, chunk_segments, cache, head_dim
            )
    return last_logits, cache


def bench_chunked_prefill(
    model,
    cfg,
    prefill_len,
    batch_size,
    token_budgets,
    warmup_iters,
    benchmark_iters,
    seed,
    check_correctness,
):
    input_ids, segment_ids = create_dummy_input(
        batch_size, prefill_len, cfg.model.vocab_size, seed
    )
    cache_template = KVCache.init(jax.random.PRNGKey(seed + 1), cfg.mesh, batch_size, cfg)
    jax.block_until_ready(cache_template.iter)

    print("\n" + "=" * 70)
    print("  Section 5: Chunked Prefill")
    print("=" * 70)
    print(f"  config       : {cfg.model.d_emb=}, {cfg.model.num_layers=}")
    print(f"  prefill_len  : {prefill_len}")
    print(f"  batch_size   : {batch_size}")
    print(f"  token_budget : {token_budgets}")
    print()

    def full_fn():
        return run_full_prefill(model, cfg, input_ids, segment_ids, cache_template)

    full_timing = benchmark_fn(full_fn, warmup_iters, benchmark_iters)
    full_latency_ms = full_timing["median_time_s"] * 1000
    total_tokens = batch_size * prefill_len
    full_throughput = total_tokens / full_timing["median_time_s"]

    full_logits, _ = full_fn()
    jax.block_until_ready(full_logits)

    print(
        f"  full prefill | chunks=1   | latency={full_latency_ms:8.2f} ms "
        f"| throughput={full_throughput:10,.0f} tok/s"
    )

    results = []
    for budget in token_budgets:
        num_chunks = int(np.ceil(prefill_len / budget))

        def chunked_fn():
            return run_chunked_prefill(
                model, cfg, input_ids, segment_ids, cache_template, budget
            )

        timing = benchmark_fn(chunked_fn, warmup_iters, benchmark_iters)
        latency_ms = timing["median_time_s"] * 1000
        throughput = total_tokens / timing["median_time_s"]
        overhead = timing["median_time_s"] / full_timing["median_time_s"]

        max_abs_diff = None
        token_match = None
        if check_correctness:
            chunk_logits, _ = chunked_fn()
            jax.block_until_ready(chunk_logits)
            max_abs_diff = float(
                jnp.max(
                    jnp.abs(
                        chunk_logits.astype(jnp.float32)
                        - full_logits.astype(jnp.float32)
                    )
                )
            )
            token_match = bool(
                jnp.all(
                    jnp.argmax(chunk_logits, axis=-1)
                    == jnp.argmax(full_logits, axis=-1)
                )
            )

        row = {
            "token_budget": budget,
            "num_chunks": num_chunks,
            "latency_ms": latency_ms,
            "tokens_per_s": throughput,
            "latency_over_full": overhead,
            "max_abs_logit_diff": max_abs_diff,
            "argmax_matches_full": token_match,
        }
        results.append(row)

        correctness = ""
        if check_correctness:
            correctness = (
                f" | max_logit_diff={max_abs_diff:8.5f} "
                f"| argmax_match={token_match}"
            )
        print(
            f"  budget={budget:5d} | chunks={num_chunks:3d} "
            f"| latency={latency_ms:8.2f} ms "
            f"| throughput={throughput:10,.0f} tok/s "
            f"| vs_full={overhead:5.2f}x"
            f"{correctness}"
        )

    return {
        "full_latency_ms": full_latency_ms,
        "full_tokens_per_s": full_throughput,
        "results": results,
    }


def save_results_csv(summary, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "token_budget",
                "num_chunks",
                "latency_ms",
                "tokens_per_s",
                "latency_over_full",
                "max_abs_logit_diff",
                "argmax_matches_full",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["results"])
    print(f"  CSV saved to: {output_path}")


def plot_results(summary):
    setup_plot_style()
    budgets = [r["token_budget"] for r in summary["results"]]
    latencies = [r["latency_ms"] for r in summary["results"]]
    throughputs = [r["tokens_per_s"] for r in summary["results"]]
    overheads = [r["latency_over_full"] for r in summary["results"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(budgets, latencies, "o-", color="#1f77b4")
    axes[0].axhline(
        summary["full_latency_ms"],
        color="#d62728",
        linestyle="--",
        label="full prefill",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("Token Budget")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Prefill Latency")
    axes[0].legend()

    axes[1].plot(budgets, throughputs, "s-", color="#2ca02c")
    axes[1].axhline(
        summary["full_tokens_per_s"],
        color="#d62728",
        linestyle="--",
        label="full prefill",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Token Budget")
    axes[1].set_ylabel("Throughput (tokens/s)")
    axes[1].set_title("Token Throughput")
    axes[1].legend()

    axes[2].plot(budgets, overheads, "^-", color="#ff7f0e")
    axes[2].axhline(1.0, color="#d62728", linestyle="--", label="full prefill")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("Token Budget")
    axes[2].set_ylabel("Latency / Full Prefill")
    axes[2].set_title("Overhead vs Full Prefill")
    axes[2].legend()

    fig.suptitle("Section 5: Chunked Prefill vs Token Budget", fontsize=15)
    fig.tight_layout()
    save_plot(fig, "chunked_prefill_budget_sweep")


def main():
    parser = argparse.ArgumentParser(description="Chunked prefill benchmark")
    parser.add_argument("--config", type=str, default="configs/small.yaml")
    parser.add_argument("--prefill-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--token-budgets",
        type=str,
        default="64,128,256,512,1024,2048,4096",
        help="Comma-separated chunk sizes to sweep.",
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--benchmark-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip logit/argmax comparison against full prefill.",
    )
    args = parser.parse_args()

    print(f"JAX devices : {jax.devices()}")
    print(f"Backend     : {jax.default_backend()}")

    model, cfg, _ = load_model(args.config)

    if args.prefill_len > cfg.model.seqlen:
        raise ValueError(
            f"--prefill-len={args.prefill_len} exceeds cfg.model.seqlen={cfg.model.seqlen}"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    token_budgets = parse_token_budgets(args.token_budgets, args.prefill_len)
    summary = bench_chunked_prefill(
        model=model,
        cfg=cfg,
        prefill_len=args.prefill_len,
        batch_size=args.batch_size,
        token_budgets=token_budgets,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        seed=args.seed,
        check_correctness=not args.skip_correctness,
    )

    plot_results(summary)
    save_results_csv(
        summary,
        "project_inference/plots/chunked_prefill_budget_sweep.csv",
    )

    print("\n" + "=" * 70)
    print("  Chunked prefill benchmark complete.")
    print("  Plot: project_inference/plots/chunked_prefill_budget_sweep.png")
    print("  CSV : project_inference/plots/chunked_prefill_budget_sweep.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()

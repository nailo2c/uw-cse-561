"""
Section 4: Continuous Batching vs Static Batching Benchmark.

Complete the functions / blocks marked ``TODO`` below. The rest of the
file (JIT kernels, KV-cache scatter, warmup, plotting, checkpoint loading) is
provided.

Part 1. **Static batching** — Fixed static batches of ``batch_size`` requests; every row in a
   batch is padded to the same prompt length; decode runs for the *maximum*
   decode length in that batch. The whole batch retires together.
Part 2. **Continuous batching** — A pool of ``num_slots`` slots; each round runs a
   fixed number of decode steps for *all* slots; finished requests are replaced
   immediately from a pending queue.

For each part, replace each ``raise NotImplementedError(...)`` with your code when that part is
done.

**JAX mesh** This codebase runs inference under a global
``jax.sharding.Mesh`` stored in ``cfg.mesh``. Any call into the JIT-wrapped model
(``_prefill``, ``_decode_loop``) must run **inside** a mesh context so array
shardings and lowering match the rest of the benchmark. Use::

    with jax.set_mesh(cfg.mesh):
        ...  # _prefill / _decode_loop here

``admit_request`` (continuous batching) already wraps its ``_prefill`` in
``jax.set_mesh``; your Part 3 decode step must do the same around ``_decode_loop``.
Part 2 must wrap **both** ``_prefill`` and ``_decode_loop`` the same way.

Usage:
    python3.11 project_inference/bench_continous_batch_template.py \\
        --config configs/small.yaml \\
        --distribution bimodal \\
        --prompt-len 128 \\
        --decode-len 1024 \\
        --n-requests 16 \\
        --decode-steps 64 \\
        --batch-size 4
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
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GPT, forward_v2, count_params
from kvcache import KVCache, count_left_padding
from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import load_config_from_yaml
from utils import DP_AXIS_NAME

from jax.sharding import Mesh

from bench_utils import print_header
from plot_utils import setup_plot_style, save_plot


# ── JIT-compiled inference primitives ─────────────────────────────────────────
@partial(jax.jit, static_argnames=("head_dim", "pad_id"))
def _prefill(params, input_ids, segment_ids, cache, head_dim, pad_id):
    """Prefill a batch of sequences and return the argmax first token per row."""
    left_pad_counts = count_left_padding(input_ids, pad_id=pad_id)
    cache = dataclasses.replace(
        cache,
        starts=left_pad_counts,
        iter=-jnp.ones_like(cache.iter),
    )
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    first_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
    return first_token[:, None], cache   # (B, 1), KVCache

@partial(jax.jit, static_argnames=("head_dim", "decode_steps"))
def _decode_loop(params, last_token, cache, head_dim, decode_steps):
    """Run `decode_steps` autoregressive steps with jax.lax.scan."""
    def body(carry, _):
        cache, token = carry
        segment_ids = jnp.ones_like(token, dtype=jnp.int32)
        logits, cache = forward_v2(params, token, segment_ids, cache, head_dim)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
        return (cache, next_token[:, None]), next_token

    (cache, _), generated = jax.lax.scan(
        body,
        (cache, last_token),
        None,
        length=decode_steps,
    )
    return generated, cache   # (B, decode_steps), KVCache


# ── Request generation ─────────────────────────────────────────────────────────
def _pad_to_pow2(tokens: jnp.ndarray, pad_id: int) -> tuple:
    """Left-pad a (1, T) token array to the next power-of-2 length."""
    t = tokens.shape[1]
    target = 1 if t == 0 else 2 ** math.ceil(math.log2(max(t, 1)))
    pad_w = target - t
    padded = jnp.pad(tokens, [(0, 0), (pad_w, 0)], constant_values=pad_id)
    seg = jnp.where(padded != pad_id, 1, 0).astype(jnp.int32)
    return padded, seg


def generate_requests(
    n_requests,
    cfg,
    distribution="bimodal",
    decode_len=64,
    prompt_len=128,
    seed=42,
):
    """
    Synthesise dummy requests with a **fixed** prompt length (``prompt_len``,
    after clipping to the model context). Only **decode** lengths vary.

    distribution (decode-length pattern)
        "fixed"   — every request decodes exactly ``decode_len`` (capped so
                    ``prompt_len + decode + margin`` fits ``cfg.model.seqlen``).
        "uniform" — decode lengths i.i.d. uniform in an integer range, upper
                    bound ``decode_len`` (again capped to the context window).
        "bimodal" — half short-decode, half long-decode (two pools), shuffled;
                    highlights static batching paying for the max decode in each
                    batch while continuous batching can retire early.
    """
    rng = np.random.RandomState(seed)
    pad_id = cfg.model.vocab_size - 1

    prompt_len = int(np.clip(prompt_len, 16, cfg.model.seqlen - 8 - 4))
    max_decode_allowed = cfg.model.seqlen - prompt_len - 4
    decode_cap = min(int(decode_len), max_decode_allowed)
    decode_cap = max(decode_cap, 1)

    if distribution == "fixed":
        decode_lens = np.full(n_requests, decode_cap, dtype=int)
    elif distribution == "uniform":
        lo = max(8, decode_cap // 8)
        hi = decode_cap
        if lo > hi:
            lo = hi
        decode_lens = rng.randint(lo, hi + 1, size=n_requests).astype(int)
    elif distribution == "bimodal":
        half = n_requests // 2
        short_opts = [
            d
            for d in [8, 16, 32, 64, 128, 256, 512]
            if d <= max(8, decode_cap // 4)
        ]
        long_opts = sorted(
            {d for d in [32, 64, 128, 256, 512, 1024] if decode_cap // 2 <= d <= decode_cap}
            | {decode_cap}
        )
        if not short_opts:
            short_opts = [max(8, decode_cap // 8)]
        if not long_opts:
            long_opts = [decode_cap]
        short_decode = rng.choice(short_opts, size=half)
        long_decode = rng.choice(long_opts, size=n_requests - half)
        decode_lens = np.concatenate([short_decode, long_decode])
        idx = np.arange(n_requests)
        rng.shuffle(idx)
        decode_lens = decode_lens[idx].astype(int)
    else:
        raise ValueError(
            f"Unknown distribution {distribution!r}; "
            "expected 'fixed', 'uniform', or 'bimodal'"
        )

    decode_lens = np.clip(decode_lens, 1, max_decode_allowed).astype(int)

    jax_key = jax.random.PRNGKey(seed)
    requests = []
    for i in range(n_requests):
        dlen = int(decode_lens[i])
        jax_key, sk = jax.random.split(jax_key)
        tokens = jax.random.randint(
            sk, (1, prompt_len), 0, cfg.model.vocab_size - 10
        )
        padded, seg = _pad_to_pow2(tokens, pad_id)
        requests.append({
            "id": i,
            "input_ids": padded,
            "segment_ids": seg,
            "prompt_len": prompt_len,
            "padded_len": padded.shape[1],
            "decode_len": dlen,
        })
    return requests


# ── Static batching ────────────────────────────────────────────────────────────
def _make_batch(requests, pad_id):
    """
    Left-pad a list of requests so every row has the same length
    (max padded_len in this mini-batch), then stack into a single array.
    """
    max_len = max(r["padded_len"] for r in requests)
    # Round max_len up to next power-of-2 for a stable JIT key.
    target = 2 ** math.ceil(math.log2(max(max_len, 1)))
    ids_list, seg_list = [], []
    for r in requests:
        extra = target - r["padded_len"]
        inp = jnp.pad(r["input_ids"], [(0, 0), (extra, 0)], constant_values=pad_id)
        seg = jnp.where(inp != pad_id, 1, 0).astype(jnp.int32)
        ids_list.append(inp)
        seg_list.append(seg)
    return jnp.concatenate(ids_list, axis=0), jnp.concatenate(seg_list, axis=0)


def run_static_batching(model, cfg, requests, batch_size=4, verbose=True):
    """
    Static batching: pack `batch_size` requests per batch, pad the whole batch
    to the longest prompt, decode for the longest decode length in the batch.
    Retire the entire batch together.
    """
    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1
    n = len(requests)
    batches = range(math.ceil(n / batch_size))

    latencies = [] # {request_id, latency_s, prompt_len}
    total_tok = 0
    gen_tokens = {} # {request_id: np.ndarray of shape (decode_len,)}

    # ── per-stage timing accumulators ───────────────────────────────────────
    t_prefill = 0.0
    t_decode = 0.0

    # Pre-allocate one cache template outside the timed section so KVCache.init
    # is not charged to static batching.  JAX is purely functional: _prefill
    # returns a NEW cache with updated k/v/iter; the template is never mutated
    # and can be reused for every static batch — exactly the same trick as CB's
    # single_cache_template.
    static_cache_template = KVCache.init(
        jax.random.PRNGKey(0), cfg.mesh, batch_size, cfg
    )
    jax.block_until_ready(static_cache_template.iter)

    wall_start = time.perf_counter()

    for b in batches:
        ### 1. TODO ###
        # select the next batch of requests from requests. Ensure that the 
        # the b-th batch has batch_size requests. If the last batch is short,
        # replicate the last request until len(batch) == batch_size.
        batch = None # TODO: Complete this 

        batch_admit = time.perf_counter()
        ### 2. TODO ###
        # determine the number of decode steps required to complete the longest 
        # request in the batch 
        decode_steps = None # TODO: Complete this 

        ### 3. TODO ###
        # Create a batch by paddign requests to the same length.
        # Use _make_batch to create the batch.
        input_ids, segment_ids = None # TODO: Complete this 

        ### 4. TODO ###
        # Run prefill and decode on the batch 

        with jax.set_mesh(cfg.mesh):
            t0 = time.perf_counter()

            ## 5. TODO ###
            # Run prefill on the batch 
            last_token, cache = None # TODO: Complete this to run prefill 

            jax.block_until_ready(last_token)
            t_prefill += time.perf_counter() - t0

            t0 = time.perf_counter()
            ## 6. TODO ###
            # Run decode on the batch 
            generated, _ = None # TODO: Complete this to run decode 

            jax.block_until_ready(generated)
            t_decode += time.perf_counter() - t0

        raise NotImplementedError("Implement the continuous batching while-loop (Part 2).")

        # Collect per-request token sequences (trim to each request's own decode_len).
        gen_np = np.array(generated)
        for req_idx, req in enumerate(requests[b * batch_size : (b + 1) * batch_size]):
            gen_tokens[req["id"]] = gen_np[: req["decode_len"], req_idx]

        batch_done = time.perf_counter()
        batch_lat = batch_done - batch_admit
        latencies.append({
            "request_id": requests[b * batch_size : (b + 1) * batch_id],
            "latency_s": batch_done - wall_start,
            "prompt_len": max(r["prompt_len"] for r in requests[b * batch_size : (b + 1) * batch_size]),
        })

        if verbose:
            print(f"  [static  BATCH] "
                  f"batch={b:02d}  "
                  f"prefill={t_prefill*1000:.1f}ms  "
                  f"decode={t_decode*1000:.1f}ms  "
                  f"lat={batch_lat*1000:.1f}ms  ")

        for req in requests[b * batch_size : (b + 1) * batch_size]:
            latencies.append({
                "request_id": req["id"],
                "latency_s": batch_done - wall_start,
                "prompt_len": req["prompt_len"],
                "decode_len": req["decode_len"],
            })

        total_tok += batch_size * (decode_steps + max(r["padded_len"] for r in batch))

    wall_end = time.perf_counter()
    total_time = wall_end - wall_start
    if verbose:
        print(f"  [static  TOTAL] "
              f"prefill={t_prefill*1000:.1f}ms  "
              f"decode={t_decode*1000:.1f}ms  "
              f"wall={total_time*1000:.1f}ms\n")
    return {
        "strategy":         "Static Batching",
        "total_time_s":     total_time,
        "throughput_req_s": n / total_time,
        "throughput_tok_s": total_tok / total_time,
        "latencies":        latencies,
        "gen_tokens":       gen_tokens,
    }


# ── Continuous batching ────────────────────────────────────────────────────────
@partial(jax.jit, static_argnums=(2,))
def _insert_into_kvcache_pool(shared_cache, single_cache, slot_idx):
    """
    Write a freshly-prefilled single-slot cache into row
    `slot_idx` of the shared KVCache pool.

    This resets that slot's KV entries, starts, and iter in-place, bootsraps
    the new requests KVCache slot while all other slots are untouched.

    JIT-compiled with slot_idx as a static arg (one compiled version per slot
    index).  This is important for two reasons:
      1. Preserves array shardings — eager at[].set() strips sharding
         annotations, which would force _decode_loop to recompile every round.
      2. Collapses 2*num_layers+2 individual eager dispatches into a single XLA
         program, eliminating ~80 ms of per-admit Python dispatch overhead.
    """
    new_k = [
        shared_cache.k[l].at[slot_idx].set(single_cache.k[l][0])
        for l in range(len(shared_cache.k))
    ]
    new_v = [
        shared_cache.v[l].at[slot_idx].set(single_cache.v[l][0])
        for l in range(len(shared_cache.v))
    ]
    new_starts = shared_cache.starts.at[slot_idx].set(single_cache.starts[0])
    new_iter   = shared_cache.iter.at[slot_idx].set(single_cache.iter[0])
    return dataclasses.replace(
        shared_cache, k=new_k, v=new_v, starts=new_starts, iter=new_iter
    )


def run_continuous_batching(model, cfg, requests, num_slots=4, num_decode_steps=32,
                            verbose=True):
    """
    TODO (Part 2) — Complete the main scheduling loop below.

    Read ``admit_request`` and the slot arrays: they track which slots are busy,
    how many tokens each slot has generated, per-request decode budget, etc.
    """
    head_dim = cfg.model.attn.head_dim
    pad_id = cfg.model.vocab_size - 1
    pending = deque(requests)
    latencies = []
    total_tokens = 0
    jax_key = jax.random.PRNGKey(1000)

    # Pre-allocate ONE single-request cache template and reuse it for every
    # admission.  Calling KVCache.init inside admit_request costs ~70 ms/call
    # (26 separate kernel launches to zero-fill k/v arrays).  Instead we reset
    # only iter and starts (two scalar ops) before each prefill; stale k/v
    # values beyond the new prompt length are masked out by the attention mask.
    single_cache_template = KVCache.init(jax.random.PRNGKey(42), cfg.mesh, 1, cfg)
    # One shared cache for all slots — iter is (num_slots,) after our KVCache change.
    shared_cache = KVCache.init(jax.random.PRNGKey(999), cfg.mesh, num_slots, cfg)

    # Sharded zero token used for inactive slots in batch_last_tokens.
    # Must have the same NamedSharding as _prefill output (P(None, None)) so
    # jit(concatenate) always sees a uniform sharding signature and hits the
    # warmup cache.  jnp.zeros((1,1)) has UnspecifiedValue sharding, which
    # causes jit(concatenate) to cold-compile on every new sharding pattern.
    inactive_tok = jax.device_put(
        jnp.zeros((1, 1), dtype=jnp.int32),
        jax.sharding.NamedSharding(cfg.mesh, jax.sharding.PartitionSpec(None, None)),
    )

    slot_active = [False] * num_slots
    slot_last_token = [inactive_tok] * num_slots   # sharded, not plain zeros
    slot_gen_count = [0] * num_slots
    slot_max_gen = [0] * num_slots
    slot_admit_time = [0.0] * num_slots
    slot_decode_start = [0.0] * num_slots
    slot_request_id = [-1] * num_slots
    slot_prompt_len = [0] * num_slots

    # ── per-stage timing accumulators ───────────────────────────────────────
    t_prefill = 0.0
    t_scatter = 0.0
    t_decode = 0.0
    n_admits = 0
    n_rounds = 0

    # ── token log for correctness checking ──────────────────────────────────
    # Accumulates every token emitted by each slot across all decode rounds.
    # Works for any num_decode_steps: extend by all `num_decode_steps` tokens
    # each round, then trim to the request's exact decode_len at retirement.
    slot_token_log = [[] for _ in range(num_slots)]   # list-of-lists, one per slot
    cb_gen_tokens  = {}                               # {request_id: np.ndarray}

    # ── Helper: prefill one request into a slot ─────────────────────────────
    def admit_request(slot_idx, req):
        nonlocal shared_cache, jax_key, t_prefill, t_scatter, n_admits
        jax_key, _ = jax.random.split(jax_key)
        t0 = time.perf_counter()
        with jax.set_mesh(cfg.mesh):
            last_tok, single_cache = _prefill(
                model, req["input_ids"], req["segment_ids"],
                single_cache_template, head_dim, pad_id=pad_id,
            )
        jax.block_until_ready(last_tok)
        t_prefill += time.perf_counter() - t0

        # Insert the freshly-computed KV entries into the shared cache.
        t0 = time.perf_counter()
        shared_cache = _insert_into_kvcache_pool(shared_cache, single_cache, slot_idx)
        jax.block_until_ready(shared_cache.iter)
        t_scatter += time.perf_counter() - t0

        n_admits += 1
        slot_active[slot_idx] = True
        slot_last_token[slot_idx] = last_tok
        slot_gen_count[slot_idx] = 0
        slot_max_gen[slot_idx] = req["decode_len"]
        slot_admit_time[slot_idx] = time.perf_counter()
        slot_decode_start[slot_idx] = time.perf_counter()
        slot_request_id[slot_idx] = req["id"]
        slot_prompt_len[slot_idx] = req["prompt_len"]

    wall_start = time.perf_counter()

    # ── Fill initial slots ──────────────────────────────────────────────────
    for s in range(min(num_slots, len(pending))):
        admit_request(s, pending.popleft())

    # ── Main decode loop ────────────────────────────────────────────────────
    # Each iteration runs exactly `num_decode_steps` steps for all active slots
    # in a single _decode_loop call.  Slots that hit their decode_len budget
    # during a round are retired and a new request is admitted immediately.
    # The user controls the scheduling granularity: small num_decode_steps
    # finer retirement (lower latency, more Python overhead); large
    # coarser scheduling (less overhead, more wasted compute on finished slots).
    last_generated = jnp.zeros((num_slots, 1), dtype=jnp.int32)
    jax.config.update("jax_log_compiles", True)   # prints every new compile to stderr
    round_num  = 0
    t_wall_ref = time.perf_counter()   # snapshot at loop entry for cumulative prints

    while any(slot_active):
        ### 1. TODO ###
        # Build the batch input consisting of the last tokens of all 
        # active slots (requests).
        batch_last_tokens = None # Complete this 

        ### 2. TODO ###
        # Run decode on the batch 
        with jax.set_mesh(cfg.mesh):
            generated, shared_cache = None # Complete this to run decode 

        last_generated = generated
        gen_np = np.array(generated)

        ### 3. TODO ###
        # Update the slot_token_log for each active slot.
        for s in range(num_slots):
            if not slot_active[s]:
                continue
            
            slot_gen_count[s] = None # TODO: Complete this 
            slot_last_token[s] = None # TODO: Complete this 
            total_tokens = None # TODO: Complete this 
            slot_token_log[s] = None # TODO: Complete this 

            # determine whether request has finished 
            request_finished = None # TODO: Complete this 
            if request_finished:
                t_now = time.perf_counter()
                t_decode += t_now - slot_decode_start[s]
                latencies.append({
                    "request_id": slot_request_id[s],
                    "latency_s": t_now - wall_start,
                    "prompt_len": slot_prompt_len[s],
                    "decode_len": slot_max_gen[s],
                })

                cb_gen_tokens[slot_request_id[s]] = np.array(
                    slot_token_log[s][:slot_max_gen[s]]
                )

                slot_token_log[s] = None # TODO: Complete this 
                slot_active[s] = None # TODO: Complete this 
                n_rounds = None # TODO: Complete this 

                # Since we have retired a request from this slot
                # Admit a new request from the pending queue
                if pending:
                    pass # TODO: Complete this to admit a new request from the pending queue

        # Cumulative progress line 
        elapsed_ms = (time.perf_counter() - t_wall_ref) * 1000
        n_active = sum(slot_active)
        if verbose:
            print(f"  [cb      ROUND] round_num={round_num:02d} "
                  f"elapsed={elapsed_ms:.1f}ms  "
                  f"num_decode_steps={num_decode_steps:4d}  "
                  f"active={n_active}/{num_slots}  "
                  f"retired={n_rounds}  "
                  f"pending={len(pending)}")
        round_num += 1

    raise NotImplementedError("Implement the continuous batching while-loop (Part 2).")

    # Single sync — wait for all device work before recording end time.
    jax.config.update("jax_log_compiles", False)  # stop logging after CB
    jax.block_until_ready(last_generated)
    wall_end = time.perf_counter()
    n = len(requests)
    total_time = wall_end - wall_start
    dlens = [r["decode_len"] for r in requests]
    if verbose:
        print(f"  [cb      TOTAL] "
              f"admits={n_admits}  retirements={n_rounds}  "
              f"prefill={t_prefill*1000:.0f}ms  "
              f"scatter={t_scatter*1000:.0f}ms  "
              f"wall={total_time*1000:.0f}ms\n")
    return {
        "strategy":         "Continuous Batching",
        "total_time_s":     total_time,
        "throughput_req_s": n / total_time,
        "throughput_tok_s": total_tokens / total_time,
        "latencies":        latencies,
        "gen_tokens":       cb_gen_tokens,
    }


# ── Correctness verification ───────────────────────────────────────────────────
def verify_correctness(static_r, cb_r):
    """
    Compare per-request token sequences from the given static and CB result dicts.

    The caller must ensure both results were produced with the same effective
    batch size for prefill (batch_size=1 for static, any num_slots for CB).
    CB's prefill always runs batch=1 (single_cache_template).  Static's prefill
    uses batch=batch_size, so pass static_r from a batch_size=1 run to guarantee
    identical XLA kernel reduction order and exact token equality.

    If static_r comes from batch_size > 1, matrix-multiply reduction orders can
    differ between batch-1 and batch-N, occasionally flipping the argmax of
    near-tied logits.  That is a floating-point precision artifact, not a CB bug.

    Any token mismatch when batch_size=1 is a genuine logic error (wrong KV-cache
    scatter, wrong iter bookkeeping, wrong slot indexing, etc.).
    """
    static_tokens = static_r["gen_tokens"]
    cb_tokens = cb_r["gen_tokens"]

    req_ids = sorted(static_tokens.keys())
    failed = []

    print(f"\n  Correctness check ({len(req_ids)} requests) ──────────────────────────")
    for req_id in req_ids:
        ref = static_tokens[req_id]
        got = cb_tokens.get(req_id)

        if got is None:
            print(f"    req {req_id:3d}: MISSING in CB output")
            failed.append(req_id)
            continue

        if len(ref) != len(got):
            print(f"    req {req_id:3d}: LENGTH mismatch  "
                  f"static={len(ref)}  cb={len(got)}")
            failed.append(req_id)
            continue

        if np.array_equal(ref, got):
            print(f"    req {req_id:3d}: PASS  (decode_len={len(ref)})")
        else:
            diff_pos = int(np.argmax(ref != got))
            lo, hi   = max(0, diff_pos - 2), diff_pos + 5
            print(f"    req {req_id:3d}: FAIL  first mismatch at step {diff_pos}  "
                  f"(decode_len={len(ref)})")
            print(f"      static[{lo}:{hi}] = {ref[lo:hi].tolist()}")
            print(f"      cb    [{lo}:{hi}] = {got[lo:hi].tolist()}")
            failed.append(req_id)

    if not failed:
        print(f"\n  ALL {len(req_ids)} requests match token-for-token.\n")
    else:
        print(f"\n {len(failed)} / {len(req_ids)} request(s) FAILED.\n")
        raise AssertionError(
            f"Correctness check failed for {len(failed)} request(s): {failed}"
        )


# ── Warmup ─────────────────────────────────────────────────────────────────────
def warmup_jit(model, cfg, requests, batch_size=4, num_slots=4, num_decode_steps=32):
    """
    Trigger JIT compilation for all prompt lengths and batch sizes that appear
    in the benchmark before the timed runs.
    """
    head_dim = cfg.model.attn.head_dim
    pad_id   = cfg.model.vocab_size - 1

    # ── _make_batch: warm up its eager JAX ops (jnp.pad / jnp.where /
    # jnp.concatenate) for every unique batch target length.  Each unique
    # (input_plen, target_plen) pair triggers a separate XLA kernel compile
    # (~80-90 ms).  Without this, the first static batch per unique shape pays the
    # compile cost inside the timed benchmark, inflating static-batching times.
    unique_lens = sorted({r["padded_len"] for r in requests})
    print(f"  [Warmup] _make_batch eager ops for lengths: {unique_lens}")
    n = len(requests)
    batches = range(math.ceil(n / batch_size))
    for b in batches:
        batch = requests[b * batch_size : (b + 1) * batch_size]
        while len(batch) < batch_size:
            batch = batch + [batch[-1]]
        ids_w, segs_w = _make_batch(batch, pad_id)
        jax.block_until_ready(ids_w)
    print(f"  [Warmup] Prefill kernels (batch=1) for lengths: {unique_lens}")
    for plen in unique_lens:
        ids  = jnp.ones((1, plen), dtype=jnp.int32)
        segs = jnp.ones_like(ids)
        cache_w = KVCache.init(jax.random.PRNGKey(77), cfg.mesh, 1, cfg)
        with jax.set_mesh(cfg.mesh):
            tok, c = _prefill(model, ids, segs, cache_w, head_dim, pad_id=pad_id)
            jax.block_until_ready(tok)


    decode_steps   = max(r["decode_len"] for r in requests)  # used below for static warmup
    max_plen       = max(r["padded_len"] for r in requests)
    target_plen    = 2 ** math.ceil(math.log2(max(max_plen, 1)))
    cache_template = KVCache.init(jax.random.PRNGKey(81), cfg.mesh, 1, cfg)
    cache_slots    = KVCache.init(jax.random.PRNGKey(80), cfg.mesh, num_slots, cfg)

    print(f"  [Warmup] Insert into KVCache pool + decode (all {len(unique_lens)} prompt lengths × "
          f"{num_slots} slots)")
    last_tok_warmup = None
    for plen in unique_lens:          # covers every unique padded length
        ids_w  = jnp.ones((1, plen), dtype=jnp.int32)
        segs_w = jnp.ones_like(ids_w)
        with jax.set_mesh(cfg.mesh):
            tok_w, sc = _prefill(model, ids_w, segs_w, cache_template, head_dim, pad_id=pad_id)
        if last_tok_warmup is None:
            last_tok_warmup = tok_w   # (1,1) with NamedSharding — same type as slot_last_token[s]
        for s in range(num_slots):    # every slot index for this prefill sharding
            cache_slots = _insert_into_kvcache_pool(cache_slots, sc, s)
    jax.block_until_ready(cache_slots.iter)

    # CB decode: pre-compile _decode_loop for exactly num_decode_steps.
    # With a fixed step size there is only one JIT trace needed (one static arg
    # value), so warmup is a single call rather than one per power of 2.
    dummy_toks = jnp.concatenate([last_tok_warmup] * num_slots, axis=0)
    print(f"  [Warmup] CB decode loop (batch={num_slots}, num_decode_steps={num_decode_steps})")
    with jax.set_mesh(cfg.mesh):
        gen_cb, _ = _decode_loop(model, dummy_toks, cache_slots, head_dim,
                                 decode_steps=num_decode_steps)
    # Pre-compile the eager dynamic_slice triggered by generated[-1:, s:s+1].
    for s in range(num_slots):
        _ = gen_cb[-1:, s:s+1]
    jax.block_until_ready(gen_cb)

    # Pre-compile jit(concatenate) for every sharding pattern that can appear
    # in the CB decode loop.  Active slots have NamedSharding (from _prefill or
    # generated[-1:,s:s+1]); inactive slots use inactive_tok (also NamedSharding
    # via device_put).  Warm up all-active and mixed-active patterns so the
    # benchmark never sees an unregistered signature.
    inactive_tok_w = jax.device_put(
        jnp.zeros((1, 1), dtype=jnp.int32),
        jax.sharding.NamedSharding(cfg.mesh, jax.sharding.PartitionSpec(None, None)),
    )
    for n_active in range(1, num_slots + 1):   # 1 active, 2 active, ..., all active
        toks = ([last_tok_warmup] * n_active +
                [inactive_tok_w] * (num_slots - n_active))
        _ = jnp.concatenate(toks, axis=0)
    jax.block_until_ready(_)

    # ── Static batching: prefill + decode for every unique padded length AND
    # every unique batch-max decode_len.  With variable decode lengths, different
    # static batches may need different numbers of decode steps; each unique value of
    # max(r["decode_len"] for r in batch) triggers a separate _decode_loop compile.
    unique_group_decode_lens = set()
    for b in batches:
        batch = requests[b * batch_size : (b + 1) * batch_size]
        while len(batch) < batch_size:
            batch = batch + [batch[-1]]
        unique_group_decode_lens.add(max(r["decode_len"] for r in batch))
    unique_group_decode_lens = sorted(unique_group_decode_lens)
    print(f"  [Warmup] Static kernels (batch={batch_size}) "
          f"for prompt_lens={unique_lens}, decode_lens={unique_group_decode_lens}")
    static_cache_tmpl = KVCache.init(jax.random.PRNGKey(78), cfg.mesh, batch_size, cfg)
    for plen in unique_lens:
        ids_b  = jnp.ones((batch_size, plen), dtype=jnp.int32)
        segs_b = jnp.ones_like(ids_b)
        with jax.set_mesh(cfg.mesh):
            tok_b, c_b = _prefill(model, ids_b, segs_b, static_cache_tmpl,
                                  head_dim, pad_id=pad_id)
        for dlen in unique_group_decode_lens:
            with jax.set_mesh(cfg.mesh):
                gen_b, _ = _decode_loop(model, tok_b, c_b, head_dim,
                                        decode_steps=dlen)
                jax.block_until_ready(gen_b)

    print("  [Warmup] Done.\n")


# ── Benchmark driver ───────────────────────────────────────────────────────────
def bench_all_distributions(
    model,
    cfg,
    n_requests=16,
    decode_len=64,
    batch_size=4,
    num_slots=4,
    prompt_len=128,
):
    """
    Run static vs continuous batching for each decode-length distribution
    (``fixed`` / ``uniform`` / ``bimodal``) and return results by name.
    """
    distributions = ["fixed", "uniform", "bimodal"]
    all_results   = {}

    for dist in distributions:
        print_header(
            f"Continuous vs Static Batching  [dist={dist}, N={n_requests}]"
        )
        requests = generate_requests(
            n_requests,
            cfg,
            distribution=dist,
            decode_len=decode_len,
            prompt_len=prompt_len,
        )
        plens = [r["prompt_len"] for r in requests]
        dlens = [r["decode_len"] for r in requests]
        print(
            f"  Prompt lengths — min={min(plens):4d}, max={max(plens):4d}, "
            f"mean={np.mean(plens):5.1f}, std={np.std(plens):5.1f}"
        )
        print(
            f"  Decode lengths — min={min(dlens):4d}, max={max(dlens):4d}, "
            f"mean={np.mean(dlens):5.1f}, std={np.std(dlens):5.1f}"
        )
        print(f"  batch_size = {batch_size}  (num_slots for CB, batch for static)\n")

        warmup_jit(model, cfg, requests, batch_size=batch_size, num_slots=num_slots)

        # Static batching
        print("  [Static]     running …")
        static_r = run_static_batching(model, cfg, requests, batch_size=batch_size)
        print(
            f"  [Static]     total={static_r['total_time_s']:.2f}s  |  "
            f"{static_r['throughput_req_s']:.2f} req/s  |  "
            f"mean_lat={np.mean([l['latency_s'] for l in static_r['latencies']])*1000:.0f} ms"
        )

        # Continuous batching
        print("  [Continuous] running …")
        cb_r = run_continuous_batching(model, cfg, requests, num_slots=num_slots)
        print(
            f"  [Continuous] total={cb_r['total_time_s']:.2f}s  |  "
            f"{cb_r['throughput_req_s']:.2f} req/s  |  "
            f"mean_lat={np.mean([l['latency_s'] for l in cb_r['latencies']])*1000:.0f} ms"
        )

        speedup = static_r["total_time_s"] / cb_r["total_time_s"]
        print(f"\n  Speedup (static / continuous) wall time: {speedup:.2f}x")

        all_results[dist] = {
            "static":     static_r,
            "continuous": cb_r,
            "requests":   requests,
        }

    return all_results


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_comparison(all_results, n_requests):
    """
    2-column figure per distribution:
      col 1 — throughput bar (req/s)
      col 2 — latency CDF
    """
    distributions = list(all_results.keys())
    n_dists = len(distributions)
    setup_plot_style()
    fig, axes = plt.subplots(n_dists, 2, figsize=(12, 5 * n_dists))
    if n_dists == 1:
        axes = axes[np.newaxis, :]

    COLORS = {"Static Batching": "#1f77b4", "Continuous Batching": "#2ca02c"}

    for row, dist in enumerate(distributions):
        res      = all_results[dist]
        static_r = res["static"]
        cb_r     = res["continuous"]

        # ── Col 0: throughput bar ──────────────────────────────────────────
        ax = axes[row, 0]
        labels = ["Static", "Continuous"]
        vals   = [static_r["throughput_req_s"], cb_r["throughput_req_s"]]
        cols   = [COLORS["Static Batching"], COLORS["Continuous Batching"]]
        bars   = ax.bar(labels, vals, color=cols, width=0.5,
                        edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01 * max(vals),
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_ylabel("Throughput (req/s)")
        ax.set_title(f"[{dist}] Throughput")
        ax.set_ylim(0, max(vals) * 1.3)

        # ── Col 1: latency CDF ─────────────────────────────────────────────
        ax = axes[row, 1]
        for strategy, r in [("Static Batching", static_r),
                             ("Continuous Batching", cb_r)]:
            lats_ms = sorted(l["latency_s"] * 1000 for l in r["latencies"])
            n_l     = len(lats_ms)
            ax.plot(
                lats_ms, np.arange(1, n_l + 1) / n_l,
                "o-", color=COLORS[strategy], label=strategy,
                linewidth=2, markersize=4,
            )
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(f"[{dist}] Latency CDF")
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Continuous vs Static Batching  (N={n_requests} requests)",
        fontsize=15,
    )
    fig.tight_layout()
    save_plot(fig, "continuous_batching_benchmark")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(config_path):
    # Single device — no data parallelism needed for inference.
    device = np.array([jax.devices()[0]])
    mesh   = Mesh(device, axis_names=DP_AXIS_NAME)
    cfg    = load_config_from_yaml(config_path, mesh=mesh)

    print("Building model …")
    model         = GPT.init(jax.random.PRNGKey(0), cfg)
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
        print("  Using random weights (timings are valid; outputs are noise).")

    return model, cfg, mesh


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Continuous vs static batching benchmark"
    )
    parser.add_argument("--config", type=str, default="configs/small.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--n-requests", type=int, default=16,
                        help="Total number of requests to process")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Token length of every prompt (same for all requests and distributions)",
    )
    parser.add_argument(
        "--decode-len",
        type=int,
        default=64,
        help="For 'fixed' distribution: decode steps per request. "
        "For 'uniform' / 'bimodal' distribution: upper bound / scale for sampled decode lengths.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="all",
        choices=["fixed", "uniform", "bimodal", "all"],
        help="Decode-length pattern: fixed, uniform, bimodal, or 'all' to sweep",
    )
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size: number of concurrent slots for CB "
                             "and requests per group for static batching")
    parser.add_argument("--decode-steps", type=int, default=32,
                        help="Fixed number of decode steps per CB scheduling round. "
                             "Small = finer-grained retirement (lower latency, more overhead); "
                             "Large = coarser scheduling (less overhead, more wasted compute).")
    args = parser.parse_args()

    print(f"JAX devices : {jax.devices()}")
    print(f"Backend     : {jax.default_backend()}")
    print()

    model, cfg, mesh = load_model(args.config)

    if args.distribution == "all":
        all_results = bench_all_distributions(
            model,
            cfg,
            n_requests=args.n_requests,
            decode_len=args.decode_len,
            batch_size=args.batch_size,
            num_slots=args.batch_size,
            prompt_len=args.prompt_len,
        )
        plot_comparison(all_results, args.n_requests)
    else:
        requests = generate_requests(
            args.n_requests,
            cfg,
            distribution=args.distribution,
            decode_len=args.decode_len,
            prompt_len=args.prompt_len,
        )
        warmup_jit(model, cfg, requests, batch_size=args.batch_size,
                   num_slots=args.batch_size, num_decode_steps=args.decode_steps)

        print_header(
            f"Continuous vs Static  [dist={args.distribution}, "
            f"N={args.n_requests}]"
        )
        plens = [r["prompt_len"] for r in requests]
        dlens = [r["decode_len"] for r in requests]
        print(
            f"  Prompt lengths — min={min(plens):4d}, max={max(plens):4d}, "
            f"mean={np.mean(plens):5.1f}, std={np.std(plens):5.1f}"
        )
        print(
            f"  Decode lengths — min={min(dlens):4d}, max={max(dlens):4d}, "
            f"mean={np.mean(dlens):5.1f}, std={np.std(dlens):5.1f}"
        )
        print(f"  batch_size = {args.batch_size}  "
              f"(num_slots for CB, batch for static)  "
              f"num_decode_steps (CB) = {args.decode_steps}\n")
        static_r = run_static_batching(model, cfg, requests,
                                       batch_size=args.batch_size)
        cb_r     = run_continuous_batching(model, cfg, requests,
                                           num_slots=args.batch_size,
                                           num_decode_steps=args.decode_steps)

        # Correctness: both paths must use batch=1 throughout so XLA kernel
        # reduction orders match and argmax is deterministic.
        #   • static batch_size=1  → batch=1 prefill + batch=1 decode
        #   • CB    num_slots=1    → batch=1 prefill + batch=1 decode
        # (The performance runs above use larger batch sizes whose floating-point
        # rounding differs from batch=1, causing spurious argmax flips that are
        # not CB logic bugs.)
        static_r_bs1 = run_static_batching(model, cfg, requests,
                                           batch_size=1, verbose=False)
        cb_r_ns1     = run_continuous_batching(model, cfg, requests,
                                               num_slots=1,
                                               num_decode_steps=args.decode_steps,
                                               verbose=False)
        verify_correctness(static_r_bs1, cb_r_ns1)

        all_results = {args.distribution: {
            "static":     static_r,
            "continuous": cb_r,
            "requests":   requests,
        }}
        plot_comparison(all_results, args.n_requests)

        print(f"\n  Static:     {static_r['total_time_s']:.2f}s  "
              f"{static_r['throughput_req_s']:.2f} req/s")
        print(f"  Continuous: {cb_r['total_time_s']:.2f}s  "
              f"{cb_r['throughput_req_s']:.2f} req/s")
        print(f"  Speedup (static / continuous): "
              f"{static_r['total_time_s'] / cb_r['total_time_s']:.2f}x")

        # Mean / percentile end-to-end latencies (from benchmark start → retirement,
        # so later static batches also carry their queue-wait cost).
        st_lats = np.array([l["latency_s"] for l in static_r["latencies"]]) * 1000
        cb_lats = np.array([l["latency_s"] for l in cb_r["latencies"]])     * 1000
        print(f"\n  Mean end-to-end request latency (includes queue wait):")
        print(f"    Static     : mean={np.mean(st_lats):7.1f}ms  "
              f"p50={np.percentile(st_lats, 50):7.1f}ms  "
              f"p95={np.percentile(st_lats, 95):7.1f}ms  "
              f"p99={np.percentile(st_lats, 99):7.1f}ms")
        print(f"    Continuous : mean={np.mean(cb_lats):7.1f}ms  "
              f"p50={np.percentile(cb_lats, 50):7.1f}ms  "
              f"p95={np.percentile(cb_lats, 95):7.1f}ms  "
              f"p99={np.percentile(cb_lats, 99):7.1f}ms")
        lat_speedup = np.mean(st_lats) / np.mean(cb_lats)
        direction   = "lower" if lat_speedup > 1 else "higher"
        print(f"  CB mean latency is {abs(lat_speedup):.2f}x {direction} than static")

    print("\n" + "=" * 70)
    print("  Benchmark complete!")
    print("  Plots saved to: project_inference/plots/")
    print("=" * 70)

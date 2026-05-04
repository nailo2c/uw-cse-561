import time
import math
import dataclasses
from pathlib import Path
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh

from model import GPT, forward_v2
from kvcache import KVCache, count_left_padding, prepare_chunk
from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import Config, load_config_from_yaml
from utils import DP_AXIS_NAME

import tiktoken


def build_tokenizer():
    """Build a GPT-2 tokenizer extended with custom chat tokens."""
    user_start = "<|user_start|>"
    user_end = "<|user_end|>"
    assistant_start = "<|assistant_start|>"
    assistant_end = "<|assistant_end|>"
    system_start = "<|system_start|>"
    system_end = "<|system_end|>"
    tool_start = "<|tool_start|>"
    tool_end = "<|tool_end|>"
    pad_token = "<|pad|>"

    custom_tokens = [
        pad_token,
        user_start,
        user_end,
        assistant_start,
        assistant_end,
        system_start,
        system_end,
        tool_start,
        tool_end,
    ]

    base = tiktoken.get_encoding("gpt2")
    custom_token_ids = {tok: base.n_vocab + i for i, tok in enumerate(custom_tokens)}

    tokenizer = tiktoken.Encoding(
        name="gpt2_with_custom_tokens",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **custom_token_ids},
    )

    bos_id = tokenizer.eot_token
    bos = tokenizer.decode([bos_id])

    return {
        "tokenizer": tokenizer,
        "bos_id": bos_id,
        "bos": bos,
        "user_start": user_start,
        "user_end": user_end,
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
        "system_start": system_start,
        "system_end": system_end,
        "tool_start": tool_start,
        "tool_end": tool_end,
        "pad_token": pad_token,
        "pad_id": custom_token_ids[pad_token],
        "assistant_start_id": custom_token_ids[assistant_start],
        "assistant_end_id": custom_token_ids[assistant_end],
        "custom_token_ids": custom_token_ids,
        "vocab_size": tokenizer.n_vocab,
    }


def pad_tokens(tokens, pad_id, pad_to_power_of_two=False):
    curr_max_len = max([len(s) for s in tokens])
    if pad_to_power_of_two:
        pad_to = 2 ** math.ceil(math.log2((curr_max_len)))
    else:
        pad_to = curr_max_len
    padded = []
    segment_ids = []

    for encoded in tokens:
        p, s = prepare_chunk(jnp.array(encoded), pad_id=pad_id, pad_to=pad_to)
        padded.append(p[0])
        segment_ids.append(s[0])

    padded = jnp.stack(padded)
    segment_ids = jnp.stack(segment_ids)
    return padded, segment_ids


@partial(jax.jit, static_argnames=("head_dim", "pad_id"))
def prefill(params, input_ids, segment_ids, cache, head_dim, pad_id):
    """Prefill with LEFT-padded sequences.

    With left padding, starts[b] = count of left pad tokens for sequence b.
    After prefill, cache.iter = seq_len % cache.size (same for all sequences due to alignment).

    Returns the logits at the last position (for sampling the first generated
    token externally) and the updated cache.
    """

    left_pad_counts = count_left_padding(input_ids, pad_id=pad_id)
    uninitialized_iter = -jnp.ones_like(cache.iter)
    cache = dataclasses.replace(cache, starts=left_pad_counts, iter=uninitialized_iter)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    last_token_logits = logits[:, -1, :]
    return last_token_logits, cache


def decode(params, input_ids, cache, head_dim):
    """Decode step for LEFT-padded sequences.

    All sequences generate at the same position since they're aligned at the right.
    """

    segment_ids = jnp.ones_like(input_ids, dtype=jnp.int32)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    return logits[:, -1, :], cache


def sample_from_logits(logits, rng, temperature=1.0, top_k=0):
    # Ideal order for most use cases is: temperature scaling -> top_k -> top_p
    vocab = logits.shape[-1]

    # Greedy path: temperature <= 0
    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    else:
        # Temperature scaling
        tiny = jnp.finfo(jnp.float32).tiny
        logits = logits / jnp.maximum(temperature, tiny)

    if top_k is not None:
        k = int(top_k)
        if 0 < k < vocab:

            def set_values(arr, idx, val):
                return arr.at[idx].set(val)

            values, indices = jax.lax.top_k(logits, k)
            filtered_logits = jnp.full_like(logits, fill_value=-jnp.inf)
            logits = jax.vmap(set_values, in_axes=(0, 0, 0))(
                filtered_logits, indices, values
            )
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)


@partial(jax.jit, static_argnames=("temperature", "head_dim", "max_new_tokens", "top_k"))  # fmt: off
def generate(
    params,
    cache,
    last_token,
    generated_tokens,
    head_dim,
    decode_key,
    temperature,
    top_k,
    max_new_tokens,
):
    def decode_body(carry, t):
        cache, last_token, decode_key, generated_tokens = carry
        logits, cache = decode(params, last_token, cache, head_dim)
        decode_key, sub = jax.random.split(decode_key)
        token = sample_from_logits(logits, sub, temperature, top_k)
        generated_tokens = generated_tokens.at[:, t].set(token)
        return (cache, token[:, None], decode_key, generated_tokens), None

    (cache, last_token, decode_key, generated_tokens), _ = jax.lax.scan(
        decode_body,
        (cache, last_token, decode_key, generated_tokens),
        jnp.arange(1, max_new_tokens),
    )
    return generated_tokens


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GPT inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to a YAML configuration file (e.g. configs/default.yaml)",
    )
    cli_args = parser.parse_args()

    devices = np.array(jax.devices())
    print("Found devices: ", devices)
    print("Platform: ", devices[0].platform)
    mesh = Mesh(devices, axis_names=DP_AXIS_NAME)
    print(f"Loading configuration from: {cli_args.config}")
    cfg = load_config_from_yaml(cli_args.config, mesh=mesh)

    # Get the weight shardings
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    model_sharding = GPT.shardings(cfg.mesh, cfg.model)
    print("Model built successfully!\n")
    ckpt_params_path = str(Path(cfg.ckpt_cfg.load_params_ckpt_path).resolve())
    model = load_weights_from_checkpoint_with_validation(
        ckpt_params_path, model, model_sharding
    )
    print("Weights loaded from the checkpoint successfully!")

    tok_info = build_tokenizer()
    tokenizer = tok_info["tokenizer"]
    PAD_ID = tok_info["pad_id"]

    head_dim = cfg.model.attn.head_dim
    max_new_tokens = 100
    top_k = 500
    temperature = 0.8

    # Warm up the compiler with a batch that matches the device count.
    prompts = ["<|endoftext|>Did you hear the noise coming "] * len(devices)
    encoded = tokenizer.encode_batch(prompts, allowed_special="all")
    input_ids, segment_ids = pad_tokens(encoded, PAD_ID, pad_to_power_of_two=True)
    key = jax.random.PRNGKey(123)
    print("Warming up the model...")

    cache_key = jax.random.PRNGKey(1)
    batch_size = input_ids.shape[0]
    cache = KVCache.init(cache_key, cfg.mesh, batch_size, cfg)

    with jax.set_mesh(cfg.mesh):
        key, prefill_key, decode_key = jax.random.split(key, 3)
        last_token_logits, cache = prefill(
            model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID
        )
        next_tokens = sample_from_logits(last_token_logits, prefill_key, temperature, top_k)

        generated_tokens = (
            jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
            .at[:, 0]
            .set(next_tokens)
        )
        last_token = next_tokens[:, None]
        warmup_generated = generate(
            model,
            cache,
            last_token,
            generated_tokens,
            head_dim,
            decode_key,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        jax.block_until_ready(warmup_generated)
    print("Warming up complete!")
    print("=" * 60)
    print("Interactive generation mode. Type your prompt and press Enter.")
    print("Type 'quit' or 'exit' to stop. Press Ctrl+C to abort.\n")

    BOS = "<|endoftext|>"
    num_devices = len(devices)

    while True:
        try:
            user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.strip().lower() in ("quit", "exit"):
            print("Exiting.")
            break

        if not user_input.strip():
            continue

        prompt = BOS + user_input
        prompts = [prompt] * num_devices
        encoded = tokenizer.encode_batch(prompts, allowed_special="all")
        input_ids, segment_ids = pad_tokens(encoded, PAD_ID, pad_to_power_of_two=True)
        batch_size = input_ids.shape[0]
        cache = KVCache.init(cache_key, cfg.mesh, batch_size, cfg)
        key, prefill_key, decode_key = jax.random.split(key, 3)

        start = time.perf_counter()
        with jax.set_mesh(cfg.mesh):
            last_token_logits, cache = prefill(
                model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID
            )
            next_tokens = sample_from_logits(last_token_logits, prefill_key, temperature, top_k)

        generated_tokens = (
            jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
            .at[:, 0]
            .set(next_tokens)
        )
        last_token = next_tokens[:, None]

        with jax.set_mesh(cfg.mesh):
            generated = generate(
                model,
                cache,
                last_token,
                generated_tokens,
                head_dim,
                decode_key,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
        elapsed = time.perf_counter() - start

        completion = tokenizer.decode(generated[0].tolist())
        print(f"\n{completion}")
        print(f"\n[{max_new_tokens} tokens in {elapsed:.2f}s | {max_new_tokens / elapsed:.1f} tok/s]")
        print("-" * 60)

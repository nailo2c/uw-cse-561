# Project 2: LLM Inference Profiling on a Single-TPU

## Project 2 Overview

In this project, you will study the inference behavior of a GPT-2 style language model on a TPU v5e accelerator. While Project 1 focused on training (forward + backward + optimizer), this project focuses on **inference** — the process of generating text from a trained model.

LLM inference consists of two distinct phases:

- **Prefill**: Process the entire input prompt in parallel, populating the KV cache.
- **Decode**: Generate tokens one at a time, reading the KV cache at each step.

These two phases have fundamentally different compute characteristics. Understanding these characteristics is critical for building efficient serving systems.

You will:

1. Analyze the **operational intensity** (compute-to-memory ratio) of key operations
2. **Microbenchmark** attention and GEMM kernels in isolation
3. Profile **end-to-end inference** to understand latency bottlenecks
4. Implement and benchmark static and continous batching policies

### Hardware & Software

- **TPU v5e**: 197 TFLOPS (BF16), 16 GB HBM, 819 GB/s memory bandwidth
- **JAX** with XLA compiler on TPU

### Setup

Follow the setup instructions from Project 1. Then verify training works (You can stop after the first several training steps):

```bash
source .venv/bin/activate
python3.11 nanogpt/train.py --config configs/small.yaml
```

---

## Section 1: Theoretical Analysis (20 points)

*This section is write-up only. No experiments to run.*

### Q1.1 — GEMM Operational Intensity (5 points)

A key building block of transformer inference is General Matrix Multiplication (GEMM). In inference, each linear layer computes `Y = X @ W` where `X ∈ ℝ^{M×K}` and `W ∈ ℝ^{K×N}` in BF16.

1. Derive the **operational intensity** (FLOPs/byte) of a GEMM in terms of M, N, K. Assume BF16 (2 bytes per element). Count both reads (A, B) and writes (C).
2. The following GEMM shapes appear in each transformer layer of the `small` model (`d_emb=768, mlp_hidden_dim=3072, q_heads=8, kv_heads=8, head_dim=96`):


| Operation         | K    | N    |
| ----------------- | ---- | ---- |
| Q projection      | 768  | 768  |
| K projection      | 768  | 768  |
| V projection      | 768  | 768  |
| Output projection | 768  | 768  |
| MLP fc1 (up)      | 768  | 3072 |
| MLP fc2 (down)    | 3072 | 768  |


   Compute the operational intensity for M = 1, 4, 16, 64, 128, 256, 512, 1024. Fill in a table.

1. The TPU v5e has a **ridge point** of approximately 240 FLOP/byte (197 TFLOPS ÷ 819 GB/s). For each M value, determine whether the GEMM is **compute-bound** or **memory-bound**. At what M does the transition occur?

### Q1.2 — Attention Operational Intensity (10 points)

1. Derive the operational intensity for **prefill attention** (batch=1, prompt length `p`):
  - Operations: Q×K^T (score computation) and Attn×V (value aggregation)
  - Data: Reading Q, K, V and writing output O, each of shape `(1, num_heads, p, head_dim)`
2. Derive the operational intensity for **decode attention** (batch=1, context length `c`):
  - Q has shape `(1, num_heads, 1, head_dim)` (single new token)
  - K, V cache have shape `(1, num_heads, c, head_dim)`
3. For both model configs (`small` and `default`), compute the operational intensity with `p, c = 2^7, 2^8, ..., 2^{13}`. Fill in a table for each and mark compute-bound vs memory-bound.

### Q1.3 — Prefill vs Decode (5 points)

Based on your analysis in Q1.1 and Q1.2:

1. Explain why **prefill is compute-bound**: What structural property of prefill enables high arithmetic intensity?
2. Explain why **decode is memory-bound**: What about autoregressive generation causes low arithmetic intensity?
3. What are the implications for hardware utilization? Which metric (TFLOPS or GB/s) is more relevant for each phase?

---

## Section 2: Microbenchmarking (20 points)

### Q2.1 — Prefill Attention Performance (10 points)

Run the prefill attention benchmark:

```bash
python3.11 project_inference/bench_prefill.py
```

This benchmarks single-layer prefill attention using XLA (`jax.nn.dot_product_attention`).

1. **Vary sequence length** (batch=1, `p = 2^7, ..., 2^{11}`): Plot compute utilization (TFLOPS) with `log₂(p)` on the x-axis. One subplot per model config.
2. **Vary batch size** (seq_len=1024, batch = 1, 2, ..., 16): Plot compute utilization with `log₂(batch)` on the x-axis.
3. At what point (sequence length / batch size) does the attention kernel's FLOP utilisation begin to level-off and saturate? How does TFLOPS scale with sequence length and batch size?

### Q2.2 — Decode Attention Performance (10 points)

Run the decode attention benchmark:

```bash
python3.11 project_inference/bench_decode.py
```

1. **Vary context length** (batch=1, `c = 2^7, ..., 2^{13}`): Plot **memory bandwidth utilization** (GB/s) with `log₂(c)` on the x-axis.
2. **Vary batch size** (ctx=1024, batch = 1, ..., 64): Plot memory bandwidth utilization with `log₂(batch)` on the x-axis.
3. Why is **memory bandwidth** (GB/s) the correct performance metric for decode, rather than TFLOPS? Relate to your operational intensity analysis.
4. How close does the measured bandwidth get to the TPU v5e peak of 819 GB/s? What factors prevent achieving full bandwidth?

---

## Section 3: End-to-End Inference (30 points)

### Q3.1 — Prefill Latency Analysis (10 points)

Run the end-to-end benchmark with profiling (this command runs experiments for Q3.1, Q3.2, and Q3.3):

```bash
python3.11 project_inference/bench_e2e.py --config configs/small.yaml --profile
```

1. With batch_size=1, measure prefill latency for `p = 2^7, ..., 2^{12}`. Plot prefill latency vs `log₂(p)`.
2. Examine the profiler traces (saved to `project_inference/profiles/`) in TensorBoard:
  ```bash
   tensorboard --logdir project_inference/profiles/ --port 6006
  ```
3. Using the Trace Viewer, identify the time breakdown across operations: QKV projection, attention, output projection, MLP, norms, embedding.
4. Which operations dominate at **short** prefill lengths (128)? Which dominate at **long** prefill lengths (4096+)? Relate this to the operational intensity analysis from Section 1.

### Q3.2 — Decode Latency & Throughput (10 points)


1. With batch=1, prefill=128, vary decode length `d = 2^5, ..., 2^{10}`:
   - Plot total decode time and per-token latency vs `log₂(d)`.
   - Does per-token decode latency stay roughly constant? Explain why or why not based on the KV cache size growth.

2. With prefill=128, decode=128, vary batch_size = 1, 2, 4, 8, 16, 32:
   - Plot total throughput `(prefill_tokens + decode_tokens) / time` vs `log₂(batch)`.
   - When does throughput saturate? Relate to the memory-bound → compute-bound transition.
   - Why is **batching** the primary technique for improving decode throughput?

### Q3.3 — Memory Footprint (10 points)

1. For the `small` model, compute theoretically:
  - Total model weight memory (BF16)
  - KV cache memory for `seqlen=1024` at batch sizes 1, 4, 8, 16
  - At what batch size does KV cache memory exceed model weight memory?
2. Run `bench_e2e.py --config configs/small.yaml --profile` to validate your calculations.
3. Given the TPU v5e has 16 GB HBM, what is the maximum batch size you can serve? What are the implications for inference serving?

## Section 4: Static and Continous Batching (40 points)
For this section, you are provided with template code in `bench_continous_batching.py` to implement static and contninous batching logic.
The implementation conists of two parts: 
   - [Part 1] Static Batching: Implement a static batching policy where requests in the `pending` queue are dequeued and batched together (upto a maximum batch size). All reequets
     in the batch are first prefilled, and then decoded until the request with the largest length has finished its token generation. The entire batch is then retired
     to execute the next batch of requests.
   - [Part 2] Continous Batching: Implement a continous batching mechnism where the request batch is constructed dynamically during token generation. Continous batching consists of a main loop
     that advances the token generation of all active requests by a pre-specified constant `num_decode_steps`. Every `num_decode_steps`, the scheduler iterates over the batch of requests to deallocate requests that have complete token generation and admits new requests from the `pending` queue, if any.
      
Parts of the code that need to be implemented are marked with a `#TODO`. Test your code by running the script with the arguments provided below. Include the following in your write-up:

```bash
 python3.11 project_inference/bench_continous_batching.py \
     --config configs/small.yaml \
     --distribution bimodal \
     --prompt-len 128 \
     --decode-len 1024 \
     --n-requests 16 \
     --decode-steps 64 \
     --batch-size 4
```

1. Generate plot showing throughput and end-to-end request latencies for static and continous batching. Which batching policy achieves a higher throughput? Which policy achieves a lower end-to-end latency? Provide reasons for the observed performance differences.
2. For continous batching, plot throughput and mean end-to-end latencies for `num_decode_steps =  2^0, .. , 2^10`. How do the throughput and latencies vary with `num_decode_steps`? Provide reasoning for your observations.
3. Profile static and continous batching for a fixed request length distribution using `--distribution fixed`. How do the throughput and latencies compare in this case? Provide reasoning for your observations. 
---

## (Optional) Section 5: (Bonus 40 points!)
Implement chunked prefill in JAX/XLA and compare its performance (token throughput, per-request latency) with full prefill implementation. You can extend the code provided in `bench_continous_batching.py` for your implementation. How does the chunked prefill performance vary with respect to prefill `token_budget`? 

## Running the Tools

All scripts are in `project_inference/`:

| Script | Section | Description |
|---|---|---|
| `bench_prefill.py` | §2.1 | Prefill attention benchmarking |
| `bench_decode.py` | §2.2 | Decode attention benchmarking |
| `bench_e2e.py` | §3 | End-to-end inference profiling |
| `bench_continous_batching.py` (template) | §4 | Static and Continous batching | 

Helper modules: `bench_utils.py` (timing/FLOP calculations), `plot_utils.py` (plotting).

Plots are saved to `project_inference/plots/`.
Profiler traces are saved to `project_inference/profiles/`.

## Submission

Please submit a PDF report and your code implemenatation for Section 4. Include:

- Your name and student ID
- Answers to all questions with tables, plots, and profiler screenshots
- Clear explanations connecting theory (Section 1) to measurements (Sections 2-3)

Upload the final PDF and code (zipped) to Gradescope.

"""
Memory efficient fused MLP in Pallas (TPU).

Operation: Y = SiLU(X @ W1) @ W2
Shapes: X: (M, K), W1: (K, H), W2: (H, N), Y: (M, N)
"""

import csv
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M, K, H, N = 4096, 1024, 16384, 1024
DTYPE = jnp.bfloat16
ACC_DTYPE = jnp.float32

PEAK_GBPS = 819.0 # TPU v5e HBM
PEAK_TFLOPS_BF16 = 197.0 # TPU v5e bf16 MXU peak

BM_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
BH_VALUES = [128, 256, 512, 1024, 2048]


def tpu_compiler_params(*, dimension_semantics):
    params_cls = getattr(pltpu, "CompilerParams", pltpu.TPUCompilerParams)
    return params_cls(dimension_semantics=dimension_semantics)


#############################################################
# reference JAX implements 
#############################################################
def jax_mlp(x, w1, w2):
    return jax.nn.silu(x @ w1) @ w2


#############################################################
# Naive Pallas kernel
#############################################################
def make_fused_pallas_naive(bm: int):
    """Naive fused MLP that loads W1 and W2 fully into VMEM. """

    def kernel(x_ref, w1_ref, w2_ref, y_ref):
        x = x_ref[...]    # [BM, K]
        w1 = w1_ref[...]  # [K, H]
        w2 = w2_ref[...]  # [H, N]

        hidden = jnp.dot(x, w1, preferred_element_type=ACC_DTYPE)
        hidden = hidden / (1 + jnp.exp(-hidden))

        y = jnp.dot(hidden.astype(w2.dtype), w2, preferred_element_type=ACC_DTYPE)
        y_ref[...] = y.astype(y_ref.dtype)

    @jax.jit
    def fn(x, w1, w2):
        m, k = x.shape
        _, h = w1.shape
        _, n = w2.shape

        assert m % bm == 0

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
            grid=(m // bm,),
            in_specs=[
                pl.BlockSpec((bm, k), lambda i: (i, 0)),
                pl.BlockSpec((k, h), lambda i: (0, 0)),
                pl.BlockSpec((h, n), lambda i: (0, 0)),
            ],
            out_specs=pl.BlockSpec((bm, n), lambda i: (i, 0)),
            compiler_params=tpu_compiler_params(
                dimension_semantics=("parallel",)
            ),
        )(x, w1, w2)

    return fn


#############################################################
# Optimized fused MLP kernel
#############################################################
def make_fused_pallas(bm: int, bh: int):
    """Implement a memory optimized fused MLP kernel using Pallas"""
    def kernel(x_ref, w1_ref, w2_ref, y_ref, y_acc_scratch):
        h_i = pl.program_id(1)

        @pl.when(h_i == 0)
        def _zero_acc():
            y_acc_scratch[...] = jnp.zeros_like(y_acc_scratch[...])

        x = x_ref[...]
        w1 = w1_ref[...]
        w2 = w2_ref[...]

        hidden = jnp.dot(x, w1, preferred_element_type=ACC_DTYPE)
        hidden = hidden / (1 + jnp.exp(-hidden))

        partial = jnp.dot(
            hidden.astype(w2.dtype),
            w2,
            preferred_element_type=ACC_DTYPE,
        )

        y_acc_scratch[...] += partial

        @pl.when(h_i == pl.num_programs(1) - 1)
        def _store_out():
            y_ref[...] = y_acc_scratch[...].astype(y_ref.dtype)

    @jax.jit
    def fn(x, w1, w2):
        m, k = x.shape
        _, h = w1.shape
        _, n = w2.shape

        assert m % bm == 0
        assert h % bh == 0

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
            grid=(m // bm, h // bh),
            in_specs=[
                pl.BlockSpec((bm, k), lambda mi, hi: (mi, 0)),
                pl.BlockSpec((k, bh), lambda mi, hi: (0, hi)),
                pl.BlockSpec((bh, n), lambda mi, hi: (hi, 0)),
            ],
            out_specs=pl.BlockSpec((bm, n), lambda mi, hi: (mi, 0)),
            scratch_shapes=(
                pltpu.VMEM((bm, n), dtype=ACC_DTYPE),
            ),
            compiler_params=tpu_compiler_params(
                dimension_semantics=("parallel", "arbitrary")
            ),
        )(x, w1, w2)

    return fn


#############################################################
# Benchmark utility
#############################################################
def benchmark(fn, args, n_warmup: int = 3, n_iters: int = 20) -> float:
    for _ in range(n_warmup):
        fn(*args).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y = fn(*args).block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


def fused_mlp_flops(m: int, k: int, h: int, n: int) -> int:
    return 2 * m * k * h + 2 * m * h * n


def tflops_and_util(t: float, m: int, k: int, h: int, n: int):
    tflops = fused_mlp_flops(m, k, h, n) / t / 1e12
    util = tflops / PEAK_TFLOPS_BF16 * 100
    return tflops, util


def make_inputs(m: int, k: int, h: int, n: int):
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (m, k), dtype=DTYPE)
    w1 = jax.random.normal(k2, (k, h), dtype=DTYPE) * 0.02
    w2 = jax.random.normal(k3, (h, n), dtype=DTYPE) * 0.02
    x.block_until_ready()
    w1.block_until_ready()
    w2.block_until_ready()
    return x, w1, w2


def check_correctness():
    m, k, h, n = 16, 128, 256, 128
    bm, bh = 8, 128
    x, w1, w2 = make_inputs(m, k, h, n)
    ref = jax.jit(jax_mlp)(x, w1, w2).astype(jnp.float32)

    checks = [
        ("naive", make_fused_pallas_naive(bm)),
        ("optimized", make_fused_pallas(bm, bh)),
    ]
    for name, fn in checks:
        y = fn(x, w1, w2).astype(jnp.float32)
        y.block_until_ready()
        max_abs = float(jnp.max(jnp.abs(y - ref)))
        ok = bool(jnp.allclose(y, ref, rtol=5e-2, atol=5e-2))
        print({"name": f"correctness_{name}", "status": "ok" if ok else "failed", "max_abs": max_abs})
        if not ok:
            raise AssertionError(f"{name} correctness failed: max_abs={max_abs}")


def run_case(name, fn, args, m, k, h, n):
    try:
        t = benchmark(fn, args)
        tflops, util = tflops_and_util(t, m, k, h, n)
        return {
            "name": name,
            "status": "ok",
            "time_s": t,
            "tflops": tflops,
            "util_percent": util,
            "error": "",
        }
    except Exception as e:
        return {
            "name": name,
            "status": "oom_or_error",
            "time_s": "",
            "tflops": "",
            "util_percent": "",
            "error": repr(e),
        }


def run_q2_naive_h_sweep():
    rows = []
    m, k, n = 4096, 1024, 1024
    bm = 1024
    h_values = [512, 1024, 2048, 4096, 8192, 16384]

    for h in h_values:
        x, w1, w2 = make_inputs(m, k, h, n)
        fn = make_fused_pallas_naive(bm)
        result = run_case(f"naive_h_{h}", fn, (x, w1, w2), m, k, h, n)
        result.update({"bm": bm, "bh": "", "m": m, "k": k, "h": h, "n": n})
        print(result)
        rows.append(result)

    return rows


def run_q2_naive_bm_sweep(h_success):
    rows = []
    m, k, n = 4096, 1024, 1024

    for bm in BM_VALUES:
        x, w1, w2 = make_inputs(m, k, h_success, n)
        fn = make_fused_pallas_naive(bm)
        result = run_case(f"naive_bm_{bm}", fn, (x, w1, w2), m, k, h_success, n)
        result.update({"bm": bm, "bh": "", "m": m, "k": k, "h": h_success, "n": n})
        print(result)
        rows.append(result)

    return rows


def run_q3_optimized_sweep():
    rows = []
    m, k, h, n = 4096, 1024, 16384, 1024
    x, w1, w2 = make_inputs(m, k, h, n)

    for bm in BM_VALUES:
        for bh in BH_VALUES:
            fn = make_fused_pallas(bm, bh)
            result = run_case(f"opt_bm_{bm}_bh_{bh}", fn, (x, w1, w2), m, k, h, n)
            result.update({"bm": bm, "bh": bh, "m": m, "k": k, "h": h, "n": n})
            print(result)
            rows.append(result)

    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    check_correctness()

    q2_h_rows = run_q2_naive_h_sweep()
    write_csv("q2_naive_h_sweep.csv", q2_h_rows)

    successful_h = [
        r["h"] for r in q2_h_rows
        if r["status"] == "ok"
    ]
    if successful_h:
        h_success = max(successful_h)
        q2_bm_rows = run_q2_naive_bm_sweep(h_success)
        write_csv("q2_naive_bm_sweep.csv", q2_bm_rows)

    q3_rows = run_q3_optimized_sweep()
    write_csv("q3_optimized_sweep.csv", q3_rows)

if __name__ == "__main__":
    main()

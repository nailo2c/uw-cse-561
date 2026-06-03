import jax
import jax.numpy as jnp
import time

from silu_pallas import silu_pallas

def silu_naive(x):
    return x/(1+jnp.exp(-x))

# (https://docs.jax.dev/en/latest/_autosummary/jax.nn.sigmoid.html)
def silu_sigmoid(x):
    return x * jax.nn.sigmoid(x)


def benchmark(fn, x, name, n_warmup=5, n_iters=100):
    # run a few warmup iterations
    for _ in range(n_warmup):
        fn(x).block_until_ready()

    with jax.profiler.trace("/tmp/silu-trace-%s.json" % name):
        t_start = time.perf_counter()
        for _ in range(n_iters):
            y = fn(x).block_until_ready()
        t_end = time.perf_counter()

    return (t_end - t_start) / n_iters

def check_correctness(fn, x, ref, name, atol=1e-5):
    y = fn(x)
    y.block_until_ready()
    abs_err = jnp.abs(y - ref)
    max_abs = float(jnp.max(abs_err))
    ok = bool(jnp.allclose(y, ref, atol=atol))
    status = "OK " if ok else "FAIL"
    # print(f"[{status}] {name:20s}  max_abs={max_abs:.2e}")
    return ok

def main():
    shape = (8192, 8192)
    dtype = jnp.float32
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, dtype=dtype)
    x.block_until_ready()

    num_elements = x.size
    bytes_per_element = jnp.dtype(dtype).itemsize
    bytes_moved = 2 * num_elements * bytes_per_element
    PEAK_GBPS = 819 # for TPU ve5
    
    benchmark_targets = [
        ("naive (no jit)", silu_naive),
        ("x*sigmoid (no jit)", silu_sigmoid),
        ("naive + jit", jax.jit(silu_naive)),
        ("x*sigmoid +jit", jax.jit(silu_sigmoid)),
        ("pallas", jax.jit(silu_pallas)),
        ("jax.nn.silu", jax.nn.silu),  # reference
    ]

    ref = jax.nn.silu(x)
    for name, fn in benchmark_targets:
        ok = check_correctness(fn, x, ref, name)
        if not ok:
            print(f"{name:20s} failed correctness check")
            continue
        t = benchmark(fn, x, name)
        gbps = bytes_moved / t / 1e9
        util = gbps / PEAK_GBPS * 100
        print(f"{name:20s}  {t*1e6:8.2f} us  {gbps:8.1f} GB/s  {util:5.1f}%")
if __name__ == "__main__":
    main()

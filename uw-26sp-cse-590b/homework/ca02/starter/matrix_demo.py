"""Progression demo for Part 2 Option 3 (not submitted).

Walks through two configurations so you can see why abstraction is
needed at all, and why axioms are needed once you abstract. Step 1
uses concrete Z3 bitvector operators (correct but scales badly);
Step 2 abstracts every operator as an uninterpreted function with no
axioms (fast, but wrong — the solver returns a spurious SAT).

Your own abstraction + axioms live in `part2_opt3_matrix.py`; run that
file directly to smoke-test your chosen configuration at N=8.

Run:  python matrix_demo.py
"""
import z3

import matrix as _matrix
from matrix import query_2
from part2_opt3_matrix import BV_BITS

BV = z3.BitVecSort(BV_BITS)

# Per-query wall-clock ceiling. Step 1 (concrete) is expected to blow
# past this; the 10s ceiling keeps the demo bounded on every machine
# without needing an external `timeout` wrapper.
_TIMEOUT_MS = 10_000


def _install(bvadd, bvmul, bveq, add_axioms):
    _matrix.bvadd = bvadd
    _matrix.bvmul = bvmul
    _matrix.bveq = bveq
    _matrix.add_axioms = add_axioms


def _run_query_2(sizes, label):
    print(f"\n=== {label} ===")
    for n in sizes:
        r, t = query_2(n, timeout_ms=_TIMEOUT_MS)
        print(f"  query_2(N={n}) -> {r} in {t:.3f}s")
        if str(r) == "unknown":
            print(f"    (timed out after {_TIMEOUT_MS/1000:.0f}s; skipping larger N)")
            break


if __name__ == "__main__":
    # ---- Step 1: everything concrete --------------------------------
    # Z3's built-in bitvector operators. Answer is correct (UNSAT) but
    # time grows fast with N — N=20 is hopeless.
    _install(
        lambda x, y: x + y,
        lambda x, y: x * y,
        lambda x, y: x == y,
        lambda s: None,
    )
    _run_query_2([3, 4, 5], "Step 1: all concrete (correct, slow)")

    # ---- Step 2: all UF, no axioms ----------------------------------
    # The solver now knows nothing about +, *, or ==. It finds a
    # spurious counterexample instantly: SAT, even though the identity
    # is actually true. This proves axioms are required for every
    # operator you abstract.
    uf_add = z3.Function("demo_add", BV, BV, BV)
    uf_mul = z3.Function("demo_mul", BV, BV, BV)
    uf_eq = z3.Function("demo_eq", BV, BV, z3.BoolSort())
    _install(
        lambda x, y: uf_add(x, y),
        lambda x, y: uf_mul(x, y),
        lambda x, y: uf_eq(x, y),
        lambda s: None,
    )
    _run_query_2([3, 4, 5], "Step 2: all UF, no axioms (fast, WRONG)")

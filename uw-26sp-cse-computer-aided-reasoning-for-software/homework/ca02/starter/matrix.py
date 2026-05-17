"""Matrix-multiplication driver + verification queries (Part 2 Option 3).

Imports operator names by name from `part2_opt3_matrix` — the student's
submission — so that toggling a definition there changes whether that
operator is abstracted as a UF.
"""
import csv
import sys
import time
from pathlib import Path

import z3

# When `matrix.py` is executed directly, also make its own directory
# importable so `from part2_opt3_matrix import ...` resolves. (In the
# test environment, conftest.py already puts the relevant dirs on
# sys.path.)
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from part2_opt3_matrix import bvadd, bvmul, bveq, BV_BITS, add_axioms


def dot_product(u, v):
    assert len(u) == len(v) and len(u) > 0
    total = bvmul(u[0], v[0])
    for i in range(1, len(u)):
        total = bvadd(total, bvmul(u[i], v[i]))
    return total


def transpose(A):
    n_rows = len(A)
    n_cols = len(A[0])
    return [[A[r][c] for r in range(n_rows)] for c in range(n_cols)]


def multiply(A, B):
    Bt = transpose(B)
    return [[dot_product(row, col) for col in Bt] for row in A]


def symbolic_matrix(n, prefix):
    return [[z3.BitVec(f"{prefix}_{i}_{j}", BV_BITS)
             for j in range(n)] for i in range(n)]


def matrices_equal(A, B):
    assert len(A) == len(B) and len(A[0]) == len(B[0])
    lits = []
    for r in range(len(A)):
        for c in range(len(A[0])):
            lits.append(bveq(A[r][c], B[r][c]))
    return z3.And(lits)


def _make_solver(timeout_ms=None):
    s = z3.Solver()
    # Disable Z3's solve-eqs preprocessor. Otherwise Z3 substitutes the
    # matrix-equality premise element-wise, canonicalizes both sides
    # with its AC rewriter, and discharges the query without ever
    # reasoning about the semantics of bvmul — defeating the point of
    # the exercise. With solve_eqs off, concrete bvmul actually gets
    # bit-blasted, which is slow enough to motivate abstraction.
    s.set("solve_eqs", False)
    if timeout_ms is not None:
        s.set("timeout", timeout_ms)
    return s


def query_1(n, timeout_ms=None):
    """A = B  ⇒  A*B = B*A"""
    s = _make_solver(timeout_ms)
    A = symbolic_matrix(n, f"q1a_{n}")
    B = symbolic_matrix(n, f"q1b_{n}")
    s.add(matrices_equal(A, B))
    s.add(z3.Not(matrices_equal(multiply(A, B), multiply(B, A))))
    # `add_axioms` runs after query construction so it can see every
    # abstracted-op invocation and emit the needed ground instances.
    add_axioms(s)
    t0 = time.time()
    result = s.check()
    return result, time.time() - t0


def query_2(n, timeout_ms=None):
    """B = A^T  ⇒  (A*A)^T = B*B"""
    s = _make_solver(timeout_ms)
    A = symbolic_matrix(n, f"q2a_{n}")
    B = symbolic_matrix(n, f"q2b_{n}")
    s.add(matrices_equal(B, transpose(A)))
    s.add(z3.Not(matrices_equal(transpose(multiply(A, A)), multiply(B, B))))
    add_axioms(s)
    t0 = time.time()
    result = s.check()
    return result, time.time() - t0


def main(out_path="results.csv", sizes=None):
    sizes = sizes if sizes is not None else list(range(3, 21))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "N", "seconds", "result"])
        for n in sizes:
            for name, q in (("query_1", query_1), ("query_2", query_2)):
                r, t = q(n)
                w.writerow([name, n, f"{t:.3f}", str(r)])
                print(f"{name}(N={n}) -> {r} in {t:.3f}s", flush=True)


if __name__ == "__main__":
    main()

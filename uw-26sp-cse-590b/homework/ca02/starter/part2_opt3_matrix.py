"""Part 2 Option 3 — operator definitions for matrix-identity verification.

This file is the only thing you submit. `matrix.py` (shipped alongside)
imports the names `bvadd`, `bvmul`, `bveq`, `BV_BITS`, and `add_axioms`
from here and uses them to build two verification queries:

    query_1(n):  A = B  =>  A*B = B*A
    query_2(n):  B = A^T => (A*A)^T = B*B

Both queries should discharge as UNSAT in seconds at N = 20. At N = 20
with the concrete Z3 bitvector multiply, they take a very long time —
your job is to abstract a carefully-chosen subset of the operators as
uninterpreted functions and, if necessary, supply ground-instance
axioms about them so the queries still discharge correctly.

### How to use this file

For each operator below there are two pre-written definitions:

    # --- concrete (default Z3 op) ---
    def bvmul(x, y):
        return x * y

    # --- abstracted (wrapped in an uninterpreted Function) ---
    # _uf_bvmul = Function("uf_bvmul", BV, BV, BV)
    # def bvmul(x, y):
    #     return _uf_bvmul(x, y)

Exactly one of the two must be active at any time. To abstract an
operator, comment out the concrete block and uncomment the abstracted
block. To switch back, do the reverse.

Add ground-instance axioms about abstracted operators inside
`add_axioms(solver)`. That hook is called by `matrix.py` after each
query is fully built, so it sees the query's symbolic variables.

### Constraints

- All axioms must be ground instances — no `ForAll`.
- Keep the set of abstracted operators as small as possible while still
  getting both queries to discharge at N = 20. More abstraction is not
  better: abstracting addition as well, for example, would force you
  to re-supply associativity and commutativity, which explodes axiom
  counts fast.

### Adjusting the provided definitions

The two pre-written definitions for each operator are a starting
point, not a contract. You may rewrite them in whatever way helps
you — for example, having the abstracted `bvmul` record every
`(x, y)` pair it sees in a module-level list, so that `add_axioms`
can emit ground commutativity instances only for the pairs that
actually appear in the queries. The only hard requirements are that
`bvadd`, `bvmul`, `bveq`, `BV_BITS`, and `add_axioms` remain
importable by `matrix.py`, and that all axioms are ground.

### Z3 primitives you may use

    BitVecSort(n)          sort of n-bit bitvectors
    BoolSort()             Boolean sort (needed if you abstract `bveq`)
    Function(name, *sorts) uninterpreted function; last sort is the
                           return sort

The operators `bvadd`, `bvmul`, `bveq` you expose here must accept Z3
expressions of type BitVec(BV_BITS). For the concrete versions, the
Python operators `+`, `*`, and `==` on Z3 BitVec expressions do the
right thing.
"""
from z3 import BitVecSort, BoolSort, Function

BV_BITS = 32
BV = BitVecSort(BV_BITS)


# ================= bvadd =================
# --- concrete (default Z3 bvadd) ---
def bvadd(x, y):
    return x + y

# --- abstracted ---
# _uf_bvadd = Function("uf_bvadd", BV, BV, BV)
# def bvadd(x, y):
#     return _uf_bvadd(x, y)


# ================= bvmul =================
# --- concrete ---
# def bvmul(x, y):
#     return x * y

# --- abstracted ---
_uf_bvmul = Function("uf_bvmul", BV, BV, BV)
seen_pairs = set()
def bvmul(x, y):
    seen_pairs.add((x, y))
    return _uf_bvmul(x, y)


# ================= bveq =================
# --- concrete (returns a Z3 Bool) ---
def bveq(x, y):
    return x == y

# --- abstracted ---
# _uf_bveq = Function("uf_bveq", BV, BV, BoolSort())
# def bveq(x, y):
#     return _uf_bveq(x, y)


def add_axioms(solver):
    """Called once by `matrix.py` after each query is built and before
    the solver is checked. Assert any ground-instance facts about the
    abstracted operators here.

    When no operators are abstracted, this can stay empty.
    """
    for x, y in seen_pairs:
        solver.add(_uf_bvmul(x, y) == _uf_bvmul(y, x))
    seen_pairs.clear()


if __name__ == "__main__":
    import z3

    from matrix import query_1, query_2

    # Smoke test: with your abstraction + axioms in place, both queries
    # should discharge as UNSAT in well under five seconds at N=8.
    for name, q in (("query_1", query_1), ("query_2", query_2)):
        r, t = q(8)
        assert r == z3.unsat, f"{name}(N=8) returned {r}"
        assert t < 5.0, f"{name}(N=8) took {t:.3f}s (budget 5.0s)"
        print(f"{name}(N=8) -> {r} in {t:.3f}s")

    print(
        "\nFor a step-by-step walk-through of why abstraction helps and why "
        "axioms are\nrequired once you abstract, run:  python matrix_demo.py"
    )

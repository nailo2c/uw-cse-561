"""Option 1 — Magic square solver.

A *magic square of order N* is an N-by-N grid filled with the integers
1, 2, ..., N**2 (each used exactly once) such that every row, every
column, and both main diagonals sum to the same constant
T = N * (N**2 + 1) // 2.

### Input / Output

    solve_magic_square(N) -> list[list[int]] | None

    N : int — order of the magic square.

    Return an N-by-N list-of-lists `square` such that
    `square[r][c]` is the value in row r, column c, and the square is
    magic. Return None if no such square exists.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    BitVec(name, w)     fresh bit-vector variable of width w
    BitVecVal(v, w)     bit-vector constant of value v and width w
    ULE(a, b)           unsigned a <= b
    ULT(a, b)           unsigned a <  b
    UGE(a, b)           unsigned a >= b
    UGT(a, b)           unsigned a >  b
    Distinct(*xs)       all of the given expressions are pairwise distinct
    And(a, b, ...)      conjunction
    Or(a, b, ...)       disjunction
    Not(a)              negation
    sat, unsat          result possibilities

Bit vectors support `+`, `-`, `*`, `==`, `!=` natively as Python operators.
`s.model()` returns a model `m`; for a bit vector `v`, `m[v].as_long()`
gives its Python int value.

You also have access to the custom solver below. Use it exactly like
Z3's Solver — `s.add(...)`, `s.check()`, `s.model()`.

    BitBlastSolver()    a Solver tuned for bit-vector queries

### Hint

A naive reference encoding is provided as `solve_magic_square_naive`
below. It uses Int variables and the built-in Distinct predicate, and
times out for N > 4 on most machines. See `ca03.md` for which three
changes will make it fast enough for N <= 8 — the changes are listed
explicitly in the spec.
"""
from bit_blast_solver import BitBlastSolver

from z3 import BitVec, BitVecVal, ULE, ULT, UGE, UGT, Distinct, And, Or, Not, sat, unsat


def solve_magic_square_naive(N):
    """Slow reference encoding using Int + Distinct. Times out for N > 4."""
    from z3 import Int, Solver, Distinct, sat
    s = Solver()
    cells = [[Int(f"c_{r}_{c}") for c in range(N)] for r in range(N)]
    flat = [c for row in cells for c in row]
    for v in flat:
        s.add(1 <= v, v <= N * N)
    s.add(Distinct(*flat))
    T = N * (N * N + 1) // 2
    for r in range(N):
        s.add(sum(cells[r]) == T)
    for c in range(N):
        s.add(sum(cells[r][c] for r in range(N)) == T)
    s.add(sum(cells[i][i] for i in range(N)) == T)
    s.add(sum(cells[i][N - 1 - i] for i in range(N)) == T)
    if s.check() != sat:
        return None
    m = s.model()
    return [[m.evaluate(cells[r][c]).as_long() for c in range(N)] for r in range(N)]


def solve_magic_square(N):
    # TODO
    raise NotImplementedError

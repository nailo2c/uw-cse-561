"""Option 2 — Sliding puzzle solver.

The sliding puzzle is played on an N-by-N grid containing tiles
1, 2, ..., N**2 - 1 and one empty cell, called the *blank* (encoded as
0). A move slides any tile that shares an edge with the blank into the
blank's position; equivalently, the blank swaps with one of its (up to
four) neighbors. Decide whether `goal` is reachable from `start` in
exactly K moves, and if so return the sequence of blank positions.

### Input / Output

    solve_sliding_puzzle(N, start, goal, K) -> list[int] | None

    N     : int — board side length. The board is N-by-N.
    start : list of length N**2, row-major. Each entry is a tile id in
            0..N**2-1, with 0 representing the blank. Every tile id
            appears exactly once.
    goal  : list of length N**2, row-major, in the same format as
            `start`.
    K     : int — exact number of moves to use (K >= 0).

    Return a list of length K+1 giving the blank's position (a flat
    row-major index, 0..N**2-1) at each step, starting with the blank
    in `start` and ending with the blank in `goal`. Return None if no
    sequence of exactly K moves takes `start` to `goal`.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    Array(name, dom, rng)   fresh array variable
    IntSort()               the sort of integers
    Int(name)               fresh integer variable
    Store(arr, i, v)        array `arr` updated so index i maps to v
    Select(arr, i)          read at index i
    And(a, b, ...)          conjunction
    Or(a, b, ...)           disjunction
    Not(a)                  negation
    If(c, a, b)             SMT-level if-then-else expression
    Solver()                supports s.add(...), s.check(), s.model()
    sat, unsat              result possibilities

Z3 expressions support `+`, `-`, `*`, `==`, `!=`, `<`, `<=`, `>`, `>=`
natively as Python operators. For an integer variable `v`,
`m.evaluate(v).as_long()` returns its Python int value in model `m`.

### Hint

The spec hint suggests representing each board state as a Z3 Array from
cell index to tile, and using `Store`/`Select` at a *symbolic* index
(i.e. an index that's itself a Z3 variable, not a Python integer) to
express a move.
"""
from z3 import Array, IntSort, Int, Store, Select, And, Or, Not, If, Solver, sat, unsat


def solve_sliding_puzzle(N, start, goal, K):
    # TODO
    raise NotImplementedError

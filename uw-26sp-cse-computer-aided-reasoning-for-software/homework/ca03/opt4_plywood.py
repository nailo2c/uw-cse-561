"""Option 4 — Cutting plywood (LRA + Bool).

You have a single piece of plywood of width `W` and height `H`, and a
list of rectangular `pieces` of various real-valued sizes `(w, h)`.
Pieces cannot rotate. Decide whether all pieces can be cut from the
board without overlapping; if so, return their bottom-left corners.

### Input / Output

    cut_plywood(W, H, pieces) -> list[tuple[float, float]] | None

    W, H    : positive reals — board width and height.
    pieces  : list of `(w, h)` real tuples — piece dimensions.

    Return a list of `(x, y)` bottom-left coordinates (one per piece,
    in input order) such that every piece fits on the board and no
    two pieces overlap. Return None if no valid layout exists.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    Real(name)          fresh real-valued variable, give string as name
    Or(a, b, ...)       disjunction
    And(a, b, ...)      conjunction
    Not(a)              negation
    Implies(a, b)       logical implication
    If(c, a, b)          SMT-level if-then-else expression
    Solver()            supports s.add(...), s.check(), s.model()
    sat, unsat          result possibilities

Z3 expressions support `+`, `-`, `*`, `==`, `!=`, `<`, `<=`, `>`, `>=`
natively as Python operators.

To turn a Z3 rational model value `m[x]` into a Python float, use
`float(m[x].as_fraction())`.
"""
from z3 import Real, Or, And, Not, Implies, If, Solver, sat, unsat


def cut_plywood(W, H, pieces):
    s = Solver()

    # init
    xs = []
    ys = []
    for i in range(len(pieces)):
        xs.append(Real(f"x_{i}"))
        ys.append(Real(f"y_{i}"))

    # add basic constraints, make sure every piece is in board
    for i, (w, h) in enumerate(pieces):
        s.add(xs[i] >= 0)
        s.add(ys[i] >= 0)
        s.add(xs[i] + w <= W)
        s.add(ys[i] + h <= H)

    # list all conbination
    pairs = []
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            pairs.append((i, j))

    for i, j in pairs:
        wi, hi = pieces[i]
        wj, hj = pieces[j]
        s.add(Or(
            xs[i] + wi <= xs[j],
            xs[j] + wj <= xs[i],
            ys[i] + hi <= ys[j],
            ys[j] + hj <= ys[i],
        ))

    if s.check() == sat:
        m = s.model()
        sol = []
        for i in range(len(pieces)):
            x = float(m[xs[i]].as_fraction())
            y = float(m[ys[i]].as_fraction())
            sol.append((x, y))
        return sol
    return None

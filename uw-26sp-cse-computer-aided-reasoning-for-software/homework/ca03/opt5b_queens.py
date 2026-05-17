"""Option 5b — Queens with regions, three encodings.

In ca02 you wrote a SAT-based solver for the regional N-queens problem:
place exactly one queen in each of N given regions of an N x N board
with no two queens attacking each other. In ca03 you encode the same
problem in two more theories and compare them experimentally:

    SAT      (provided)         1 Boolean per cell.
    SMT(LIA) (your task)        1 (r, c) Int pair per region; native
                                Or for region membership, native !=
                                for non-attack.
    Pure LIA (your task)        1 (r, c) Int pair per region, but the
                                disjunctive structure is flattened
                                with 0-1 indicators and big-M — no
                                Or, Not, !=, If, Implies, Distinct,
                                And.

See `ca03.md` for the full problem statement and the SAT-vs-LIA-vs-SMT
motivation. The deliverable for this option also includes a runtime
chart PDF — generate it from the smoke-test timings below (or your
own larger sweep) once the three encodings work.

### Input / Output (all three functions)

    solve_queens_*(n, regions) -> list[tuple[int, int]] | None

    n       : int
    regions : list of N regions; each region is a list of (r, c) cells.

    Return a length-N list of (row, col) placements, one per region,
    with no two queens attacking each other, or None if no such
    placement exists.

### Z3 primitives you may use

You may use the primitives imported below.

    Bool(name)          fresh Boolean variable
    Int(name)           fresh integer variable
    Or(a, b, ...)       disjunction
    And(a, b, ...)      conjunction
    Not(a)              negation
    Implies(a, b)       logical implication
    If(c, a, b)         SMT-level if-then-else
    Sum(*xs)            arithmetic sum of integer expressions
    Distinct(*xs)       all of the given expressions are pairwise distinct
    Solver()            supports s.add(...), s.check(), s.model()
    sat, unsat          result possibilities
    is_true(expr)       True iff `expr` is the Z3 constant True; apply
                        to `m.evaluate(bool_var)` to read a Bool model
                        value as a Python bool

For SMT(LIA): all of the above are fair game. For pure LIA: avoid
`Or`, `Not`, `If`, `Implies`, `Distinct`, and `!=` on Z3 expressions —
those are exactly the constructs the flattening exercise removes.

Integer expressions support `+`, `-`, `*`, `==`, `<=`, `>=` natively
as Python operators. For an integer expression `e`,
`m.evaluate(e).as_long()` returns its Python int value in model `m`.

### Hint

See `ca03.md` for the SMT vs LIA framing. As the spec notes, Hint 1
from Option 5a (the Tseitin-style flattening) carries over to the
pure-LIA encoding here; Hint 2 is more than you need.
"""

from itertools import combinations

from z3 import Bool, Int, Or, And, Not, Implies, If, Sum, Distinct, Solver, sat, unsat, is_true


def solve_queens_sat(n, regions):
    if len(regions) != n:
        return None
    s = Solver()
    x = [[Bool(f"x_{r}_{c}") for c in range(n)] for r in range(n)]

    def at_most_one(cells):
        lits = [x[r][c] for (r, c) in cells]
        for a, b in combinations(lits, 2):
            s.add(Or(Not(a), Not(b)))

    for region in regions:
        lits = [x[r][c] for (r, c) in region]
        s.add(Or(*lits))
        for a, b in combinations(lits, 2):
            s.add(Or(Not(a), Not(b)))

    for r in range(n):
        at_most_one([(r, c) for c in range(n)])
    for c in range(n):
        at_most_one([(r, c) for r in range(n)])
    for d in range(-(n - 1), n):
        at_most_one([(r, c) for r in range(n) for c in range(n) if r - c == d])
    for d in range(2 * n - 1):
        at_most_one([(r, c) for r in range(n) for c in range(n) if r + c == d])

    if s.check() != sat:
        return None
    m = s.model()
    placement = []
    for region in regions:
        for (r, c) in region:
            if is_true(m.evaluate(x[r][c])):
                placement.append((r, c))
                break
    return placement


def solve_queens_smt(n, regions):
    # TODO
    raise NotImplementedError


def solve_queens_lia(n, regions):
    # TODO
    raise NotImplementedError

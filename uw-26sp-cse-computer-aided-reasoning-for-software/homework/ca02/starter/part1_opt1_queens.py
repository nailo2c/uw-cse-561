"""Part 1 Option 1 — Queens with regions.

Generalized N-queens: the N×N chessboard is partitioned into N regions.
Place exactly one queen in each region such that no two queens attack
each other (no shared row, column, or diagonal).

### Input / Output

    solve_queens(n, regions) -> list[tuple[int, int]] | None

    n        : int — board size. The board is n-by-n, 0-indexed.
    regions  : list of exactly `n` regions. Each region is a list of
               (row, col) tuples. The regions partition the board:
               every cell appears in exactly one region, and the union
               of all regions is the full n×n board.

    Return a list of `n` (row, col) positions — one per region, in the
    same order as the input `regions` — such that no two queens attack
    each other. Return None if no such placement exists.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    Bool(name)          fresh Boolean variable, give string as name
    Or(a, b, ...)       disjunction (any number of boolean expressions)
    Not(a)              negation
    Solver()            supports the following operations:
                            s.add(formula): add all constraints like this
                            s.check(): returns sat or unsat
                            s.model(): if sat, gets a solution
    sat, unsat          result possibilities
    is_true(expr)       True iff `expr` is the Z3 constant True;
                        apply to m.evaluate(bool_var) to get a Python bool

### Hint

It may be helpful to have a helper function that adds constraints to encode
the statement "At most one of vars x_1, x_2, ..., x_n are true". The
function `itertools.combinations` may be helpful for this. 
"""
from itertools import combinations

from z3 import Bool, Or, Not, Solver, sat, unsat, is_true


def _attacks(p, q):
    (r1, c1), (r2, c2) = p, q
    return r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)


def solve_queens(n, regions):
    # In ex1, n=4 &
    # regions = [
    #     [(0, 0)], 
    #     [(1, 1)], 
    #     [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0)], 
    #     [(2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    # ]
    s = Solver()

    # Build a boolean board to record whether queen exist or not
    board_idx_flat = []
    for r in range(n):
        for c in range(n):
            board_idx_flat.append((r, c))

    q = [[Bool(f'q_{r}_{c}') for c in range(n)]
         for r in range(n)]

    # Every region only has one queen
    for region in regions:
        # at least one
        s.add(Or([q[r][c] for (r, c) in region]))

        # at most one
        for (r1, c1), (r2, c2) in combinations(region, 2):
            s.add(Or(Not(q[r1][c1]), Not(q[r2][c2])))

    # attack constraints
    for (r1, c1), (r2, c2) in combinations(board_idx_flat, 2):
        if _attacks((r1, c1), (r2, c2)):
            s.add(Or(Not(q[r1][c1]), Not(q[r2][c2])))

    if s.check() == sat:
        m = s.model()
        placement = []
        for region in regions:
            for (r, c) in region:
                if is_true(m.evaluate(q[r][c])):
                    placement.append((r, c))
                    break
        return placement
    return None


def print_placement(n, placement):
    """Pretty-print an n-by-n board with queens at the given positions."""
    queens = set(placement)
    for r in range(n):
        row = " ".join("Q" if (r, c) in queens else "." for c in range(n))
        print(row)


if __name__ == "__main__":
    # ---- Example 1 -----------------------------------------------------
    # 4x4 board. Two single-cell regions at (0,0) and (1,1); the
    # remaining 14 cells are split into two 7-cell regions. Any queen
    # in the first region attacks any queen in the second, so UNSAT.
    n = 4
    r1 = [(0, 0)]
    r2 = [(1, 1)]
    rest = [(r, c) for r in range(n) for c in range(n) if (r, c) not in r1 + r2]
    regions = [r1, r2, rest[:7], rest[7:]]
    assert solve_queens(n, regions) is None

    # ---- Example 2 -----------------------------------------------------
    # 3x3 board, rows-as-regions. Classically UNSAT — there is no way
    # to place three non-attacking queens on a 3x3 board.
    regions = [[(r, c) for c in range(3)] for r in range(3)]
    assert solve_queens(3, regions) is None

    # ---- Example 3 -----------------------------------------------------
    # Classic 8-queens with rows-as-regions: exactly one queen per row,
    # no two queens attacking each other.
    n = 8
    regions = [[(r, c) for c in range(n)] for r in range(n)]
    placement = solve_queens(n, regions)
    assert placement is not None and len(placement) == n
    region_sets = [set(region) for region in regions]
    for i, (r, c) in enumerate(placement):
        assert (r, c) in region_sets[i]
    for i in range(n):
        for j in range(i + 1, n):
            assert not _attacks(placement[i], placement[j])
    print("8-queens:", placement)
    print_placement(n, placement)

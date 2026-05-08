"""
Studio Exercise 1: Exploring sudoku with the solver.

The SMT encoding from Practice solved one puzzle and stopped.
Today we use the same encoding to answer richer questions. How
many solutions does a puzzle have? When a puzzle is
under-determined, what is the smallest set of givens that makes
it unique?

Three exercises.

  Exercise 1 (warm-up). Count all solutions of the Practice
    puzzle using a blocking clause loop.

  Exercise 2 (experiment). Remove a few givens and count
    again. Watch the count grow.

  Exercise 3 (design). Given an under-determined puzzle, find
    extra givens that restore uniqueness.

Stuck? A solution file is next to this one at
01-sudoku-explore-soln.py. Try each exercise yourself first.
"""

from z3 import Int, Solver, Distinct, Or, sat


# --- Board I/O (given) -------------------------------------------

def parse_board(text, n):
    size = n * n
    width = len(str(size))
    blank = '.' * width
    board = []
    for line in text.strip().splitlines():
        tokens = line.split()
        row = [0 if tok == blank else int(tok) for tok in tokens]
        board.append(row)
    return board


def format_board(board, n):
    size = n * n
    width = len(str(size))
    blank = '.' * width
    lines = []
    for row in board:
        tokens = [blank if v == 0 else str(v).zfill(width) for v in row]
        lines.append(' '.join(tokens))
    return '\n'.join(lines)


# --- Encoding (given, same as Practice) --------------------------

def make_solver(board, n):
    """Build a Solver loaded with sudoku constraints for `board`.

    Returns (solver, cells) so callers can add more constraints
    and inspect models."""
    size = n * n
    s = Solver()
    cells = [[Int(f'c_{r}_{c}') for c in range(size)]
             for r in range(size)]
    for r in range(size):
        for c in range(size):
            s.add(cells[r][c] >= 1)
            s.add(cells[r][c] <= size)
            if board[r][c] != 0:
                s.add(cells[r][c] == board[r][c])
    for r in range(size):
        s.add(Distinct(cells[r]))
    for c in range(size):
        s.add(Distinct([cells[r][c] for r in range(size)]))
    for br in range(n):
        for bc in range(n):
            block = [cells[br * n + r][bc * n + c]
                     for r in range(n) for c in range(n)]
            s.add(Distinct(block))
    return s, cells


# --- Exercise 1: Count all solutions -----------------------------
#
# The Practice demo called s.check() once and stopped. To find
# ALL solutions, we use BLOCKING CLAUSE enumeration:
#
#   1. s.check() -> get one solution (a model m).
#   2. Build a clause saying "at least one cell must differ
#      from this model." That's the BLOCKING CLAUSE.
#   3. s.add(blocking clause).
#   4. s.check() again -> either a new solution, or unsat.
#   5. Repeat.
#
# Each blocking clause cuts the current model out of the search
# space. When s.check() returns unsat, the space is empty and
# we have seen every solution.

def count_solutions(board, n, limit=1000):
    """Count solutions of `board` by blocking-clause enumeration."""
    s, cells = make_solver(board, n)
    size = n * n
    count = 0
    while s.check() == sat and count < limit:
        m = s.model()
        count += 1

        # TODO (Exercise 1): build the blocking clause.
        #
        # Using Or and a list comprehension, assert that AT LEAST
        # ONE cell differs from its value in the current model.
        # The value of cell (r, c) in the model is retrieved via:
        #     m[cells[r][c]].as_long()
        #
        # Replace the None below with your expression.
        blocking = Or([
            cells[r][c] != m[cells[r][c]].as_long()
            for r in range(size) for c in range(size)
        ])

        s.add(blocking)
    return count


# --- Exercise 3: Restore uniqueness ------------------------------
#
# Given an under-determined puzzle (multiple solutions), find a
# small set of extra givens that restore uniqueness. Approach:
#
#   while the puzzle has more than one solution:
#     1. Enumerate two distinct solutions m1 and m2.
#     2. Find a blank cell where m1 and m2 disagree.
#     3. Pin that cell to the value from m1 (or m2).
#
# Each pin eliminates at least one solution, so the loop
# terminates. This is a greedy search: it finds an extension,
# not necessarily the smallest one. Finding the smallest is
# harder.

def restore_uniqueness(board, n):
    """Add givens until the puzzle has a unique solution.

    Returns a new board with extra givens filled in."""
    size = n * n
    board = [row[:] for row in board]

    while True:
        s, cells = make_solver(board, n)

        # First solution
        if s.check() != sat:
            raise ValueError("Puzzle has no solution")
        m1 = s.model()

        # Block m1 and ask for a second solution
        blocking = Or([
            cells[r][c] != m1[cells[r][c]].as_long()
            for r in range(size) for c in range(size)
        ])
        s.add(blocking)

        # No second solution => unique
        if s.check() != sat:
            return board
        m2 = s.model()
        pinned = False
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:
                    v1 = m1[cells[r][c]].as_long()
                    v2 = m2[cells[r][c]].as_long()
                    if v1 != v2:
                        board[r][c] = v1
                        pinned = True
                        break
            if pinned:
                break

    return board


# --- Puzzles -----------------------------------------------------

PRACTICE_PUZZLE = """\
5 3 . . 7 . . . .
6 . . 1 9 5 . . .
. 9 8 . . . . 6 .
8 . . . 6 . . . 3
4 . . 8 . 3 . . 1
7 . . . 2 . . . 6
. 6 . . . . 2 8 .
. . . 4 1 9 . . 5
. . . . 8 . . 7 9"""

# Practice puzzle with three givens removed.
# This one is under-determined: more than one solution.
UNDER_DETERMINED = """\
. . . . 7 . . . .
6 . . . 9 5 . . .
. 9 8 . . . . 6 .
8 . . . 6 . . . 3
4 . . 8 . 3 . . 1
7 . . . 2 . . . 6
. 6 . . . . 2 8 .
. . . 4 1 9 . . 5
. . . . 8 . . 7 9"""


if __name__ == '__main__':
    n = 3

    # --- Exercise 1 ----------------------------------------------
    # Expected: Practice puzzle has 1 solution.
    print("=== Exercise 1: Count solutions of the Practice puzzle ===")
    board = parse_board(PRACTICE_PUZZLE, n)
    count = count_solutions(board, n)
    print(f"  {count} solution(s)")
    print()

    # --- Exercise 2 ----------------------------------------------
    # Try editing UNDER_DETERMINED above. Remove or restore givens,
    # re-run, observe. How few givens can you have before the
    # count gets huge?
    #
    # Aside on encoding. Instead of editing the data, try commenting
    # out one of the Distinct() calls inside make_solver and re-run.
    # The count explodes even more dramatically. The "all different"
    # constraint might feel implied by the other rules. It is not.
    # Removing a redundant-looking constraint can silently give you
    # a correct answer to the wrong question.
    print("=== Exercise 2: Count solutions with fewer givens ===")
    board = parse_board(UNDER_DETERMINED, n)
    count = count_solutions(board, n, limit=200)
    capped = " (capped at 200)" if count == 200 else ""
    print(f"  {count} solution(s){capped}")
    print()

    # --- Exercise 3 ----------------------------------------------
    # Uncomment the lines below once you have implemented
    # restore_uniqueness.
    print("=== Exercise 3: Restore uniqueness ===")
    size = n * n
    board = parse_board(UNDER_DETERMINED, n)
    new_board = restore_uniqueness(board, n)
    extras = sum(
        1 for r in range(size) for c in range(size)
        if new_board[r][c] != 0 and board[r][c] == 0
    )
    print(f"  Added {extras} given(s) to reach uniqueness.")
    print(f"  New count: {count_solutions(new_board, n)} solution(s)")
    print()
    print("  Augmented puzzle:")
    for line in format_board(new_board, n).splitlines():
        print(f"    {line}")
    # print("  (not yet implemented)")

"""Part 1 Option 2 — Event scheduling.

Assign each of `n_events` events to one of `n_blocks` time blocks such
that no participant has two of their desired events in the same block.

### Input / Output

    schedule(n_participants, n_events, n_blocks, prefs) -> list[int] | None

    n_participants : int
    n_events       : int
    n_blocks       : int
    prefs          : list of length `n_participants`. `prefs[i]` is a
                     list of distinct event indices (0..n_events-1)
                     that participant `i` wants to attend. Each
                     participant's list contains no duplicates.

    Return a list of length `n_events` where the i-th entry is the
    block index (0..n_blocks-1) assigned to event i, such that no
    participant has any conflict. Return None if no such assignment
    exists.

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

Focus first on nailing down how to encode the structure of an event schedule.
Then, add the constraints that prevent conflicts.
"""
from itertools import combinations

from z3 import Bool, Or, Not, Solver, sat, unsat, is_true


def schedule(n_participants, n_events, n_blocks, prefs):
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    def _no_conflict(assignment, prefs):
        for pref in prefs:
            seen = {}
            for e in pref:
                b = assignment[e]
                if b in seen and seen[b] != e:
                    return False
                seen[b] = e
        return True

    # ---- Example 1 -----------------------------------------------------
    # Two participants, each wants a different single event. One time
    # block suffices — neither participant has a conflict.
    prefs = [[0], [1]]
    a = schedule(2, 2, 1, prefs)
    assert a is not None and len(a) == 2 and all(b == 0 for b in a)

    # ---- Example 2 -----------------------------------------------------
    # The ca02.md example. Three participants with prefs {0,1}, {1,2},
    # {0,2}. At k=3 blocks the three events can all be scheduled
    # separately; at k=2 no assignment avoids every conflict.
    prefs = [[0, 1], [1, 2], [0, 2]]
    a = schedule(3, 3, 3, prefs)
    assert a is not None and _no_conflict(a, prefs)
    assert len(set(a)) == 3
    assert schedule(3, 3, 2, prefs) is None

    # ---- Example 3 -----------------------------------------------------
    # 14 participants, 10 events, 4 blocks. Preference sizes vary from
    # 2 to 4 events; several participants want 3 or more events, so
    # the solver has to interleave assignments across all four blocks.
    prefs = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 2, 3],
        [0, 5, 6, 7],
        [0, 1, 2],
        [5, 6, 7],
        [3, 4, 5],
        [8, 9, 6],
        [1, 2, 3],
        [0, 7], [1, 6], [2, 5], [3, 8], [4, 9],
    ]
    a = schedule(14, 10, 4, prefs)
    assert a is not None and _no_conflict(a, prefs)
    print("14-participant schedule:", a)

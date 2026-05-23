"""Part A — Count groups.

Input:  A 32-bit signed integer a with a >= 0, thought of in binary.

Output: A *group* is two or more consecutive 1's. Return the number
        of groups.

The implementation of `count_groups` below has a bug.

### Your tasks

1. Add annotations (`assume`, `assert`, optionally `havoc`) to
   `count_groups` so the L07 symbolic execution engine finds a
   counterexample. You may introduce new variables that are used
   only in annotations.

2. Set `counterexample` to a non-negative integer that, interpreted
   as a 32-bit signed integer, demonstrates the bug.

3. Adjust `MAX_DEPTH` and `BITWIDTH` if needed. The autograder uses
   these values when it calls `check_assertions`. 
"""

from engine.mini_imp import assume, havoc


MAX_DEPTH = 4
BITWIDTH = 4


def count_groups(x):
    assume(x >= 0)

    cnt = 0
    in_run = 0
    run_len = 0
    i = 0
    while i < BITWIDTH:
        bit = (x >> i) & 1
        if bit == 1:
            in_run = 1
            run_len = run_len + 1
        else:
            if in_run == 1:
                in_run = 0
                if run_len >= 2:
                    cnt = cnt + 1
                    run_len = 0
        i = i + 1

    assert not ((x & (x >> 1)) == 0) or cnt == 0

    return cnt


counterexample = 5

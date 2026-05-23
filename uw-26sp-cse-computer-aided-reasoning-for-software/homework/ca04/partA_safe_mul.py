"""Part A — Safe multiplication.

Input:  Two 32-bit signed integers a and b.

Output: If a * b fits in a 32-bit signed integer without overflow,
        return a * b, otherwise return 0.

The implementation of `safe_mul` below has a bug.

### Your tasks

1. Add annotations (`assume`, `assert`, optionally `havoc`) to
   `safe_mul` so the L07 symbolic execution engine finds a
   counterexample. You may introduce new variables that are used only
   in annotations.

2. Set `counterexample` to a tuple `(a, b)` of 32-bit signed integers
   that demonstrate the bug.

3. Adjust `MAX_DEPTH` and `BITWIDTH` if needed. The autograder uses
   these values when it calls `check_assertions`. 
"""

from engine.mini_imp import assume, havoc


MAX_DEPTH = 1
BITWIDTH = 32


def safe_mul(a, b):
    if a == 0 or b == 0:
        result = 0
    else:
        c = a * b
        if c // a != b:
            result = 0
        else:
            result = c

    # assert not (not (a == 0 or b == 0) and c // a != b) or result != 0
    # assert (a != 0 and b != 0) or (result != 0 and result != a * b)
    assert not (a == -1 and b < 0) or result >= 0

    return result


counterexample = (-1, -2147483648)

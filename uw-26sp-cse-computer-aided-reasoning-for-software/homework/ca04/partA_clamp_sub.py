"""Part A — Clamped subtraction.

Input:  Two 32-bit signed integers a and b.

Output: If a - b fits in a 32-bit signed integer without overflow,
        return a - b, otherwise return INT_MAX or INT_MIN depending on
        the direction of overflow.

The implementation of `clamp_sub` below has a bug.

### Your tasks

1. Add annotations (`assume`, `assert`, optionally `havoc`) to
   `clamp_sub` so the L07 symbolic execution engine finds a
   counterexample. You may introduce new variables that are used only
   in annotations.

2. Set `counterexample` to a tuple `(a, b)` of 32-bit signed integers
   that demonstrate the bug.

3. Adjust `MAX_DEPTH` and `BITWIDTH` if needed. The autograder uses
   these values when it calls `check_assertions`.
"""

from engine.mini_imp import INT_MAX, INT_MIN, assume, havoc


MAX_DEPTH = 1
BITWIDTH = 32


def clamp_sub(a, b):
    if a > 0 and b < 0 and a - b < 0:
        result = INT_MAX
    elif a < 0 and b > 0 and a - b > 0:
        result = INT_MIN
    else:
        result = a - b

    # e.g. if a = 2**31 -1
    #         b = -1
    #      then result must greater than zero, ideally
    #      however, this assert can help us to catch the counterexample
    assert not (a >= 0 and b < 0) or result > 0

    return result


counterexample = (0, -2147483648)

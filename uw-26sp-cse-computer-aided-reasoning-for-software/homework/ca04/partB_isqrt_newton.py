"""Part B — Newton's isqrt.

Input:  An integer n with 1 <= n <= 10**9.

Output: floor(sqrt(n)).

### Your task

Fill in the `invariant(...)` call so the L08 WP engine discharges
every VC (loop entry, preservation, sufficiency). You may not modify
the precondition, the postcondition, or any of the function's
executable code.
"""

from engine.mini_imp import assume, havoc, invariant


def isqrt_newton(n):
    assume(n >= 1)
    assume(n <= 1_000_000_000)

    x = n
    y = (x + 1) // 2
    while y < x:
        invariant(
            n >= 1
            and x >= 1
            and y >= 1
            and y == (x + n // x) // 2
            and (x + 1) * (x + 1) > n
        )
        x = y
        y = (x + n // x) // 2

    assert x * x <= n
    assert (x + 1) * (x + 1) > n
    return x

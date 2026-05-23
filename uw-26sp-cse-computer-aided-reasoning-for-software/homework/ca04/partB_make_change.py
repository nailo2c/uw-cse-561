"""Part B — Make change.

Input:  A non-negative integer `amount` in cents.

Output: A minimal-coin breakdown `(quarters, dimes, nickels, pennies)`
        of `amount` using US denominations, such that the number of
        larger denominations is maximized.

### Your task

Fill in each `invariant(...)` call so the L08 WP engine discharges
every VC (loop entry, preservation, sufficiency). You may not modify
the precondition, the postcondition, or any of the function's
executable code.
"""

from engine.mini_imp import assume, havoc, invariant


def make_change(amount):
    assume(amount >= 0)

    orig = amount
    quarters = 0
    dimes = 0
    nickels = 0
    pennies = 0

    while amount >= 25:
        invariant(
            amount + 25 * quarters == orig
            and dimes == 0
            and nickels == 0
            and pennies == 0
            and amount >= 0
        )
        amount = amount - 25
        quarters = quarters + 1

    while amount >= 10:
        invariant(
            amount + 25 * quarters + 10 * dimes == orig
            and nickels == 0
            and pennies == 0
            and amount >= 0
        )
        amount = amount - 10
        dimes = dimes + 1

    while amount >= 5:
        invariant(
            amount + 25 * quarters + 10 * dimes + 5 * nickels == orig
            and pennies == 0
            and amount >= 0
        )
        amount = amount - 5
        nickels = nickels + 1

    while amount >= 1:
        invariant(
            amount + 25 * quarters + 10 * dimes + 5 * nickels + pennies == orig
            and amount >= 0
        )
        amount = amount - 1
        pennies = pennies + 1

    assert 25 * quarters + 10 * dimes + 5 * nickels + pennies == orig
    assert dimes < 3 and nickels < 2 and pennies < 5
    return (quarters, dimes, nickels, pennies)

"""mini_imp.py — runtime markers for mini IMP (ca04 edition).

mini IMP is the subset of Python that our symbolic-execution engine
understands. The non-Python primitives are defined here as runtime
functions so that mini-IMP programs run normally under CPython. The
SE engine recognizes calls to these by name and treats them as IVL
primitives:

    assume(C)       extend the path condition with C
    havoc(x)        introduce a fresh symbolic value for variable x
    invariant(I)    declares a loop invariant (L08 marker; see note
                    below for how the L07 engine handles it)

In concrete mode, `assume` and `invariant` check their condition with
a regular `assert`, and `havoc(x)` is a no-op — x retains its runtime
value. This lets you sprinkle annotations into real code without
changing what the code computes for any well-formed input.

`havoc` and `invariant` are intentionally written to match the L08
signatures: `havoc` takes one variable per call (use multiple calls
to havoc multiple variables), and `invariant` takes one condition.
The L07 SE engine in this assignment does *bounded model checking*
(loop unrolling), not loop-cut transformation, so any `invariant(...)`
call you write is recognized but does not influence the symbolic
exploration — the engine will warn the first time it sees one. Use
the L08 WP engine if you want invariant-based verification.
"""


# INT_MIN / INT_MAX / BITWIDTH — concretely the 32-bit values. The SE
# engine rebinds these to the current bitwidth when it sees them by
# name in source code, so a program that uses BITWIDTH (e.g. to bound
# a bit-scan loop) works correctly at 8, 16, or 32 bits.
INT_MAX = 2147483647
INT_MIN = -2147483648
BITWIDTH = 32


def assume(condition):
    """Concrete mode: check that the condition holds.

    Symbolic mode: the SE engine intercepts this call and extends the
    path condition with the (symbolic) argument.
    """
    assert condition, "assume() condition is false in concrete mode"


def havoc(x):
    """Concrete mode: no-op. The argument keeps its value.

    Symbolic mode: the SE engine intercepts the call and replaces the
    named variable's symbolic value with a fresh symbol, modelling
    demonic non-determinism. Takes exactly one variable per call.
    """
    return None


def invariant(condition):
    """Concrete mode: check that the invariant holds at this point.

    Symbolic mode: this is the L08 marker for loop invariants and is
    not used by the L07 SE engine. The L07 engine will issue a warning
    on the first call it sees and otherwise treat it as a no-op.
    """
    assert condition, "invariant() condition is false in concrete mode"

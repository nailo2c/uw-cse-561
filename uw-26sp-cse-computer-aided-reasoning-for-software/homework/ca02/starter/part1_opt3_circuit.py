"""Part 1 Option 3 — Circuit equivalence via the Tseitin transformation.

You will:

1. Implement `tseitin_encode(circuit, solver)` — add CNF clauses to
   `solver` that encode the circuit, and return the Z3 Bool
   representing the circuit's output.
2. Implement `equivalent(c1, c2)` — use `tseitin_encode` on both
   circuits, plus additional clause(s) to decide equivalence.

### Input schema

Circuits are built from the dataclasses in `circuit.py`:

    VAR(name)          input variable
    NOT(arg)           unary not
    AND(left, right)   binary and
    OR(left, right)    binary or
    Circuit(inputs, output)
        inputs : tuple[str, ...]  — declared input variable names
        output : Gate             — the top-level gate

### Entry points

    tseitin_encode(circuit: Circuit, solver: Solver)
        Add clauses encoding `circuit` to `solver`. Return the Z3 Bool
        that represents the circuit's output.

        Every formula in `solver.assertions()` after this call must be
        a single *literal* (a Bool constant or its Not) or an *Or of
        literals*. The autograder inspects the assertion shape.

    equivalent(c1: Circuit, c2: Circuit) -> bool
        Return True iff c1 and c2 compute the same Boolean function.
        Assume c1.inputs == c2.inputs.

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

You will call `tseitin_encode` from `equivalent` twice, so make sure 
the Bool names you introduce in `tseitin_encode` are unique across both
calls. Otherwise the two encodings will collide on reused names.

Refer back to the lecture notes if you forgot how to do a Tseitin
encoding -- you may want to work it out by hand before starting to code.

A Python reminder: use `isinstance(obj, Class)` to determine if obj
is of type Class (e.g. VAR, NOT, AND, OR).
"""

from z3 import Bool, Or, Not, Solver, sat, unsat, is_true

from circuit import VAR, NOT, AND, OR, Circuit, Gate


def tseitin_encode(circuit, solver):
    # TODO
    raise NotImplementedError


def equivalent(c1, c2):
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    a, b, c = VAR("a"), VAR("b"), VAR("c")

    def xor(x, y):
        return OR(AND(x, NOT(y)), AND(NOT(x), y))

    # ---- Example 1 -----------------------------------------------------
    # AND(a, b) vs AND(a, NOT(b)). These compute different Boolean
    # functions: they disagree when a=True, b=True.
    c1 = Circuit(("a", "b"), AND(a, b))
    c2 = Circuit(("a", "b"), AND(a, NOT(b)))
    assert equivalent(c1, c2) is False

    # ---- Example 2 -----------------------------------------------------
    # De Morgan's law: NOT(a AND b) equals (NOT a) OR (NOT b).
    c1 = Circuit(("a", "b"), NOT(AND(a, b)))
    c2 = Circuit(("a", "b"), OR(NOT(a), NOT(b)))
    assert equivalent(c1, c2) is True

    # ---- Example 3 -----------------------------------------------------
    # Two 4-bit adders, compared on their carry-out bit. Both take two
    # 4-bit numbers a = a3 a2 a1 a0 and b = b3 b2 b1 b0 (LSB first, no
    # carry-in) and output the final carry-out c4 — the fifth bit of
    # the sum. They differ in how that carry is computed:
    #
    #   * c1 is a ripple-carry adder: each bit's carry-out depends on
    #     the previous bit's carry-out, forming a chain of length 4.
    #     Short to build, but slow in real hardware because each carry
    #     has to "ripple" through the chain.
    #
    #   * c2 is a carry-lookahead adder: the final carry is expanded
    #     into a flat sum-of-products over all bit positions at once,
    #     trading depth for width. In real hardware this is faster
    #     but uses more gates.
    #
    # Both use the standard generate/propagate primitives
    # g_i = a_i AND b_i and p_i = a_i OR b_i. Each circuit reuses
    # those primitives via Python references, and c2 additionally
    # reuses the p3·p2 and p3·p2·p1 prefixes across multiple terms.
    # Verifying that the two adders compute the same function is a
    # small instance of the classic hardware-equivalence question
    # "does my optimized design still match the slow, obvious one?"
    inputs = tuple(f"a{i}" for i in range(4)) + tuple(f"b{i}" for i in range(4))
    a_ = [VAR(f"a{i}") for i in range(4)]
    b_ = [VAR(f"b{i}") for i in range(4)]
    g = [AND(a_[i], b_[i]) for i in range(4)]
    p = [OR(a_[i], b_[i]) for i in range(4)]

    # Ripple: c_{i+1} = g_i OR (p_i AND c_i), with c_0 = 0 so c_1 = g_0.
    c_ripple = g[0]
    for i in range(1, 4):
        c_ripple = OR(g[i], AND(p[i], c_ripple))
    c1 = Circuit(inputs, c_ripple)

    # Carry-lookahead: c_4 = g_3 OR (p_3·g_2) OR (p_3·p_2·g_1) OR (p_3·p_2·p_1·g_0).
    p32 = AND(p[3], p[2])
    p321 = AND(p32, p[1])
    term_g3 = g[3]
    term_p3g2 = AND(p[3], g[2])
    term_p32g1 = AND(p32, g[1])
    term_p321g0 = AND(p321, g[0])
    c_cla = OR(OR(term_g3, term_p3g2), OR(term_p32g1, term_p321g0))
    c2 = Circuit(inputs, c_cla)

    assert equivalent(c1, c2) is True
    print("all circuit examples pass")

"""Part 2 Option 2 — Program equivalence under EUF axioms.

Two straight-line programs with inputs `a`, `b` use two uninterpreted
functions `f` (with one parameter) and `g` (with two parameters). 
Decide whether the two programs compute the same final value, assuming 
`f` and `g` satisfy:

    f(f(t)) == f(t)      for every term t
    g(u, v) == g(v, u)   for every pair of terms u, v

### Input / Output

    equivalent(prog1, prog2) -> bool

    prog1, prog2 : Program (see `program.py`)
        Program.body : tuple[Stmt, ...]  — straight-line statements
            F(var, arg)              var = f(arg)
            G(var, arg1, arg2)       var = g(arg1, arg2)

        Program.ret  : str           name of the returned variable

        Every `var` is fresh and unique within its program; every
        `arg` / `arg1` / `arg2` is either "a", "b", or a previously-
        defined var.

    Return True iff prog1 and prog2 return equal values for all `a`,
    `b` under the two axioms above.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    DeclareSort(name)        uninterpreted sort (i.e. type)
    Function(name, *sorts)   uninterpreted function; last sort is the
                             return sort. Examples:
                                 Function("f", U, U)     f: U → U
                                 Function("g", U, U, U)  g: U × U → U
                             Returns an object that you can call with () 
                             like a normal function. 
    Const(name, sort)        constant of the given sort
    Solver()            supports the following operations:
                            s.add(formula): add all constraints like this
                            s.check(): returns sat or unsat
                            s.model(): if sat, gets a solution
    sat, unsat          result possibilities
    is_true(expr)       True iff `expr` is the Z3 constant True;
                        apply to m.evaluate(bool_var) to get a Python bool

### Hint

The axioms we stated above are intended to be true for *every* possible term,
but basic EUF only supports axioms about specific terms. Luckily, we don't
actually need to use the axioms on all possible terms.

For example, to decide whether `f(f(f(a))) == f(f(a))` holds, we only need the
`f(f(t)) == f(t)` axiom instantiated at `t = f(a)`. It doesn't matter what happens 
when `t = f(b)` or `t = g(a, b)` — those terms never appear in the query, so 
their behavior can't affect the answer. More generally, it suffices to instantiate
each axiom once at every *subterm* that appears in either program.

Fill out the `subterms` function first, then use it in `equivalent`.

A Python reminder: use `isinstance(obj, Class)` to determine if obj
is of type Class (e.g. F, G).
"""
from z3 import DeclareSort, Function, Const, Solver, sat, unsat, is_true

from program import F, G, Program


def subterms(prog):
    """Return the set of subterm names reachable from `prog.ret`.

    A subterm is one of the inputs "a" or "b", or the `var` of a body
    statement reachable from `prog.ret` along `arg` / `arg1` / `arg2`
    dependency edges. Only names appear in the result — names
    correspond one-to-one with subterms because every `var` is fresh
    and every `arg` references an input or a previously-defined `var`.

    Example: for the program

        t1 = f(a)
        t2 = g(t1, b)
        t3 = f(t2)
        return t3

    `subterms(prog)` returns `{"a", "b", "t1", "t2", "t3"}`.

    Example (unreachable stmt pruned): for

        t1 = f(a)
        t2 = f(b)
        return t2

    `subterms(prog)` returns `{"b", "t2"}` — `"a"` and `"t1"` are
    excluded because nothing reachable from the return value uses them.
    """
    # TODO
    raise NotImplementedError


def equivalent(prog1, prog2):
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # ---- Example 1 -----------------------------------------------------
    # f(a) vs f(b). Different inputs, and no axiom relates `a` and `b`,
    # so the two programs return different values.
    p1 = Program((F("t1", "a"),), "t1")
    p2 = Program((F("t1", "b"),), "t1")
    assert equivalent(p1, p2) is False

    # ---- Example 2 -----------------------------------------------------
    # f(g(a, b)) vs f(f(g(b, a))). Swapping the arguments of g gives
    # the same value by g_commutative, and wrapping it in an extra f
    # collapses by f_idempotent, so the two programs are equivalent.
    p1 = Program((G("t1", "a", "b"), F("t2", "t1")), "t2")
    p2 = Program((G("t1", "b", "a"), F("t2", "t1"), F("t3", "t2")), "t3")
    assert equivalent(p1, p2) is True

    # ---- Example 3 -----------------------------------------------------
    # The motivating 12-line vs 6-line example from ca02.md.
    prog_1 = Program((
        F("t1",  "a"),
        G("t2",  "t1", "b"),
        F("t3",  "t2"),
        G("t4",  "a",  "b"),
        G("t5",  "t3", "t1"),
        F("t6",  "t5"),
        G("t7",  "t1", "b"),
        F("t8",  "t7"),
        F("t9",  "t5"),
        G("t10", "t8", "t1"),
        F("t11", "t10"),
        G("t12", "t9", "t11"),
    ), "t12")
    prog_2 = Program((
        F("u1", "a"),
        G("u2", "u1", "b"),
        F("u3", "u2"),
        G("u4", "u3", "u1"),
        F("u5", "u4"),
        G("u6", "u5", "u5"),
    ), "u6")
    assert equivalent(prog_1, prog_2) is True
    print("all program-equivalence examples pass")

"""Option 5a — SMT(LIA) over CNF flattened to pure LIA.

You are given an SMT(LIA) query already in CNF: a list of variables and
a list of clauses, where each clause is a disjunction of linear atoms
of the form `Σ c_i · x_i  <=  rhs`. Decide satisfiability and return
a satisfying integer assignment, but build the Z3 query in **pure LIA**
— no Boolean structure, no disequalities, no conditionals.

In particular, you may NOT use any of:

    Or, Not, If, Implies, !=, Distinct, And

The whole point of this exercise is to encode the CNF's disjunctions
using only linear inequalities over integers. (Python's `sum`, `+`,
`-`, `*`, `<=`, `>=`, `==` on Z3 expressions are all fine.)

### Input / Output

    solve_lia_cnf(variables, cnf) -> dict[str, int] | None

    variables : list[str]
        Distinct variable names, e.g. ["x", "y", "z"].
    cnf       : list[list[Atom]]
        Outer list: AND across clauses.
        Inner list: OR across atoms within a clause.
        See `lia_cnf.py` for the `Atom` schema. Each
        `Atom(coeffs={...}, rhs=k)` represents the inequality
        `Σ c_i · x_i  <=  k` (non-strict, no negation).

    Return a `dict` mapping each variable name to an `int` value
    satisfying every clause, or `None` if the CNF is unsatisfiable.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    Int(name)           fresh integer variable
    Solver()            supports s.add(...), s.check(), s.model()
    sat, unsat          result possibilities

Integer expressions support `+`, `-`, `*`, `==`, `<=`, `>=` natively
as Python operators. Python's built-in `sum(...)` works on Z3
expressions too. For an integer expression `e`,
`m.evaluate(e).as_long()` returns its Python int value in model `m`.

### Hints

The two hints in `ca03.md` are the relevant ones for this option.
"""
from lia_cnf import Atom

from z3 import Int, Solver, sat, unsat


def solve_lia_cnf(variables, cnf):
    # TODO
    raise NotImplementedError

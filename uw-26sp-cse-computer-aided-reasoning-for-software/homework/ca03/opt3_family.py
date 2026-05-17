"""Option 3 — Family-tree puzzle (EUF + LIA).

Given a finite collection of named people, some age facts, and some
parent facts, decide whether the puzzle is consistent and, if so,
return one valid age assignment for everyone.

Every puzzle is governed by the *generation gap rule*: a parent must
be at least `GENERATION_GAP` years older than their child. This rule
applies automatically to every parent-child pair the puzzle declares
(both explicitly via "parent_of" facts and implicitly via the
"same_parent" facts, whose shared parent must still be one of the
named people).

### Input / Output

    solve_family_tree(people, age_facts, parent_facts) -> dict[str, int] | None

    people        : list of distinct name strings, e.g. ["Alice", "Bob"].
    age_facts     : list of `AgeFact` records (see `family_tree.py`).
    parent_facts  : list of `ParentFact` records (see `family_tree.py`).

    Return a dict mapping every name in `people` to a non-negative
    integer age that satisfies every fact and the generation gap rule.
    Return None if no such assignment exists.

### Z3 primitives you may use

You may only use the Z3 primitives imported below. Do not use anything
else from Z3.

    DeclareSort(name)        a fresh uninterpreted sort (i.e. type)
    Const(name, sort)        a constant of the given sort
    Function(name, *sorts)   an uninterpreted function from the first
                             sorts to the last. Returns a callable
                             that you can apply with `()` like a
                             normal Python function.
    IntSort()                the integer sort
    Distinct(*xs)            all of the given expressions are pairwise distinct
    Or(a, b, ...)            disjunction
    And(a, b, ...)           conjunction
    Not(a)                   negation
    Implies(a, b)            logical implication
    If(c, a, b)              SMT-level if-then-else expression
    Solver()                 supports s.add(...), s.check(), s.model()
    sat, unsat               result possibilities

Z3 expressions support `+`, `-`, `*`, `==`, `!=`, `<`, `<=`, `>`, `>=`
natively as Python operators. For an integer expression `e`,
`m.evaluate(e).as_long()` returns its Python int value in model `m`.
"""
from family_tree import AgeFact, ParentFact, GENERATION_GAP

from z3 import (
    DeclareSort, Const, Function, IntSort, Distinct,
    Or, And, Not, Implies, If, Solver, sat, unsat,
)


def solve_family_tree(people, age_facts, parent_facts):
    # return type: dict[str, int] | None
    # e.g. {"Alice": 8, "Bob": 26, "Carol": 10}

    s = Solver()
    Person = DeclareSort("Person")
    people_consts = {}
    for p in people:
        people_consts[p] = Const(p, Person)
    s.add(Distinct(*people_consts.values()))

    age_func = Function("age", Person, IntSort())
    for p in people:
        s.add(age_func(people_consts[p]) >= 0)

    for age_fact in age_facts:
        people_name = age_fact.person
        people_value = age_fact.value
        op = age_fact.op

        # handle people_value is int or string
        if isinstance(people_value, str):
            right = age_func(people_consts[people_value])
        else:
            right = people_value

        left = age_func(people_consts[people_name])

        if op == "==":
            s.add(left == right)
        elif op == "!=":
            s.add(left != right)
        elif op == "<": 
            s.add(left < right)
        elif op == "<=": 
            s.add(left <= right)
        elif op == ">":
            s.add(left > right)
        elif op == ">=":
            s.add(left >= right)
    
    for parent_fact in parent_facts:
        if parent_fact.kind == "parent_of":
            child = people_consts[parent_fact.a]
            parent = people_consts[parent_fact.b]
            s.add(age_func(parent) >= age_func(child) + GENERATION_GAP)
        elif  parent_fact.kind == "same_parent":
            child = people_consts[parent_fact.a]
            sibling = people_consts[parent_fact.b]
            candidates = []
            for p in people:
                possible_parent = people_consts[p]
                candidates.append(
                    And(
                        age_func(possible_parent) >= age_func(child) + GENERATION_GAP,
                        age_func(possible_parent) >= age_func(sibling) + GENERATION_GAP,
                    )
                )
            s.add(Or(*candidates))

    if s.check() == sat:
        m = s.model()
        sol = {}
        for p in people:
            sol[p] = m.evaluate(age_func(people_consts[p])).as_long()
        return sol
    return None

    # adhoc solution
    # sol = {}
    # max_age = 0
    # for age_fact in age_facts:
    #     if age_fact.value > max_age:
    #         max_age = age_fact.value

    #     if age_fact.op == "==":
    #         sol[age_fact.person] = age_fact.value

    # for parent_fact in parent_facts:
    #     if parent_fact.kind == "parent_of":
    #         if parent_fact.a in sol and parent_fact.b not in sol:
    #             sol[parent_fact.b] = max_age + GENERATION_GAP

    # return sol

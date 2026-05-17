"""Plain-data records for the family-tree puzzle (Option 3).

A puzzle instance is a triple (people, age_facts, parent_facts):

    people        : list of distinct name strings, e.g. ["Alice", "Bob"]
    age_facts     : list of AgeFact records
    parent_facts  : list of ParentFact records

Beyond the explicit facts, every puzzle is governed by the generation-gap
rule: a parent must be at least GENERATION_GAP years older than their
child.

### AgeFact

`AgeFact(person, op, value)` asserts an inequality on `person`'s age.
`op` is one of "==", "!=", "<", "<=", ">", ">=". `value` is either an
`int` (e.g. AgeFact("Alice", "==", 8) — "Alice is 8 years old") or
another person's name string (e.g. AgeFact("Bob", ">", "Eve") — "Bob is
older than Eve").

### ParentFact

Two shapes:

    ParentFact("parent_of", child=..., parent=...)
        — `child`'s parent is `parent`. Both are names from `people`.

    ParentFact("same_parent", a=..., b=...)
        — `a` and `b` share the same parent. The parent's identity is
          unspecified but must be one of the people in the domain.
"""
from typing import NamedTuple, Union


GENERATION_GAP = 16


class AgeFact(NamedTuple):
    person: str
    op: str
    value: Union[int, str]


class ParentFact(NamedTuple):
    kind: str           # "parent_of" or "same_parent"
    a: str              # for parent_of: the child; for same_parent: first sibling
    b: str              # for parent_of: the parent; for same_parent: second sibling

"""Plain-data records for the SMT-over-LIA → pure-LIA flattening exercise (Option 5a).

A query is a pair (variables, cnf):

    variables : list of distinct variable name strings, e.g. ["x", "y", "z"]
    cnf       : list[list[Atom]]
                outer list: AND across clauses
                inner list: OR across atoms within a clause

Each Atom represents a linear inequality "Σ c_i · x_i  <=  rhs":

    Atom(coeffs={"x": 1, "y": 1}, rhs=3)     # x + y <= 3
    Atom(coeffs={"y": 1, "z": -2}, rhs=-1)   # y - 2z <= -1
    Atom(coeffs={"z": 1}, rhs=5)             # z <= 5

The CNF for ((x + y <= 3) OR (y - 2z <= -1)) AND (z <= 5) is therefore:

    cnf = [
        [Atom({"x": 1, "y": 1}, 3), Atom({"y": 1, "z": -2}, -1)],
        [Atom({"z": 1}, 5)],
    ]

By assumption (per the spec), there is no negation, equality, or strict
inequality. All atoms are non-strict <= inequalities.
"""
from typing import NamedTuple


class Atom(NamedTuple):
    coeffs: dict      # variable name -> integer coefficient
    rhs: int          # Σ c_i · x_i <= rhs

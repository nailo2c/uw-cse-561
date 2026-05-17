"""Shared straight-line program AST for Part 2 Option 2 (program equivalence)."""
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class F:
    var: str
    arg: str


@dataclass(frozen=True)
class G:
    var: str
    arg1: str
    arg2: str


Stmt = Union[F, G]


@dataclass(frozen=True)
class Program:
    body: tuple
    ret: str

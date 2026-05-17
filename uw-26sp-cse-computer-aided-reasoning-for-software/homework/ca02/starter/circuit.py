"""Shared circuit AST for Part 1 Option 3 (circuit equivalence)."""
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class VAR:
    name: str


@dataclass(frozen=True)
class NOT:
    arg: "Gate"


@dataclass(frozen=True)
class AND:
    left: "Gate"
    right: "Gate"


@dataclass(frozen=True)
class OR:
    left: "Gate"
    right: "Gate"


Gate = Union[VAR, NOT, AND, OR]


@dataclass(frozen=True)
class Circuit:
    inputs: tuple
    output: Gate

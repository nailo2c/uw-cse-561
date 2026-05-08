"""Shared plain-data types for Part 2 Option 1 (Zebra puzzle)."""
from typing import NamedTuple


class Property(NamedTuple):
    category: str
    value: str


class Clue(NamedTuple):
    subject: Property
    target: Property
    about_roommate: bool
    subject_negated: bool
    target_negated: bool

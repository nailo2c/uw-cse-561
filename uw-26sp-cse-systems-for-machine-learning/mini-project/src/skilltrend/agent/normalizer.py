"""Deterministic alias-based skill normalizer.

The proposal explicitly says normalization should only collapse names when the
taxonomy supports it. We resolve raw skill strings to canonical names from
config/taxonomy.yaml; anything not in the taxonomy keeps its original (just
title-cased) form so we don't accidentally hide emerging skills."""
from __future__ import annotations

from functools import lru_cache

from ..settings import settings


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    """Map lower-cased alias -> canonical name."""
    canonical = settings.load_taxonomy()
    out: dict[str, str] = {}
    for canon, aliases in canonical.items():
        out[canon.lower()] = canon
        for a in aliases or []:
            out[a.strip().lower()] = canon
    return out


def normalize(name: str) -> str:
    if not name:
        return name
    idx = _alias_index()
    key = name.strip().lower()
    if key in idx:
        return idx[key]
    # Conservative cleanup: collapse whitespace, preserve user casing.
    return " ".join(name.split())

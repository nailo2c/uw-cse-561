"""Skill -> category lookup.

Reads config/skill_categories.yaml once and exposes:

    category_for("Python")        -> "Languages"
    category_for("RandomNewTool") -> "Uncategorized"
    all_categories()              -> list of canonical category names

Matching is case-insensitive, whitespace-collapsed. Skills not present in
the YAML fall into "Uncategorized" — this is deliberate so emerging
technologies show up in the trend without forcing a YAML edit first.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from .settings import settings

UNCATEGORIZED = "Uncategorized"


@lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    """Map lowercased skill name -> canonical category name."""
    path = settings.config_dir / "skill_categories.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    out: dict[str, str] = {}
    for category, skills in data.items():
        for skill in skills or []:
            key = " ".join(str(skill).split()).strip().lower()
            if key:
                out[key] = category
    return out


def category_for(skill_name: str) -> str:
    if not skill_name:
        return UNCATEGORIZED
    key = " ".join(str(skill_name).split()).strip().lower()
    return _index().get(key, UNCATEGORIZED)


@lru_cache(maxsize=1)
def all_categories() -> list[str]:
    """All declared category names, plus Uncategorized at the end."""
    path = settings.config_dir / "skill_categories.yaml"
    if not path.exists():
        return [UNCATEGORIZED]
    data = yaml.safe_load(path.read_text()) or {}
    return list(data.keys()) + [UNCATEGORIZED]

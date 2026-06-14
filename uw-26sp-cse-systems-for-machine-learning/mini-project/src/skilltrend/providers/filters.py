from __future__ import annotations


def config_terms(extra: dict, key: str) -> list[str]:
    raw = extra.get(key) or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(term).strip().lower() for term in raw if str(term).strip()]


def allowed_by_terms(
    text: str,
    include_terms: list[str] | None = None,
    exclude_terms: list[str] | None = None,
) -> bool:
    haystack = str(text or "").lower()
    if exclude_terms and any(term in haystack for term in exclude_terms):
        return False
    if include_terms and not any(term in haystack for term in include_terms):
        return False
    return True

"""Human-readable explanations for a recommendation.

Every card names *which model* produced it. That matters here: the blend
deliberately degrades to content-only for the ~86% of titles with no
MovieLens join, and a missing collaborative reason must read as a
by-design outcome rather than a bug.
"""

from __future__ import annotations

import pandas as pd

MODEL_LABELS = {
    "content": "Content match",
    "cf": "Collaborative filtering",
    "popularity": "Popular",
}


def _split(value: object) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def content_reason(cand: pd.Series, seed: pd.Series | None = None,
                   profile_genres: set[str] | None = None) -> str:
    """Shared attributes with the seed title, or with the taste profile."""
    parts: list[str] = []

    if seed is not None:
        shared_genres = sorted(_split(cand.get("listed_in")) & _split(seed.get("listed_in")))
        if shared_genres:
            parts.append("Shares " + ", ".join(shared_genres[:3]))

        cand_cast = set(_split(cand.get("cast")))
        overlap = len(cand_cast & _split(seed.get("cast")))
        if overlap:
            parts.append(f"{overlap} cast member{'s' if overlap != 1 else ''} in common")

        cand_dir = _split(cand.get("director")) & _split(seed.get("director"))
        if cand_dir:
            parts.append("Same director: " + ", ".join(sorted(cand_dir)[:2]))

        if _split(cand.get("country")) & _split(seed.get("country")):
            parts.append("Same country")

        if not parts:
            parts.append("Similar description and genre mix")

    elif profile_genres:
        matches = sorted(_split(cand.get("listed_in")) & profile_genres)
        if matches:
            parts.append("Matches your taste in " + ", ".join(matches[:3]))
        else:
            parts.append("Fits the genres you picked")

    else:
        parts.append("Similar tone and genre mix")

    return " · ".join(parts)


def cf_reason(liked_title: str | None) -> str:
    if liked_title:
        clean = liked_title.rsplit(" (", 1)[0]      # strip MovieLens year suffix
        return f"Because you liked {clean}"
    return "Rated similarly by people with your taste"


def popularity_reason() -> str:
    return "Popular with MovieLens raters"


def label(kind: str) -> str:
    return MODEL_LABELS.get(kind, kind)

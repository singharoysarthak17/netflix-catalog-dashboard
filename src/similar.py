"""Seed-title similarity with a soft profile boost.

Pure ranking logic, no Streamlit widgets: `similar_titles()` takes the catalog
frame plus the sidebar selections and returns a ranked list of
``{"title": ..., "match_score": ...}`` dicts.

The cosine signal is the existing content model in :mod:`src.features` -- the
same TF-IDF + multi-hot matrix the blend uses -- so this module never fits a
second vectorizer. Only the seed's score vector is cached, keyed by show_id:
it is the one thing recomputed per interaction, and it is deterministic, so
``st.cache_data`` is safe (a cache hit is the same numbers as a cold build).

Genre / country / mood are a **boost, not a filter**: they add a small bonus to
the cosine score and re-rank. Nothing is ever excluded, so an over-specified
profile still returns results instead of an empty grid.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_TOP_N = 6
ANY_MOOD = "Any mood"      # sidebar option that applies no mood boost

# Soft re-ranking bonus, in cosine units. For calibration: cosine over this
# catalog runs ~0.01 (median) to ~0.60 (nearest neighbour), so +0.10 lifts a
# genre match by roughly a percentile rank of the distribution while a genuine
# content match (0.4+) still wins outright.
GENRE_BONUS = 0.10
COUNTRY_BONUS = 0.06
MOOD_BONUS = 0.08

# Fallback base score: genre overlap leads, country is the tie-breaker.
FALLBACK_GENRE_W = 0.70
FALLBACK_COUNTRY_W = 0.30

# "Too little text to vectorize meaningfully" -- the seed either has almost no
# description, or almost nothing survived the vectorizer's min_df=2 cutoff.
MIN_TEXT_CHARS = 40
MIN_TEXT_FEATURES = 3

# Mood tag -> (associated genres, words that carry the mood in prose).
# Fixed and hand-written on purpose: the mapping is editorial, not learned.
MOODS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "suspenseful": (
        ("Thrillers", "TV Thrillers", "TV Mysteries", "Crime TV Shows",
         "Horror Movies", "TV Horror"),
        ("suspense", "thriller", "mystery", "killer", "chase", "tension"),
    ),
    "light-hearted": (
        ("Comedies", "TV Comedies", "Stand-Up Comedy", "Children & Family Movies",
         "Kids' TV", "Romantic Movies"),
        ("laugh", "funny", "comic", "romantic", "goofy", "cheerful"),
    ),
    "dark": (
        ("Crime TV Shows", "Thrillers", "TV Thrillers", "Horror Movies",
         "TV Horror", "Cult Movies"),
        ("dark", "murder", "serial killer", "sinister", "gritty", "noir"),
    ),
    "feel-good": (
        ("Comedies", "TV Comedies", "Children & Family Movies", "Kids' TV",
         "Romantic Movies", "Documentaries"),
        ("heartwarming", "feel-good", "uplifting", "friendship", "hope"),
    ),
    "intense": (
        ("Action & Adventure", "TV Action & Adventure", "Thrillers", "Dramas",
         "TV Dramas", "Sports Movies"),
        ("brutal", "relentless", "explosive", "high-stakes", "visceral"),
    ),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_set(value: object) -> set[str]:
    """Normalise a list-valued cell (or a stray scalar / NaN) to a set."""
    if isinstance(value, list):
        return {str(v) for v in value if isinstance(v, str)}
    if isinstance(value, str) and value:
        return {value}
    return set()


def _hits(cell: object, wanted: set[str]) -> bool:
    return bool(_as_set(cell) & wanted)


def mood_profile(mood: str | None) -> tuple[set[str], tuple[str, ...]]:
    """Genres + prose keywords for a mood tag; empty for None / unknown."""
    if not mood:
        return set(), ()
    genres, keywords = MOODS.get(str(mood).strip().lower(), ((), ()))
    return set(genres), tuple(keywords)


def _keyword_mask(df: pd.DataFrame, keywords: Sequence[str]) -> np.ndarray:
    """True where a mood word appears in the title or description."""
    if not keywords:
        return np.zeros(len(df), dtype=bool)
    text = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
    mask = np.zeros(len(df), dtype=bool)
    for word in keywords:
        mask |= text.str.contains(word, regex=False, na=False).to_numpy()
    return mask


# ---------------------------------------------------------------------------
# cosine signal (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cosine_scores(seed_show_id: str) -> np.ndarray:
    """Cosine similarity of every catalog row against one seed, in model order.

    Cached on the seed id: the matrix behind it is already built once per
    process by ``features.get_content_model()``, so a hit skips only the
    dot product. ``st.cache_data`` hands back a fresh array each call, which
    keeps the boosting below free of aliasing bugs.
    """
    from src.features import get_content_model, score_against

    model = get_content_model()
    query = model.vector(seed_show_id)
    if query is None:
        return np.zeros(model.n_titles, dtype=np.float32)
    return score_against(model, query).astype(np.float32)


def _score_vector(df: pd.DataFrame, seed_show_id: str) -> np.ndarray:
    """Model-order cosine vector remapped onto `df`'s row order by show_id.

    Remapping by id rather than position means a subset or re-ordered frame
    still scores correctly -- the model is built from the full catalog.
    """
    from src.features import get_content_model

    scores = _cosine_scores(seed_show_id)
    positions = df["show_id"].map(get_content_model().index_of)
    valid = positions.notna().to_numpy()
    out = np.zeros(len(df), dtype=np.float32)
    if valid.any():
        out[valid] = scores[positions[valid].astype(int).to_numpy()]
    return out


def _thin_text(row: pd.Series) -> bool:
    """True when the seed carries too little prose to vectorize usefully."""
    description = row.get("description")
    if not isinstance(description, str) or len(description.strip()) < MIN_TEXT_CHARS:
        return True
    query = _cosine_for(str(row["show_id"]))
    return query is None or query.nnz <= MIN_TEXT_FEATURES


def _cosine_for(seed_show_id: str):
    from src.features import get_content_model

    return get_content_model().vector(seed_show_id)


# ---------------------------------------------------------------------------
# fallback: genre + country only
# ---------------------------------------------------------------------------
def _fallback_scores(df: pd.DataFrame, seed_genres: set[str],
                     seed_countries: set[str]) -> np.ndarray:
    """Degraded ranking for a seed with no usable text signal.

    Genre overlap carries most of the weight; a shared country breaks ties.
    Scale is comparable to cosine so the match percentage still reads sanely.
    """
    if seed_genres:
        genre_score = df["genres"].map(
            lambda g: len(_as_set(g) & seed_genres) / len(seed_genres)
        ).astype(float).to_numpy()
    else:
        genre_score = np.zeros(len(df), dtype=float)

    if seed_countries:
        country_score = df["countries"].map(
            lambda c: float(bool(_as_set(c) & seed_countries))
        ).astype(float).to_numpy()
    else:
        country_score = np.zeros(len(df), dtype=float)

    return (FALLBACK_GENRE_W * genre_score + FALLBACK_COUNTRY_W * country_score
            ).astype(np.float32)


# ---------------------------------------------------------------------------
# profile boost
# ---------------------------------------------------------------------------
def _boost(df: pd.DataFrame, genres: Sequence[str], country: str | None,
           mood: str | None) -> np.ndarray:
    """Additive bonus per row. Never subtracts, never excludes."""
    bonus = np.zeros(len(df), dtype=np.float32)

    selected = {str(g) for g in genres if g}
    if selected:
        bonus += df["genres"].map(
            lambda g: GENRE_BONUS if _hits(g, selected) else 0.0
        ).astype(float).to_numpy()

    if country:
        wanted = {str(country)}
        bonus += df["countries"].map(
            lambda c: COUNTRY_BONUS if _hits(c, wanted) else 0.0
        ).astype(float).to_numpy()

    mood_genres, mood_words = mood_profile(mood)
    if mood_genres or mood_words:
        hit = np.zeros(len(df), dtype=bool)
        if mood_genres:
            hit |= df["genres"].map(lambda g: _hits(g, mood_genres)).to_numpy()
        if mood_words:
            hit |= _keyword_mask(df, mood_words)
        bonus += hit.astype(np.float32) * MOOD_BONUS

    return bonus


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def similar_titles(
    df: pd.DataFrame,
    seed_title: str,
    *,
    genres: Sequence[str] = (),
    country: str | None = None,
    mood: str | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, object]]:
    """Top-N titles most similar to `seed_title`, boosted by the profile.

    Deterministic: same frame, same seed and same selections always produce
    the same list in the same order. Ties break on title then show_id, so the
    ranking never depends on dict or hash ordering.
    """
    if not seed_title or df.empty:
        return []

    rows = df.index[df["title"] == seed_title]
    if len(rows) == 0:
        return []
    seed = df.loc[rows[0]]

    seed_show_id = str(seed["show_id"])
    candidates = df["show_id"] != seed_show_id
    if not candidates.any():
        return []

    seed_genres = _as_set(seed.get("genres"))
    seed_countries = _as_set(seed.get("countries"))

    base = _score_vector(df, seed_show_id)
    # The seed scores 1.0 against itself, so the "nothing scored" check has to
    # look at the candidates only -- a zero query row is the real signal.
    best = float(base[candidates.to_numpy()].max(initial=0.0))
    if _thin_text(seed) or best <= 0.0:
        base = _fallback_scores(df, seed_genres, seed_countries)

    scores = np.clip(base + _boost(df, genres, country, mood), 0.0, 1.0)

    ranked = df.loc[candidates, ["title", "show_id"]].copy()
    ranked["_score"] = scores[candidates.to_numpy()]
    ranked = ranked.sort_values(
        by=["_score", "title", "show_id"],
        ascending=[False, True, True],
        kind="stable",
    ).head(max(int(top_n), 0))

    return [
        {"title": str(row["title"]), "match_score": int(round(row["_score"] * 100))}
        for _, row in ranked.iterrows()
    ]


def match_caption(results: list[dict[str, object]], seed_title: str) -> str:
    """The one-line summary above the grid."""
    n = len(results)
    return f"{n} title{'s' if n != 1 else ''} similar to {seed_title}"

"""Score blending: collaborative + content + popularity.

Gate outcome from scripts/build_title_index.py (measured, not guessed):
the Netflix <-> MovieLens join covers ~14% of the catalog, so collaborative
filtering is a *secondary* signal and the content model leads. The blend
degrades gracefully:

  * no watchlist yet      -> content (if a seed title is picked) + popularity
  * watchlist, unmatched  -> content + popularity, CF weight silently 0
  * watchlist, >=3 matched rated titles -> full collaborative weight

`alpha` is therefore earned rather than hard-coded: collaborative weight
scales with how many of the user's rated titles actually joined to
MovieLens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import explain
from src.catalog import load_mapping, show_to_movie
from src.cf import get_cf_model, score_items
from src.features import get_content_model, score_against
from src.filters import Filters, _list_overlaps_selection

W_CONTENT = 0.55
W_CF = 0.30
W_POPULAR = 0.15
CF_MIN_ENGAGEMENT = 3     # matched, rated titles before CF counts in full
NEUTRAL_POPULARITY = 0.5  # for titles with no MovieLens rating signal

RESULT_COLUMNS = [
    "show_id", "title", "type", "release_year", "year_added", "rating", "duration",
    "listed_in", "country", "description", "director", "cast",
    "score", "content_score", "cf_score", "popularity_score",
    "model", "reason",
]


def _allowed_mask(df: pd.DataFrame, filters: Filters | None) -> np.ndarray:
    if filters is None:
        return np.ones(len(df), dtype=bool)
    mask = (
        df["year_added"].between(*filters.year_range) &
        df["type"].isin(filters.types)
    )
    if filters.countries:
        mask &= df["countries"].apply(lambda c: _list_overlaps_selection(c, filters.countries))
    if filters.ratings:
        mask &= df["rating"].isin(filters.ratings)
    if filters.genres:
        mask &= df["genres"].apply(lambda g: _list_overlaps_selection(g, filters.genres))
    return mask.to_numpy()


def _unit(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Min-max normalise a score vector over the allowed candidates only."""
    view = values[candidates].astype(np.float64)
    lo, hi = view.min(), view.max()
    if hi - lo < 1e-9:
        return np.zeros_like(view)
    return (view - lo) / (hi - lo)


def _content_query(watchlist: dict[str, dict], content_model, df: pd.DataFrame):
    """Rating-weighted average of the watchlisted titles' vectors."""
    vectors = []
    for show_id, meta in watchlist.items():
        vec = content_model.vector(show_id)
        if vec is None:
            continue
        rating = meta.get("rating")
        weight = (float(rating) - 3.5) if rating else 1.0
        if abs(weight) < 1e-9:
            continue
        vectors.append(vec.multiply(weight))
    if not vectors:
        return None
    query = vectors[0].copy()
    for extra in vectors[1:]:
        query = query + extra
    norm = float(np.sqrt(query.multiply(query).sum()))
    if norm <= 1e-9:
        return None
    return query.multiply(1.0 / norm)


def joined_count(watchlist: dict[str, dict]) -> int:
    """How many saved titles have a MovieLens join (i.e. real CF signal)."""
    if not watchlist:
        return 0
    mapping = show_to_movie(load_mapping())
    return sum(1 for sid in watchlist if sid in mapping)


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def recommend(
    df: pd.DataFrame,
    filters: Filters | None = None,
    *,
    seed_show_id: str | None = None,
    watchlist: dict[str, dict] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top_n recommendations as a DataFrame ready for display."""
    watchlist = watchlist or {}
    content_model = get_content_model()
    cf_model = get_cf_model()
    show_map = show_to_movie(load_mapping())

    allowed = _allowed_mask(df, filters)
    excluded = set(watchlist)
    if seed_show_id:
        excluded.add(seed_show_id)
    allowed &= ~df["show_id"].isin(excluded).to_numpy()

    candidates = np.flatnonzero(allowed)
    if len(candidates) == 0:
        return _empty_result()

    n = len(df)

    # --- content signal ------------------------------------------------
    seed_row: pd.Series | None = None
    content = np.zeros(n, dtype=np.float32)
    profile_genres: set[str] = set()

    if seed_show_id:
        query = content_model.vector(seed_show_id)
        if query is not None:
            content = score_against(content_model, query).astype(np.float32)
            matches = df.index[df["show_id"] == seed_show_id]
            if len(matches):
                seed_row = df.loc[matches[0]]
    elif watchlist:
        query = _content_query(watchlist, content_model, df)
        if query is not None:
            content = score_against(content_model, query).astype(np.float32)
        for show_id in watchlist:
            rows = df.index[df["show_id"] == show_id]
            if len(rows):
                listed = df.loc[rows[0], "listed_in"]
                if isinstance(listed, str):
                    profile_genres.update(p.strip() for p in listed.split(",") if p.strip())

    has_content = bool(np.any(content[candidates]))

    # --- collaborative signal -----------------------------------------
    liked: list[tuple[int, float, str]] = []
    for show_id, meta in watchlist.items():
        rating = meta.get("rating")
        movie_id = show_map.get(show_id)
        if rating is None or movie_id is None:
            continue
        liked.append((int(movie_id), float(rating), show_id))

    cf_result = score_items(cf_model, [(m, r) for m, r, _ in liked],
                            return_attribution=True) if liked else None
    cf_scores = cf_result[0] if cf_result is not None else None
    attribution = cf_result[1] if cf_result is not None else None

    # --- per-row popularity (real signal where joined, neutral elsewhere)
    pop_row = np.full(n, NEUTRAL_POPULARITY, dtype=np.float32)
    cf_row = np.zeros(n, dtype=np.float32)
    cf_source_row = np.zeros(n, dtype=np.int64)   # index into `liked`

    for i, show_id in enumerate(df["show_id"].to_numpy()):
        movie_id = show_map.get(show_id)
        if movie_id is None:
            continue
        col = cf_model.item_index.get(int(movie_id))
        if col is None:
            continue
        pop_row[i] = cf_model.popularity[col]
        if cf_scores is not None and attribution is not None:
            cf_row[i] = cf_scores[col]
            cf_source_row[i] = attribution[col]

    has_cf = cf_scores is not None

    # --- blend ---------------------------------------------------------
    active: dict[str, float] = {}
    if has_content:
        active["content"] = W_CONTENT
    if has_cf:
        active["cf"] = W_CF * min(1.0, len(liked) / CF_MIN_ENGAGEMENT)
    active["popularity"] = W_POPULAR
    total = sum(active.values())

    content_u = _unit(content, candidates) if has_content else np.zeros(len(candidates))
    cf_u = _unit(cf_row, candidates) if has_cf else np.zeros(len(candidates))
    pop_u = pop_row[candidates]

    content_part = content_u * active.get("content", 0.0)
    cf_part = cf_u * active.get("cf", 0.0)
    pop_part = pop_u * active["popularity"]
    final = (content_part + cf_part + pop_part) / total

    # Which model liked this title *most*, independent of blend weight.
    # Weighted contributions would make CF structurally unable to win the
    # label (0.30 can never out-weigh 0.55), so the badge is decided on the
    # normalised signals; inactive signals get -1 so they can never win a tie.
    signals = np.vstack([
        content_u if has_content else np.full(len(candidates), -1.0),
        cf_u if has_cf else np.full(len(candidates), -1.0),
        pop_u,
    ])
    dominant = np.argmax(signals, axis=0)

    order = np.argsort(-final)[:top_n]
    records = []
    for pos in order:
        row_index = int(candidates[pos])
        row = df.iloc[row_index]

        kind = ("content", "cf", "popularity")[int(dominant[pos])]
        if kind == "content":
            reason = explain.content_reason(row, seed_row, profile_genres)
        elif kind == "cf" and liked:
            reason = explain.cf_reason(_title_for(df, liked[int(cf_source_row[row_index])][2]))
        else:
            reason = explain.popularity_reason()

        records.append({
            **{col: row.get(col) for col in (
                "show_id", "title", "type", "release_year", "year_added", "rating",
                "duration", "listed_in", "country", "description", "director", "cast")},
            "score": float(final[pos]),
            "content_score": float(content_part[pos]),
            "cf_score": float(cf_part[pos]),
            "popularity_score": float(pop_part[pos]),
            "model": explain.label(kind),
            "reason": reason,
        })

    return pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)


def _title_for(df: pd.DataFrame, show_id: str) -> str | None:
    """Netflix title for a watchlisted show_id."""
    rows = df.index[df["show_id"] == show_id]
    if len(rows):
        return str(df.loc[rows[0], "title"])
    return None

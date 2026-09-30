"""Seed-and-profile "Find Similar Titles" page: sidebar controls + result grid.

Import-safe: every Streamlit call lives inside render(), so pytest can import
this module without a running app.

Stateless by design. The last requested selection lives in the URL query
string (`?sim_seed=...`) rather than in st.session_state, so results survive
every rerun and are shareable/bookmarkable -- the same "the address bar is the
bookmark" trick the watchlist already uses. Nothing is learned or remembered
beyond that one selection, so the same inputs always draw the same grid.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from src.similar import ANY_MOOD, DEFAULT_TOP_N, MOODS, match_caption, similar_titles

ANY = "Any"
TITLE_LIMIT = 46

# URL state for the last-requested selection.
P_SEED = "sim_seed"
P_GENRES = "sim_genres"
P_COUNTRY = "sim_country"
P_MOOD = "sim_mood"
GENRE_SEP = ";"          # none of the 42 genre labels contain a semicolon


# ---------------------------------------------------------------------------
# option lists (derived from the catalog, like the shared sidebar does)
# ---------------------------------------------------------------------------
def _titles(df: pd.DataFrame) -> list[str]:
    return sorted(df["title"].dropna().astype(str).unique())


def _genres(df: pd.DataFrame) -> list[str]:
    return sorted(df["genres"].dropna().explode().dropna().unique())


def _countries(df: pd.DataFrame) -> list[str]:
    return sorted(df["countries"].dropna().explode().dropna().unique())


def _param(key: str) -> str:
    """Read one query param as a string, tolerating repeated keys."""
    value = st.query_params.get(key)
    if isinstance(value, list):
        value = value[0] if value else ""
    return "" if value is None else str(value)


def _active_selection() -> dict[str, Any] | None:
    """The last requested selection, straight from the URL. None if never run."""
    seed = _param(P_SEED)
    if not seed:
        return None
    return {
        "seed": seed,
        "genres": [g for g in _param(P_GENRES).split(GENRE_SEP) if g],
        "country": _param(P_COUNTRY) or ANY,
        "mood": _param(P_MOOD) or ANY_MOOD,
    }


def _publish(selection: dict[str, Any]) -> None:
    """Write the requested selection into the URL (no rerun triggered)."""
    st.query_params[P_SEED] = selection["seed"]
    st.query_params[P_GENRES] = GENRE_SEP.join(selection["genres"])
    st.query_params[P_COUNTRY] = selection["country"]
    st.query_params[P_MOOD] = selection["mood"]


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def _sidebar(df: pd.DataFrame, defaults: dict[str, Any] | None) -> dict[str, Any]:
    """Page controls, appended under the shared filter sidebar."""
    defaults = defaults or {}
    titles, all_genres, all_countries = _titles(df), _genres(df), _countries(df)

    st.sidebar.divider()
    st.sidebar.header("Find similar titles")

    seed_index = titles.index(defaults["seed"]) if defaults.get("seed") in titles else 0
    seed = st.sidebar.selectbox("Seed title", titles, index=seed_index)

    selected_genres = [g for g in defaults.get("genres", []) if g in all_genres]
    genres = st.sidebar.multiselect("Genre", all_genres, default=selected_genres)

    wanted = defaults.get("country") or ANY
    country_index = all_countries.index(wanted) + 1 if wanted in all_countries else 0
    country = st.sidebar.selectbox("Country", [ANY, *all_countries], index=country_index)

    mood_list = [ANY_MOOD, *MOODS]
    wanted_mood = defaults.get("mood") or ANY_MOOD
    mood_index = mood_list.index(wanted_mood) if wanted_mood in mood_list else 0
    mood = st.sidebar.selectbox("Mood", mood_list, index=mood_index)

    clicked = st.sidebar.button(
        "Get recommendations", type="primary", use_container_width=True,
    )

    return {"seed": seed, "genres": genres, "country": country,
            "mood": mood, "clicked": clicked}


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Selections match, order-insensitively for the genre list."""
    return (
        a["seed"] == b["seed"]
        and a["country"] == b["country"]
        and a["mood"] == b["mood"]
        and sorted(a["genres"]) == sorted(b["genres"])
    )


# ---------------------------------------------------------------------------
# result grid
# ---------------------------------------------------------------------------
def _initials(title: str) -> str:
    words = [w for w in title.split() if w]
    if len(words) > 1 and words[0].lower() in {"the", "a", "an"}:
        words = words[1:]
    return "".join(w[0] for w in words[:2]).upper() or "?"


def _truncate(title: str, limit: int = TITLE_LIMIT) -> str:
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def _poster(title: str) -> str:
    """Thumbnail stand-in. The catalog carries no artwork, so the tile is a
    neutral block keyed off the title -- the only place here that needs HTML,
    since Streamlit has no native element for a sized placeholder."""
    return (
        '<div style="background:#f0f2f6;border-radius:8px;height:118px;'
        "display:flex;align-items:center;justify-content:center;"
        'color:#9aa1ab;font-size:2rem;font-weight:700;letter-spacing:.06em;">'
        f"{html.escape(_initials(title))}</div>"
    )


def _title_block(title: str) -> str:
    """Bold title, clipped to one line; the full string stays on hover."""
    return (
        '<div title="' + html.escape(title, quote=True) + '"'
        ' style="font-weight:600;margin:.5rem 0 .1rem;overflow:hidden;'
        'text-overflow:ellipsis;white-space:nowrap;">'
        + html.escape(_truncate(title))
        + "</div>"
    )


def _card(column, item: dict[str, Any]) -> None:
    title = str(item["title"])
    with column:
        st.markdown(_poster(title), unsafe_allow_html=True)
        st.markdown(_title_block(title), unsafe_allow_html=True)
        st.caption(f"{int(item['match_score'])}% match")


def _grid(results: list[dict[str, Any]]) -> None:
    for start in range(0, len(results), 3):
        columns = st.columns(3)
        for column, item in zip(columns, results[start:start + 3]):
            _card(column, item)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame, filtered_df: pd.DataFrame, filters) -> None:
    st.title("🔍 Find Similar Titles")

    active = _active_selection()
    selection = _sidebar(df, active)

    if selection["clicked"]:
        _publish(selection)
        active = {k: selection[k] for k in ("seed", "genres", "country", "mood")}

    if active is None:
        st.info("Pick a **seed title** and, if you like, a genre, country or mood "
                "in the sidebar — then press **Get recommendations**.")
        return

    with st.spinner("Ranking the catalog…"):
        results = similar_titles(
            df,
            active["seed"],
            genres=active["genres"],
            country=None if active["country"] == ANY else active["country"],
            mood=None if active["mood"] == ANY_MOOD else active["mood"],
            top_n=DEFAULT_TOP_N,
        )

    st.caption(match_caption(results, active["seed"]))

    if not results:
        st.warning(f"Nothing similar to **{active['seed']}** was found in the catalog.")
        return

    _grid(results)

    if not _same(selection, active):
        st.caption("Sidebar changed since this ran — press **Get recommendations** "
                   "to refresh.")

"""Recommendation UI: For You / Because You Watched / Watchlist / How It Works.

Import-safe: all Streamlit calls happen inside render(), so pytest can
import this module without a running app.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src import blend
from src.catalog import join_coverage
from src.storage import get_storage, new_key

TOP_N = 12
RATING_OPTIONS = ["Not rated", 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# per-user key + storage plumbing
# ---------------------------------------------------------------------------
def _ensure_key() -> str:
    """Resolve this visitor's watchlist key.

    Precedence: ?w= in the URL (a link they saved or were handed) > a key
    already in session > a freshly generated one, which is written back into
    the URL so the address bar itself is the bookmark.
    """
    incoming = st.query_params.get("w")
    if incoming:
        st.session_state["watch_key"] = str(incoming)
    elif "watch_key" not in st.session_state:
        key = new_key()
        st.session_state["watch_key"] = key
        st.query_params["w"] = key
    return str(st.session_state["watch_key"])


def _load_watchlist(storage, key: str) -> dict[str, dict]:
    data = storage.load(key)
    if not isinstance(data, dict):
        return {}
    # tolerate hand-edited / imported files rather than crashing the page
    return {sid: meta for sid, meta in data.items() if isinstance(meta, dict)}


def _persist(storage, key: str, watchlist: dict) -> None:
    """Write-on-change only: never saves on an ordinary rerun."""
    storage.save(key, watchlist)


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def _add_key(scope: str, show_id: object) -> str:
    """Button key for a card.

    Every tab renders on every rerun, and For You / Because You Watched can
    both recommend the same title -- a bare ``add_<show_id>`` then raises
    StreamlitDuplicateElementKey on the second registration.
    """
    return f"{scope}_add_{show_id}"


def _card(row: pd.Series, watchlist: dict, key: str, storage, *,
          scope: str, addable: bool) -> None:
    year = row.get("release_year")
    left, body, right = st.columns([1, 6, 2])

    with left:
        st.metric("Match", f"{float(row['score']) * 100:.0f}%")

    with body:
        st.markdown(f"**{row['title']}**" + (f"  ({int(year)})" if pd.notna(year) else ""))
        st.caption(f"{row['model']} — {row['reason']}")
        st.caption(
            f"{row['type']} · {row['rating']} · {row['duration']} · {row['listed_in']}"
        )
        if isinstance(row.get("description"), str):
            st.write(row["description"])

    with right:
        in_list = row["show_id"] in watchlist
        if in_list:
            st.success("In watchlist")
        elif addable and st.button("➕ Add", key=_add_key(scope, row["show_id"])):
            watchlist[row["show_id"]] = {
                "rating": None,
                "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _persist(storage, key, watchlist)
            st.rerun()

    st.divider()


def _render_results(frame: pd.DataFrame, watchlist, key, storage, *,
                    scope: str, addable: bool = True) -> None:
    if frame.empty:
        st.info("No titles match the current sidebar filters. Widen them to see more.")
        return
    for _, row in frame.iterrows():
        _card(row, watchlist, key, storage, scope=scope, addable=addable)


def _coverage_caption() -> None:
    matched, total = join_coverage()
    st.caption(
        f"Collaborative filtering is joined to {matched:,} of {total:,} catalog titles "
        f"({matched / max(total, 1):.0%}). Everything else is scored on content alone — "
        f"including all {total - matched:,} TV shows, which MovieLens does not cover."
    )


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------
def _tab_for_you(df, filters, watchlist, key, storage) -> None:
    if not watchlist:
        st.info(
            "Your watchlist is empty. Add a few titles (➕) and rate them, and the "
            "collaborative model starts contributing. Until then, here is what is "
            "popular with MovieLens raters."
        )
        with st.spinner("Ranking the catalog…"):
            frame = blend.recommend(df, filters, top_n=8)
        _render_results(frame, watchlist, key, storage, scope="foryou")
        return

    rated = sum(1 for e in watchlist.values() if e.get("rating"))
    joined = blend.joined_count(watchlist)
    st.caption(f"{len(watchlist)} titles saved · {rated} rated · {joined} of them joined to MovieLens")

    with st.spinner("Ranking the catalog…"):
        frame = blend.recommend(df, filters, watchlist=watchlist, top_n=TOP_N)
    _render_results(frame, watchlist, key, storage, scope="foryou")


def _tab_because_you_watched(df, filters, watchlist, key, storage) -> None:
    options = filters_df_titles(df, filters)
    if not options:
        st.info("No titles match the current sidebar filters.")
        return

    choice = st.selectbox("Pick a title you enjoyed", options=options, index=0)
    seed_row = df[df["title"] == choice]
    if seed_row.empty:
        return
    seed_show_id = str(seed_row.iloc[0]["show_id"])

    with st.spinner("Finding neighbours…"):
        frame = blend.recommend(df, filters, seed_show_id=seed_show_id, top_n=TOP_N)
    st.caption(f"Because you watched **{choice}**")
    _render_results(frame, watchlist, key, storage, scope="because")


def _tab_watchlist(df, filters, watchlist, key, storage) -> None:
    st.subheader("Your list follows you")
    url = _current_url(key)
    st.code(url, language=None)
    st.caption(
        "Bookmark or share that link — it is how you get your watchlist back. "
        "**Anyone with the link can read and edit it**, so treat it like a password."
    )
    if storage.last_error:
        st.warning(f"Storage fallback active ({storage.last_error}). "
                   f"Data is only in this browser for now.")

    if not watchlist:
        st.info("Nothing saved yet. Use ➕ on any recommendation, or add one below.")
        _add_form(df, watchlist, key, storage)
        return

    rows = []
    for show_id, entry in watchlist.items():
        match = df[df["show_id"] == show_id]
        if match.empty:
            continue
        rows.append((show_id, match.iloc[0], entry))
    rows.sort(key=lambda r: r[1]["title"])

    for show_id, title_row, entry in rows:
        c1, c2, c3, c4 = st.columns([5, 2, 2, 1])
        with c1:
            year = title_row["release_year"]
            st.markdown(f"**{title_row['title']}**" + (f" ({int(year)})" if pd.notna(year) else ""))
            st.caption(f"{title_row['type']} · {title_row['rating']} · {title_row['listed_in']}")
        with c2:
            current = entry.get("rating")
            try:
                index = 0 if current in (None, "") else min(5, max(1, int(round(float(current)))))
            except (TypeError, ValueError):
                index = 0
            choice = st.selectbox("Rating", RATING_OPTIONS, key=f"rate_{show_id}",
                                  index=index)
            if choice != "Not rated" and entry.get("rating") != float(choice):
                entry["rating"] = float(choice)
                _persist(storage, key, watchlist)
                st.rerun()
        with c3:
            st.caption(f"Saved {str(entry.get('added_at', ''))[:10]}")
        with c4:
            if st.button("🗑️", key=f"rm_{show_id}", help="Remove"):
                del watchlist[show_id]
                _persist(storage, key, watchlist)
                st.rerun()
        st.divider()

    _add_form(df, watchlist, key, storage)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.download_button(
            "⬇️ Export JSON",
            data=json.dumps(watchlist, indent=2, sort_keys=True),
            file_name="netflix-watchlist.json",
            mime="application/json",
        )
    with right:
        uploaded = st.file_uploader("⬆️ Import JSON", type=["json"], key="importer")
        if uploaded is not None:
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))
                if isinstance(payload, dict):
                    merged = {**watchlist, **payload}
                    _persist(storage, key, merged)
                    st.rerun()
            except (json.JSONDecodeError, UnicodeDecodeError):
                st.error("That file is not valid JSON.")


def _add_form(df, watchlist, key, storage) -> None:
    st.markdown("**Add a title**")
    titles = sorted(df["title"].dropna().unique())
    choice = st.selectbox("Search the catalog", options=titles, key="add_search")
    if st.button("➕ Add selected", key="add_selected") and choice:
        rows = df[df["title"] == choice]
        if not rows.empty:
            show_id = str(rows.iloc[0]["show_id"])
            if show_id not in watchlist:
                watchlist[show_id] = {
                    "rating": None,
                    "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                _persist(storage, key, watchlist)
                st.rerun()


def _tab_how_it_works() -> None:
    st.subheader("How the ranking is built")
    st.write(
        "Three signals are normalised to 0-1 over your filtered candidate set, "
        "then combined:"
    )
    st.dataframe(
        pd.DataFrame([
            ("Content", blend.W_CONTENT,
             "TF-IDF over description, genres, director, cast, country, "
             "plus genre/country/rating multi-hot blocks"),
            ("Collaborative", blend.W_CF,
             "Item-item cosine over 100,836 MovieLens ratings; weight scales up "
             "once 3+ of your rated titles join to MovieLens"),
            ("Popularity", blend.W_POPULAR,
             "log rating count per MovieLens item; neutral 0.5 where no join exists"),
        ], columns=["Signal", "Weight", "What it measures"]),
        hide_index=True, width="stretch",
    )
    st.caption(
        "The badge on each card shows which signal scored that title highest, "
        "not which one carried the most weight."
    )

    st.subheader("Catalog join")
    matched, total = join_coverage()
    st.progress(matched / max(total, 1),
                text=f"{matched:,} / {total:,} titles joined ({matched / max(total, 1):.1%})")
    st.caption(
        "The join is a title-level fuzzy match (see scripts/build_title_index.py). "
        "MovieLens holds 9,742 movies through 2019 and no TV shows, so the overlap "
        "is genuinely small — that is a property of the datasets, not a missed fetch."
    )

    st.subheader("Offline evaluation")
    try:
        with st.spinner("Running leave-one-out over 609 MovieLens users (~20s first run)…"):
            from src.evaluate import summary_table
            st.dataframe(summary_table(), hide_index=True, width="stretch")
        st.caption(
            "Leave-one-out: hide one item a user rated ≥ 3.5, rank everything they "
            "have not rated, check whether it lands in the top 10. The baseline "
            "column is the same protocol using a popularity-only ranker."
        )
    except Exception as exc:                                    # noqa: BLE001
        st.warning(f"Evaluation unavailable: {type(exc).__name__}: {exc}")

    st.subheader("Limitations")
    st.markdown(
        "- Ratings come from 610 anonymized MovieLens users (data through 2018) — "
        "not Netflix viewing data, which is not public.\n"
        "- ~86% of the catalog has no collaborative signal and is scored on content "
        "alone.\n"
        "- The title join is deliberately precision-first; a few hundred borderline "
        "titles were rejected rather than guessed (see data/movielens/title_review.csv).\n"
        "- Not affiliated with Netflix or GroupLens."
    )
    st.caption(
        "MovieLens data © GroupLens. Please cite: Harper & Konstan (2015), "
        "The MovieLens Datasets: History and Context, ACM TiiS 5(4), 19:1-19:19."
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def filters_df_titles(df: pd.DataFrame, filters) -> list[str]:
    """Titles currently allowed by the sidebar filters, for the seed picker."""
    from src.blend import _allowed_mask

    mask = _allowed_mask(df, filters)
    return sorted(df.loc[mask, "title"].dropna().unique())


def _current_url(key: str) -> str:
    params = dict(st.query_params)
    params["w"] = key
    query = "&".join(f"{k}={v}" for k, v in params.items())
    origin = None
    try:
        origin = st.context.headers.get("origin")
    except Exception:                                          # noqa: BLE001
        origin = None
    # Origin is absent on a plain document GET, so fall back to the query
    # string alone rather than inventing a host.
    return f"{origin}/?{query}" if origin else f"?{query}"


def render(df: pd.DataFrame, filtered_df: pd.DataFrame, filters) -> None:
    st.title("🎬 Netflix Recommendations")
    _coverage_caption()

    storage = get_storage()
    key = _ensure_key()
    watchlist = _load_watchlist(storage, key)

    tabs = st.tabs(["✨ For You", "🎯 Because You Watched",
                    "🍿 My Watchlist", "🔬 How It Works"])

    with tabs[0]:
        _tab_for_you(df, filters, watchlist, key, storage)
    with tabs[1]:
        _tab_because_you_watched(df, filters, watchlist, key, storage)
    with tabs[2]:
        _tab_watchlist(df, filters, watchlist, key, storage)
    with tabs[3]:
        _tab_how_it_works()

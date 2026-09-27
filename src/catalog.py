"""Netflix <-> MovieLens bridge.

The mapping is produced by scripts/build_title_index.py and committed as
data/movielens/title_mapping.csv. Only ~14% of the Netflix catalog joins
(anything more would be dishonest: MovieLens holds 9,742 movies and no TV),
so every caller must be able to degrade to content-only scoring.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

MAPPING_CSV = Path("data/movielens/title_mapping.csv")


@st.cache_data
def load_mapping(path: str = str(MAPPING_CSV)) -> pd.DataFrame:
    """Columns: show_id, movieId, source, netflix_title, movielens_title."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["show_id", "movieId", "source"])
    df = pd.read_csv(path)
    df = df.rename(columns={"best_movieId": "movieId", "best_title": "movielens_title"})
    df = df[["show_id", "movieId"] + [c for c in ("source", "netflix_title",
                                                  "movielens_title") if c in df.columns]]
    df["show_id"] = df["show_id"].astype(str)
    df["movieId"] = df["movieId"].astype(int)
    return df.drop_duplicates(subset=["show_id"], keep="first")


def show_to_movie(mapping: pd.DataFrame | None = None) -> dict[str, int]:
    m = mapping if mapping is not None else load_mapping()
    if m.empty:
        return {}
    return dict(zip(m["show_id"], m["movieId"]))


def movie_to_show(mapping: pd.DataFrame | None = None) -> dict[int, str]:
    """Reverse lookup. Where the catalog contains duplicates, the first
    show_id wins -- both resolve to the same MovieLens item anyway."""
    m = mapping if mapping is not None else load_mapping()
    if m.empty:
        return {}
    return dict(zip(m["movieId"].astype(int), m["show_id"]))


@st.cache_data
def join_coverage() -> tuple[int, int]:
    """(matched titles, total titles) for the coverage meter."""
    from src.data import load_data
    mapping = load_mapping()
    df = load_data()
    return int(len(mapping)), int(len(df))

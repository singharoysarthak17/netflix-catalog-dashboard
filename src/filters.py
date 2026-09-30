"""Sidebar filter widgets and the filtering logic they drive.

Both the dashboard and the recommendation pages share these, so the filters
stay live no matter which page you are on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st


@dataclass
class Filters:
    year_range: tuple[int, int]
    types: list[str]
    countries: list[str]
    ratings: list[str]
    genres: list[str]
    # option lists are built from the *unfiltered* frame so widgets always
    # offer the full set of choices, never a shrinking subset
    all_countries: list[str] = field(default_factory=list)
    all_genres: list[str] = field(default_factory=list)
    all_ratings: list[str] = field(default_factory=list)


def build_sidebar(df: pd.DataFrame) -> Filters:
    st.sidebar.header("Filters")

    year_range = st.sidebar.slider(
        "Year Added",
        int(df["year_added"].min()), int(df["year_added"].max()),
        (2015, int(df["year_added"].max())),
    )

    type_options = sorted(df["type"].unique())
    type_filter = st.sidebar.multiselect(
        "Content Type",
        options=type_options,
        default=type_options,
    )

    all_countries = sorted(df["countries"].dropna().explode().dropna().unique())
    all_genres = sorted(df["genres"].dropna().explode().dropna().unique())
    all_ratings = sorted(df["rating"].dropna().unique())

    # Default to empty selection = "no filter" (show all). Avoids pre-selecting
    # 100+ countries / 42 genres in widget state, which bloats session on Cloud.
    country_filter = st.sidebar.multiselect(
        "Country", options=all_countries, default=[])
    rating_filter = st.sidebar.multiselect(
        "Rating", options=all_ratings, default=[])
    genre_filter = st.sidebar.multiselect(
        "Genre", options=all_genres, default=[])

    return Filters(
        year_range=year_range,
        types=type_filter,
        countries=country_filter,
        ratings=rating_filter,
        genres=genre_filter,
        all_countries=all_countries,
        all_genres=all_genres,
        all_ratings=all_ratings,
    )


def _list_overlaps_selection(cell: object, selected: list[str]) -> bool:
    """True if a list-valued cell (genres/countries) shares at least one
    item with the selected filter options. Handles NaN / non-list cells."""
    if not isinstance(cell, list):
        return False
    return any(item in selected for item in cell)


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    mask = (df["year_added"].between(*f.year_range)) & (df["type"].isin(f.types))
    if f.countries:
        mask &= df["countries"].apply(lambda c: _list_overlaps_selection(c, f.countries))
    if f.ratings:
        mask &= df["rating"].isin(f.ratings)
    if f.genres:
        mask &= df["genres"].apply(lambda g: _list_overlaps_selection(g, f.genres))
    return df[mask]

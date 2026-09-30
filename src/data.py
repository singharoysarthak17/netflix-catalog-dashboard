"""Netflix catalog data loading and cleaning."""

from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(path: str = "netflix_titles.csv") -> pd.DataFrame:
    """Load and clean the Netflix catalog.

    Cached so it does not reload on every widget interaction. 8,807 raw rows,
    8,709 after dropping the 98 rows with no usable date_added.
    """
    # Only load columns the app actually uses. Drop heavy ones we don't need.
    usecols = [
        "show_id", "type", "title", "director", "cast", "country",
        "date_added", "release_year", "rating", "duration",
        "listed_in", "description",
    ]
    df = pd.read_csv(path, usecols=usecols)

    # Truncate text fields to cap TF-IDF memory (all feed the vectorizer)
    for col in ("description", "director", "cast", "listed_in", "country", "rating"):
        if col in df.columns:
            df[col] = df[col].astype(str).str[:500]

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df = df.dropna(subset=["year_added"])
    df["year_added"] = df["year_added"].astype(int)
    df["genres"] = df["listed_in"].str.split(", ")
    df["countries"] = df["country"].str.split(", ")
    return df

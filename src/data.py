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
    df = pd.read_csv(path)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df = df.dropna(subset=["year_added"])
    df["year_added"] = df["year_added"].astype(int)
    df["genres"] = df["listed_in"].str.split(", ")
    df["countries"] = df["country"].str.split(", ")
    df["cast_list"] = df["cast"].str.split(", ")
    df["director_list"] = df["director"].str.split(", ")
    return df

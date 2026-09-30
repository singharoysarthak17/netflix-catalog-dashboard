"""Page entry point for the analytics dashboard.

Thin wrapper: Streamlit executes this file directly, so it only pulls state
published by app.py and calls the import-safe renderer in ui.dashboard.
"""

from __future__ import annotations

import streamlit as st
from src.filters import apply_filters, build_sidebar

from ui.dashboard import render

df = st.session_state["netflix_df_provider"]()
filters = build_sidebar(df)
filtered_df = apply_filters(df, filters)

render(df, filtered_df)
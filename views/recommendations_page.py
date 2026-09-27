"""Page entry point for recommendations.

Thin wrapper: Streamlit executes this file directly, so it only pulls state
published by app.py and calls the import-safe renderer in ui.recommend.
"""

from __future__ import annotations

import streamlit as st

from ui.recommend import render

df = st.session_state["netflix_df"]
filtered_df = st.session_state["filtered_df"]
filters = st.session_state["filters"]

render(df, filtered_df, filters)

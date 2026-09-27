"""Page entry point for the analytics dashboard.

Thin wrapper: Streamlit executes this file directly, so it only pulls state
published by app.py and calls the import-safe renderer in ui.dashboard.
"""

from __future__ import annotations

import streamlit as st

from ui.dashboard import render

df = st.session_state["netflix_df"]
filtered_df = st.session_state["filtered_df"]

render(df, filtered_df)

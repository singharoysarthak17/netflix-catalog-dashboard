"""Page entry point for "Find Similar Titles".

Thin wrapper: Streamlit executes this file directly, so it only pulls state
published by app.py and calls the import-safe renderer in ui.similar.
"""

from __future__ import annotations

import streamlit as st

from ui.similar import render

df = st.session_state["netflix_df"]
filtered_df = st.session_state["filtered_df"]
filters = st.session_state["filters"]

render(df, filtered_df, filters)

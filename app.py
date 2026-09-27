"""Netflix Catalog & Recommendations - application entry point.

Runs the shared sidebar filters, then hands off to the selected page:

    python -m streamlit run app.py

Pages live in views/ as thin scripts (Streamlit executes them directly);
the import-safe rendering logic lives in ui/.
"""

from __future__ import annotations

import streamlit as st

# Page config must be the very first Streamlit command.
st.set_page_config(
    page_title="Netflix Catalog & Recommendations",
    page_icon="🎬",
    layout="wide",
)

from src.data import load_data            # noqa: E402  (after set_page_config)
from src.filters import apply_filters, build_sidebar   # noqa: E402

df = load_data()
filters = build_sidebar(df)
filtered_df = apply_filters(df, filters)

# Published once per rerun; views/ scripts read these back.
st.session_state["netflix_df"] = df
st.session_state["filters"] = filters
st.session_state["filtered_df"] = filtered_df

navigation = st.navigation(
    {
        "Discover": [
            st.Page("views/recommendations_page.py",
                    title="Recommendations", icon="🎬", default=True),
            st.Page("views/dashboard_page.py",
                    title="Dashboard", icon="📊"),
        ]
    },
    position="sidebar",
)
navigation.run()

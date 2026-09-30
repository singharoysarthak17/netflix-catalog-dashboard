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

# Import-safe: defer heavy work until a page actually renders.
# The views/ scripts call their render() with state we publish here.

navigation = st.navigation(
    {
        "Discover": [
            st.Page("views/recommendations_page.py",
                    title="Recommendations", icon="🎬", default=True),
            st.Page("views/similar_page.py",
                    title="Find Similar Titles", icon="🔍"),
            st.Page("views/dashboard_page.py",
                    title="Dashboard", icon="📊"),
        ]
    },
    position="sidebar",
)

# Publish a lazy data provider - views call render(), which imports and calls
# src.data.load_data() on first use (cached via @st.cache_data).
def _get_df() -> "pd.DataFrame":
    from src.data import load_data
    return load_data()

st.session_state["netflix_df_provider"] = _get_df

navigation.run()
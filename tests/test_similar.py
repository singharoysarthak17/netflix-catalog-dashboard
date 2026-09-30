"""Tests for the seed-title similarity engine (src/similar.py).

The content model builds once for the whole suite; every test after the
first reuses it through Streamlit's resource cache.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import load_data
from src.similar import (
    ANY_MOOD,
    DEFAULT_TOP_N,
    MOODS,
    match_caption,
    mood_profile,
    similar_titles,
)

SEED = "The Matrix"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


def _all_scores(df: pd.DataFrame, **kwargs) -> dict[str, int]:
    """Every candidate title with its match score, for precise comparisons."""
    return {item["title"]: item["match_score"]
            for item in similar_titles(df, SEED, top_n=len(df), **kwargs)}


def test_returns_the_requested_number_of_results(df):
    assert len(similar_titles(df, SEED, top_n=6)) == 6
    assert len(similar_titles(df, SEED, top_n=3)) == 3
    assert len(similar_titles(df, SEED)) == DEFAULT_TOP_N


def test_result_shape_is_exactly_title_and_match_score(df):
    out = similar_titles(df, SEED)
    assert len(out) == DEFAULT_TOP_N
    for item in out:
        assert set(item) == {"title", "match_score"}
        assert isinstance(item["title"], str)
        assert isinstance(item["match_score"], int)
        assert 0 <= item["match_score"] <= 100


def test_seed_is_never_recommended(df):
    assert SEED not in {item["title"] for item in similar_titles(df, SEED)}


def test_scores_are_ranked_highest_first(df):
    scores = [item["match_score"] for item in similar_titles(df, SEED)]
    assert scores == sorted(scores, reverse=True)


def test_is_deterministic(df):
    kwargs = dict(genres=["Dramas"], country="India", mood="dark")
    assert similar_titles(df, SEED, **kwargs) == similar_titles(df, SEED, **kwargs)
    assert similar_titles(df, SEED) == similar_titles(df, SEED)


def test_sequel_of_the_same_franchise_ranks_first(df):
    """Sanity: the content model must actually know what a neighbour is."""
    assert "Matrix" in similar_titles(df, SEED, top_n=1)[0]["title"]


def test_unknown_seed_returns_empty_list(df):
    assert similar_titles(df, "Not a real title") == []
    assert similar_titles(df, "") == []


def test_genre_boost_raises_matching_titles(df):
    plain = _all_scores(df)
    boosted = _all_scores(df, genres=["Dramas"])

    genres = df.set_index("title")["genres"]
    drama_titles = [t for t, g in genres.items()
                    if t != SEED and isinstance(g, list) and "Dramas" in g]
    assert drama_titles, "catalog has no Dramas"

    for title in drama_titles:
        assert boosted[title] > plain[title], f"{title} was not boosted"


def test_boost_is_soft_not_a_filter(df):
    """A profile nobody matches still returns a full grid of results."""
    out = similar_titles(df, SEED, genres=["Documentaries"], country="India",
                         mood="feel-good")
    assert len(out) == DEFAULT_TOP_N
    assert all(0 <= item["match_score"] <= 100 for item in out)


def test_boosts_are_additive_and_independent(df):
    plain = _all_scores(df)
    mood = _all_scores(df, mood="intense")
    country = _all_scores(df, country="United States")

    assert any(mood[t] > plain[t] for t in plain), "mood boost never applied"
    assert any(country[t] > plain[t] for t in plain), "country boost never applied"
    assert all(mood[t] >= plain[t] for t in plain), "boost must never subtract"
    assert all(country[t] >= plain[t] for t in plain), "boost must never subtract"

    untouched = [t for t in plain if mood[t] == plain[t] and country[t] == plain[t]]
    assert untouched, "expected titles unaffected by both boosts"


def test_every_mood_tag_has_a_profile():
    for tag in ("suspenseful", "light-hearted", "dark", "feel-good", "intense"):
        assert tag in MOODS
        genres, keywords = mood_profile(tag)
        assert genres, f"{tag} maps to no genres"
        assert keywords, f"{tag} maps to no keywords"
    assert mood_profile(ANY_MOOD) == (set(), ())
    assert mood_profile(None) == (set(), ())
    assert mood_profile("not a mood") == (set(), ())


# --- thin-text fallback ----------------------------------------------------
@pytest.fixture(scope="module")
def thin_catalog() -> pd.DataFrame:
    """Metadata-only catalog: descriptions are too short to vectorize and the
    ids are unknown to the content model, so scoring must fall back to
    genre + country."""
    return pd.DataFrame({
        "show_id": ["x1", "x2", "x3", "x4", "x5"],
        "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "description": ["a", "", "b", "c", "d"],
        "genres": [["Dramas"], ["Dramas", "Comedies"], ["Action & Adventure"],
                   ["Kids' TV"], ["Music & Musicals"]],
        "countries": [["United States"], ["United Kingdom"], ["United States"],
                      ["India"], ["Canada"]],
    })


def test_thin_text_falls_back_to_genre_and_country(thin_catalog):
    out = similar_titles(thin_catalog, "Alpha")
    # Beta shares the only genre, Gamma the country, the rest share neither
    # (ties break on title, so Delta precedes Epsilon).
    assert [i["title"] for i in out] == ["Beta", "Gamma", "Delta", "Epsilon"]
    assert out[0]["match_score"] > out[-1]["match_score"]


def test_fallback_still_applies_the_profile_boost(thin_catalog):
    plain = {i["title"]: i["match_score"] for i in similar_titles(thin_catalog, "Alpha")}
    canada = {i["title"]: i["match_score"]
              for i in similar_titles(thin_catalog, "Alpha", country="Canada")}
    assert canada["Epsilon"] > plain["Epsilon"]


def test_fallback_is_deterministic(thin_catalog):
    assert (similar_titles(thin_catalog, "Alpha")
            == similar_titles(thin_catalog, "Alpha"))


def test_frame_with_no_candidates_returns_empty(df):
    assert similar_titles(df.head(0), SEED) == []


def test_single_row_frame_returns_empty():
    one = pd.DataFrame({
        "show_id": ["only"], "title": ["Solo"], "description": ["x"],
        "genres": [[]], "countries": [[]],
    })
    assert similar_titles(one, "Solo") == []


def test_match_caption_reads_naturally():
    assert match_caption([], "The Matrix") == "0 titles similar to The Matrix"
    one = [{"title": "x", "match_score": 1}]
    assert match_caption(one, "The Matrix") == "1 title similar to The Matrix"
    assert match_caption(one * 6, "The Matrix") == "6 titles similar to The Matrix"

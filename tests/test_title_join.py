"""Title-join tests: normalisation, genre mapping, and the shipped mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_title_index import (
    genre_jaccard,
    load_movielens,
    movielens_genres,
    netflix_genres,
    normalize_title,
)

MAPPING = Path("data/movielens/title_mapping.csv")


def test_normalize_strips_year_punctuation_and_article():
    assert normalize_title("The Departed (2006)") == "departed"
    assert normalize_title("Departed, The (2006)") == "departed"   # MovieLens form
    assert normalize_title("An Education (2009)") == "education"
    assert normalize_title("Education, An (2009)") == "education"
    assert normalize_title("Verónica") == "veronica"          # accent folded
    assert normalize_title("Mac & Devin Go to High School") == "mac and devin go to high school"
    assert normalize_title(None) == ""


def test_netflix_plural_genres_map_to_movielens_vocab():
    assert "Drama" in netflix_genres("Dramas, International Movies")
    assert "Comedy" in netflix_genres("Stand-Up Comedy")
    assert "Children" in netflix_genres("Children & Family Movies, Comedies")
    assert netflix_genres("") == set()
    # "International Movies" has no MovieLens equivalent and must contribute
    # nothing rather than a bogus label
    assert netflix_genres("International Movies") == set()


def test_movielens_uses_children_not_children_s():
    assert "Children" in movielens_genres("Adventure|Children|Comedy")
    assert movielens_genres("(no genres listed)") == set()


def test_genre_jaccard_none_means_no_evidence():
    assert genre_jaccard(set(), {"Drama"}) is None            # must not penalise
    assert genre_jaccard({"Drama"}, {"Comedy"}) == 0.0
    assert genre_jaccard({"Drama"}, {"Drama", "Comedy"}) == 0.5


def test_shipped_mapping_is_sane():
    assert MAPPING.exists(), "run: python scripts/build_title_index.py"
    from src.catalog import load_mapping
    mapping = load_mapping()                      # canonical consumer, renames columns
    assert {"show_id", "movieId"}.issubset(mapping.columns)
    assert mapping["show_id"].is_unique, "a show_id must join to at most one movie"
    assert mapping["movieId"].nunique() > 500

    ml_ids = set(load_movielens()["movieId"])
    assert set(mapping["movieId"]).issubset(ml_ids), "mapping points at unknown movieId"


def test_join_covers_a_meaningful_share_of_movies():
    mapping = pd.read_csv(MAPPING)
    netflix = pd.read_csv("netflix_titles.csv")
    movies = netflix[netflix["type"] == "Movie"]
    rate = len(set(mapping["show_id"]) & set(movies["show_id"])) / len(movies)
    # Phase 0 gate: measured at 14% with ml-latest-small. Guard against a
    # silent regression (e.g. a normalisation change that stops matching).
    assert 0.10 <= rate <= 0.40, f"unexpected movie join rate {rate:.1%}"


def test_tv_shows_are_never_joined():
    """MovieLens has no TV; any TV row in the mapping is a matcher bug."""
    mapping = pd.read_csv(MAPPING)
    netflix = pd.read_csv("netflix_titles.csv")
    tv_ids = set(netflix.loc[netflix["type"] == "TV Show", "show_id"])
    assert set(mapping["show_id"]).isdisjoint(tv_ids)

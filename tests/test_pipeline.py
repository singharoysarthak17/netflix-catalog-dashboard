"""End-to-end tests over the real catalog and ratings data.

First collection builds the content matrix (~2s) and ratings model; every
test after that reuses them through Streamlit's resource cache.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import blend
from src.cf import get_cf_model, score_items
from src.data import load_data
from src.features import get_content_model
from src.filters import Filters


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data()


@pytest.fixture(scope="module")
def content_model():
    return get_content_model()


@pytest.fixture(scope="module")
def cf_model():
    return get_cf_model()


def _filters() -> Filters:
    return Filters(
        year_range=(2015, 2021), types=["Movie"], countries=["United States"],
        ratings=["PG-13"], genres=["Dramas"],
        all_countries=[], all_genres=[], all_ratings=[],
    )


# --- content model ---------------------------------------------------------
def test_content_matrix_rows_are_unit_length(content_model):
    norms = np.sqrt(content_model.matrix.multiply(content_model.matrix).sum(axis=1)).A1
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_content_model_is_sparse_not_dense(content_model):
    rows, cols = content_model.matrix.shape
    assert rows == len(load_data())
    # a dense n x n similarity matrix would be the OOM bug this design avoids
    assert content_model.matrix.nnz < rows * cols * 0.05


def test_vector_lookup_handles_unknown_ids(content_model):
    assert content_model.vector("does-not-exist") is None
    assert content_model.vector("s1") is not None


# --- collaborative model ---------------------------------------------------
def test_score_items_returns_none_without_a_join(cf_model):
    assert score_items(cf_model, [(999_999_999, 5.0)]) is None


def test_score_items_returns_full_length_scores(cf_model):
    movie_id = int(cf_model.movie_ids[10])
    scores = score_items(cf_model, [(movie_id, 5.0)])
    assert scores is not None
    assert scores.shape == (cf_model.n_items,)
    # an item is identical to itself, so it must be (joint) top
    pos = cf_model.item_index[movie_id]
    assert scores[pos] >= scores.max() - 1e-6


def test_attribution_points_back_into_the_input_list(cf_model):
    pairs = [(int(cf_model.movie_ids[10]), 5.0), (int(cf_model.movie_ids[20]), 1.0)]
    result = score_items(cf_model, pairs, return_attribution=True)
    scores, attribution = result
    assert scores.shape == attribution.shape
    assert set(np.unique(attribution)).issubset({0, 1})


def test_a_disliked_title_pulls_the_other_way(cf_model):
    liked_id = int(cf_model.movie_ids[10])
    loved = score_items(cf_model, [(liked_id, 5.0)])
    hated = score_items(cf_model, [(liked_id, 1.0)])
    pos = cf_model.item_index[liked_id]
    assert loved[pos] > 0 > hated[pos]


# --- blend -----------------------------------------------------------------
def test_seed_recommendations_exclude_the_seed(df):
    seed = str(df["show_id"].iloc[42])
    out = blend.recommend(df, None, seed_show_id=seed, top_n=10)
    assert len(out) == 10
    assert seed not in set(out["show_id"])
    assert not out.empty


def test_watchlisted_titles_are_excluded(df):
    from src.catalog import show_to_movie
    joined = list(show_to_movie())[:5]
    watchlist = {sid: {"rating": 5.0} for sid in joined}
    out = blend.recommend(df, None, watchlist=watchlist, top_n=10)
    assert set(out["show_id"]).isdisjoint(watchlist)


def test_result_schema_is_stable(df):
    out = blend.recommend(df, None, top_n=3)
    for column in blend.RESULT_COLUMNS:
        assert column in out.columns, f"missing {column}"
    assert out["score"].is_monotonic_decreasing or len(out) <= 1


def test_filters_are_respected(df):
    from src.filters import apply_filters

    filters = _filters()
    out = blend.recommend(df, filters, top_n=20)
    assert not out.empty

    # compare against the sidebar's own filter logic rather than re-deriving
    # it: `country` holds multi-country strings like "United States, Canada"
    allowed = set(apply_filters(df, filters)["show_id"])
    assert set(out["show_id"]) <= allowed
    assert set(out["type"]) == {"Movie"}


def test_empty_candidate_set_returns_empty_frame(df):
    impossible = Filters(
        year_range=(1920, 1921), types=["Movie"], countries=["United States"],
        ratings=["NC-17"], genres=["Western"], all_countries=[], all_genres=[],
        all_ratings=[],
    )
    out = blend.recommend(df, impossible, top_n=5)
    assert out.empty


def test_cold_start_falls_back_to_popularity(df):
    out = blend.recommend(df, None, top_n=5)
    assert (out["model"] == "Popular").all()


def test_unjoined_titles_can_still_be_recommended(df):
    """The ~86% without a MovieLens join must remain recommendable."""
    from src.catalog import show_to_movie
    joined = set(show_to_movie())
    seed = str(df["show_id"].iloc[7])
    out = blend.recommend(df, None, seed_show_id=seed, top_n=50)
    assert any(sid not in joined for sid in out["show_id"]), \
        "recommendations only ever come from the joined minority"

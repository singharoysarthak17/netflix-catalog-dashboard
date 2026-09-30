"""Item-item collaborative filtering over the MovieLens ratings matrix.

Model
-----
X          users x items, cell = the user's 0.5-5.0 rating
Xn         same matrix with every *column* L2-normalised, so
           Xn[:, i].T @ Xn[:, j] is the cosine similarity of two items

Scoring never materialises the full items x items similarity matrix
(9,742 x 9,742 float32 = ~380 MB). Instead only the rows for the titles a
user actually engaged with are computed, which is |L| x n_items -- a few
hundred KB even for a long watchlist.

The user's profile is a rating-weighted sum of those rows:
    score[j] = sum_i (rating_i - 3.5) * cos(i, j)
so a liked title pushes candidates up and a disliked one pushes them down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

RATING_CENTER = 3.5   # 0.5-5.0 half-star scale; 3.5 is "indifferent"


@dataclass
class CFModel:
    Xn: sparse.csr_matrix                 # column-normalised ratings (users x items)
    X_raw: sparse.csr_matrix              # original 0.5-5.0 ratings, same shape
    movie_ids: np.ndarray                 # column index -> movieId
    item_index: dict[int, int]            # movieId -> column index
    titles: dict[int, str]                # movieId -> MovieLens title
    popularity: np.ndarray                # normalised log rating count per column
    n_ratings: np.ndarray                 # raw rating count per column
    item_means: np.ndarray                # mean rating per column
    n_users: int

    @property
    def n_items(self) -> int:
        return self.Xn.shape[1]

    def has(self, movie_id: int) -> bool:
        return int(movie_id) in self.item_index


def build_cf_model(
    ratings_csv: Path = Path("data/movielens/ml-latest-small/ratings.csv"),
    movies_csv: Path = Path("data/movielens/ml-latest-small/movies.csv"),
) -> CFModel:
    ratings = pd.read_csv(ratings_csv)
    movies = pd.read_csv(movies_csv)

    movie_ids = movies["movieId"].to_numpy()
    item_index = {int(m): i for i, m in enumerate(movie_ids)}
    titles = dict(zip(movies["movieId"].astype(int), movies["title"]))

    user_codes = pd.Categorical(ratings["userId"]).codes
    item_codes = pd.Categorical(ratings["movieId"], categories=list(movie_ids)).codes
    if (item_codes < 0).any():
        raise ValueError("ratings.csv references movieIds missing from movies.csv")

    values = (ratings["rating"].astype(np.float32).to_numpy(),
              (user_codes, item_codes))
    X = sparse.csr_matrix(
        values, shape=(len(np.unique(user_codes)), len(movie_ids)))

    # column L2 norms -> cosine item similarity via a plain matmul
    norms = np.sqrt(X.multiply(X).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    Xn = X.multiply(1.0 / norms).tocsr().astype(np.float32)

    counts = np.bincount(item_codes, minlength=len(movie_ids)).astype(np.float64)
    sums = np.bincount(item_codes,
                       weights=ratings["rating"].to_numpy(),
                       minlength=len(movie_ids))
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    pop = np.log1p(counts)
    span = pop.max() - pop.min()
    popularity = (pop - pop.min()) / span if span > 0 else np.zeros_like(pop)

    return CFModel(
        Xn=Xn.astype(np.float32),
        X_raw=X.astype(np.float32),
        movie_ids=movie_ids,
        item_index=item_index,
        titles=titles,
        popularity=popularity.astype(np.float32),
        n_ratings=counts.astype(np.float32),
        item_means=means.astype(np.float32),
        n_users=X.shape[0],
    )


def score_items(
    model: CFModel,
    liked: list[tuple[int, float]],
    return_attribution: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
    """CF score for every item given (movieId, rating) pairs.

    Returns None when nothing in `liked` is a MovieLens title -- the caller
    then falls back to content + popularity, which is the designed
    degradation path for the 86% of the catalog without a join.

    With `return_attribution`, also returns for each item the index (into
    `liked`) of the title that contributed most positively, so the UI can
    say "Because you liked ...".
    """
    positions, weights, kept = [], [], []
    for i, (movie_id, rating) in enumerate(liked):
        pos = model.item_index.get(int(movie_id))
        if pos is None:
            continue
        weight = float(rating) - RATING_CENTER
        if abs(weight) < 1e-9:
            continue
        positions.append(pos)
        weights.append(weight)
        kept.append(i)

    if not positions:
        return None

    block = model.Xn[:, positions].T @ model.Xn     # (|L|, n_items) sparse
    dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
    w = np.asarray(weights, dtype=np.float32)

    scores = w @ dense
    if not return_attribution:
        return scores

    # Most *positive* contributor per candidate (not largest magnitude:
    # a disliked title must not end up as the stated reason).
    best = np.argmax(dense, axis=0)
    attribution = np.asarray([kept[b] for b in best], dtype=np.int64)
    return scores, attribution


# Lazy: only built when CF signal is actually needed (watchlist with
# joined titles). Avoids ~100MB sparse matrix on cold start for users
# who haven't built a matched watchlist yet.

_cached: CFModel | None = None


def get_cf_model() -> CFModel:
    """Build on first call; ratings.csv is 100k rows so this is quick."""
    global _cached
    if _cached is None:
        _cached = build_cf_model()
    return _cached

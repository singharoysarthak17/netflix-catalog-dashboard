"""Content-based similarity model.

Builds one sparse matrix over the whole catalog combining:
  * TF-IDF over description + genres + director + cast + country + rating
  * multi-hot blocks for genres, countries and rating

Rows are L2-normalised, so a dot product between two rows is cosine
similarity. Scoring is always "dot product against an allowed-candidate
mask", which lets filters be applied at query time without rebuilding
anything.

Memory: the matrix is sparse end to end (~2-4M non-zeros). A dense
n x n similarity matrix is never materialised -- it would be ~310 MB for
8,809 titles and OOM the Streamlit Community Cloud instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Relative importance of each block after per-block L2 normalisation.
W_TEXT = 1.0
W_GENRE = 0.9
W_COUNTRY = 0.5
W_RATING = 0.4


@dataclass
class ContentModel:
    matrix: sparse.csr_matrix          # (n_titles, dim), rows L2-normalised
    show_ids: np.ndarray               # positional index -> show_id
    index_of: dict[str, int]           # show_id -> positional index

    @property
    def n_titles(self) -> int:
        return self.matrix.shape[0]

    def vector(self, show_id: str) -> sparse.csr_matrix | None:
        """Row vector for one title, as a 1 x dim sparse matrix."""
        pos = self.index_of.get(show_id)
        if pos is None:
            return None
        return self.matrix[pos:pos + 1]


def _build_text(df: pd.DataFrame) -> sparse.csr_matrix:
    def join(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str)

    text = (
        join(df["description"]) + " " +
        join(df["listed_in"]) + " " +
        join(df["director"]) + " " +
        join(df["cast"]) + " " +
        join(df["country"]) + " " +
        join(df["rating"])
    )
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w']*\b",
    )
    return vectorizer.fit_transform(text).tocsr()


def _build_block(series: pd.Series) -> sparse.csr_matrix:
    """Multi-hot one column of list-valued cells (scalars are wrapped)."""
    mlb = MultiLabelBinarizer()

    def as_list(value: object) -> list:
        if isinstance(value, list):
            return value
        if value is None or value == "" or pd.isna(value):
            return []
        return [value]

    block = mlb.fit_transform(series.map(as_list))
    return sparse.csr_matrix(block.astype(np.float32))


def _l2_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms).astype(np.float32).dot(matrix).tocsr()


def build_content_model(df: pd.DataFrame) -> ContentModel:
    text = _build_text(df)
    genres = _build_block(df["genres"])
    countries = _build_block(df["countries"])
    rating = _build_block(df["rating"])

    # normalise each block so weights mean the same thing across blocks
    blocks = []
    for block, weight in ((text, W_TEXT), (genres, W_GENRE),
                          (countries, W_COUNTRY), (rating, W_RATING)):
        blocks.append(_l2_normalize(block) * weight)

    combined = sparse.hstack(blocks, format="csr").tocsr()
    combined = _l2_normalize(combined)

    show_ids = df["show_id"].to_numpy()
    return ContentModel(
        matrix=combined,
        show_ids=show_ids,
        index_of={sid: i for i, sid in enumerate(show_ids)},
    )


def _build_default() -> ContentModel:
    from src.data import load_data

    return build_content_model(load_data())


# No arguments on purpose: st.cache_resource re-hashes its args on every
# call, and hashing all 8,800 rows (descriptions included) on each rerun
# would cost more than building the matrix once.
_cached_builder = st.cache_resource(_build_default)


def get_content_model() -> ContentModel:
    """Build once per process; subsequent reruns reuse the matrix."""
    return _cached_builder()


def score_against(model: ContentModel, query: sparse.csr_matrix) -> np.ndarray:
    """Cosine similarity of every catalog row against a 1 x dim query."""
    return np.asarray((model.matrix @ query.T).todense()).ravel()

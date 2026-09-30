"""Offline evaluation of the collaborative model.

There is no ground truth for "what a Netflix user would watch" -- that data
does not exist in the catalog -- but MovieLens *does* carry real ratings, so
the CF component can be held to a standard protocol:

  leave-one-out     for each user, hide one item they rated >= 3.5,
                    recommend k from the items they have not rated, score
                    whether the hidden item appears

  baseline          the same protocol using a popularity-only ranker, so
                    every number below is "vs. always recommend the most
                    rated films"

Reported: precision@k, recall@k, NDCG@k, catalog coverage, novelty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.cf import CFModel, RATING_CENTER

RELEVANT_AT = 3.5
K = 10
SEED = 42
MIN_USER_RATINGS = 20


@dataclass
class EvalResult:
    users: int
    k: int
    precision: float
    recall: float
    ndcg: float
    baseline_precision: float
    baseline_ndcg: float
    coverage: float
    novelty: float

    @property
    def lift(self) -> float:
        """Precision relative to the popularity baseline (1.0 = no better)."""
        if self.baseline_precision <= 0:
            return float("inf")
        return self.precision / self.baseline_precision

    def as_dict(self) -> dict:
        return asdict(self)


def _leave_one_out(model: CFModel, rng: np.random.Generator) -> EvalResult:
    from scipy import sparse

    raw = model.X_raw.tocsr()
    counts = model.n_ratings.astype(np.float64)
    rated_total = max(counts.sum(), 1.0)

    hits, ndcgs, baseline_hits, baseline_ndcgs = [], [], [], []
    covered: set[int] = set()
    novelties: list[float] = []

    activity = np.diff(raw.indptr)          # ratings per user
    eligible = np.flatnonzero(activity >= MIN_USER_RATINGS)

    for user in eligible:
        start, end = raw.indptr[user], raw.indptr[user + 1]
        rated = raw.indices[start:end]
        values = raw.data[start:end]
        if rated.size < 2:
            continue

        relevant = rated[values >= RELEVANT_AT]
        if relevant.size == 0:
            continue

        held_out = int(relevant[int(rng.integers(0, relevant.size))])

        keep = rated != held_out
        positions = rated[keep]
        weights = (values[keep] - RATING_CENTER).astype(np.float32)
        if positions.size == 0 or not np.any(weights):
            continue

        block = model.Xn[:, positions].T @ model.Xn
        dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
        scores = weights @ dense
        # mask everything already rated EXCEPT the held-out item -- it has
        # to stay eligible or recall is identically zero by construction
        scores[positions] = -np.inf
        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])]

        base = counts.copy()
        base[positions] = -np.inf                      # popularity baseline
        btop = np.argpartition(-base, K)[:K]
        btop = btop[np.argsort(-base[btop])]

        for arr, hit_list, ndcg_list in (
            (top, hits, ndcgs),
            (btop, baseline_hits, baseline_ndcgs),
        ):
            where = np.flatnonzero(arr == held_out)
            rank = int(where[0]) if where.size else -1
            hit_list.append(1 if rank >= 0 else 0)
            ndcg_list.append(1.0 / np.log2(rank + 2) if rank >= 0 else 0.0)

        covered.update(int(i) for i in top)
        novelties.append(float(-np.log2((counts[top] + 1.0) / rated_total).mean()))

    n = max(len(hits), 1)
    return EvalResult(
        users=len(hits),
        k=K,
        precision=float(np.mean(hits)) / K,
        recall=float(np.mean(hits)),
        ndcg=float(np.mean(ndcgs)),
        baseline_precision=float(np.mean(baseline_hits)) / K,
        baseline_ndcg=float(np.mean(baseline_ndcgs)),
        coverage=len(covered) / max(model.n_items, 1),
        novelty=float(np.mean(novelties)) if novelties else 0.0,
    )


def run_evaluation() -> EvalResult:
    """Leave-one-out over MovieLens. Runs once per process."""
    from src.cf import build_cf_model

    model = build_cf_model()
    rng = np.random.default_rng(SEED)
    return _leave_one_out(model, rng)


# Evaluation is lazy: only computed when the "How It Works" tab is viewed.
# The result is cached in st.session_state so the 17s leave-one-out runs once
# per session, not once per process (which would hit the first user's request).

def get_evaluation() -> EvalResult:
    """Get or compute the evaluation result (lazy, per-session)."""
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = run_evaluation()
    return st.session_state["eval_result"]


def summary_table() -> pd.DataFrame:
    """Metrics side by side with the popularity baseline, for the UI."""
    result = get_evaluation()
    rows = [
        ("Precision@10", f"{result.precision:.4f}",
         f"{result.baseline_precision:.4f}", f"{result.lift:.2f}x"),
        ("Recall@10", f"{result.recall:.4f}", "-", "-"),
        ("NDCG@10", f"{result.ndcg:.4f}", f"{result.baseline_ndcg:.4f}",
         f"{result.ndcg / result.baseline_ndcg:.2f}x"
         if result.baseline_ndcg else "-"),
        ("Catalog coverage", f"{result.coverage:.1%}", "-", "-"),
        ("Novelty (bits)", f"{result.novelty:.2f}", "-", "-"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "This model", "Popularity baseline", "Lift"])

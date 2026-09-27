"""Headless smoke check of the recommendation pipeline.

Run: python scripts/smoke_test.py
Builds both models, exercises seed/watchlist/no-input paths and prints the
evaluation block. Fails loudly on any exception.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    from src.data import load_data
    from src.filters import Filters, apply_filters
    from src.blend import recommend

    df = load_data()
    print(f"[load] {len(df):,} titles in {time.perf_counter() - t0:.1f}s")

    t = time.perf_counter()
    from src.features import get_content_model
    cm = get_content_model()
    print(f"[content] matrix {cm.matrix.shape} "
          f"nnz={cm.matrix.nnz:,} ({cm.matrix.nnz * 12 / 1e6:.0f} MB est) "
          f"in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    from src.cf import get_cf_model
    cf = get_cf_model()
    print(f"[cf] {cf.n_users} users x {cf.n_items} items, "
          f"{int(cf.n_ratings.sum()):,} ratings in {time.perf_counter() - t:.1f}s")

    # --- path 1: seed title -------------------------------------------
    seed = df[df["type"] == "Movie"]["show_id"].iloc[100]
    t = time.perf_counter()
    res = recommend(df, None, seed_show_id=seed, top_n=5)
    print(f"\n[seed] {df.loc[df.show_id == seed, 'title'].iloc[0]!r} "
          f"in {time.perf_counter() - t:.2f}s")
    for r in res.itertuples():
        print(f"   {r.score:5.3f} [{r.model:24}] {r.title[:38]:38} | {r.reason[:60]}")

    # --- path 2: watchlist with rated, joined titles --------------------
    from src.catalog import load_mapping, show_to_movie
    mapping = load_mapping()
    joined = show_to_movie(mapping)
    sample_shows = list(joined)[:5]
    watchlist = {sid: {"rating": 5.0 if i % 2 == 0 else 2.0}
                 for i, sid in enumerate(sample_shows)}
    t = time.perf_counter()
    res = recommend(df, None, watchlist=watchlist, top_n=5)
    print(f"\n[watchlist] {len(watchlist)} rated titles "
          f"in {time.perf_counter() - t:.2f}s")
    for r in res.itertuples():
        print(f"   {r.score:5.3f} [{r.model:24}] {r.title[:34]:34} "
              f"c={r.content_score:.3f} cf={r.cf_score:.3f} p={r.popularity_score:.3f}"
              f" | {r.reason[:44]}")
    if "cf_score" in res and res["cf_score"].sum() == 0:
        raise AssertionError("collaborative component contributed nothing")

    # --- path 3: no input at all ---------------------------------------
    t = time.perf_counter()
    res = recommend(df, None, top_n=5)
    print(f"\n[popular] in {time.perf_counter() - t:.2f}s")
    for r in res.itertuples():
        print(f"   {r.score:5.3f} [{r.model:24}] {r.title[:38]:38} | {r.reason[:60]}")

    # --- path 4: filters ------------------------------------------------
    filters = Filters(
        year_range=(2015, 2021), types=["Movie"], countries=["United States"],
        ratings=["PG-13"], genres=["Dramas"],
        all_countries=[], all_genres=[], all_ratings=[],
    )
    subset = apply_filters(df, filters)
    res = recommend(df, filters, seed_show_id=seed, top_n=5)
    print(f"\n[filters] {len(subset):,} titles pass filters -> "
          f"{len(res)} recommendations")
    for r in res.itertuples():
        print(f"   {r.score:5.3f} [{r.model:24}] {r.title[:38]:38}")

    # --- evaluation ------------------------------------------------------
    t = time.perf_counter()
    from src.evaluate import get_evaluation, summary_table
    ev = get_evaluation()
    print(f"\n[eval] leave-one-out over {ev.users} users "
          f"in {time.perf_counter() - t:.1f}s")
    print(summary_table().to_string(index=False))
    print(f"       precision lift vs popularity baseline: {ev.lift:.2f}x")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

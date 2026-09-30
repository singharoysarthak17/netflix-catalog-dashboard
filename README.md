# 📺 Netflix Catalog & Recommendations

A Streamlit analytics app that **recommends what to watch next** on top of the
original catalog dashboard.

It is a real recommender, not a filter: content similarity over the Netflix
catalog, collaborative filtering over 100,836 MovieLens ratings, and a
popularity prior are blended into one ranking — and the app tells you which of
the three actually drove each result.

![Recommendations](images/overview.png)

---

## 🚀 Live Demo

https://netflix-catalog-dashboard-rx4sadc2kg3a4appmf6pfqd.streamlit.app

---

## ✨ What's in the app

The app opens on **Recommendations**; the original four dashboard tabs are one
click away under **Dashboard**.

| Tab | What it does |
| --- | --- |
| ✨ **For You** | Ranked for your watchlist: content + collaborative + popularity blend |
| 🎯 **Because You Watched** | Item-to-item neighbours of any title you pick |
| 🍿 **My Watchlist** | Persistent per-user list: star ratings, JSON export/import, shareable key link |
| 🔬 **How It Works** | Blend weights, join coverage, offline evaluation, limitations, citation |
| 🔍 **Find Similar Titles** | Seed title + genre/country/mood profile → six content matches on a card grid |
| 📊 **Dashboard** | Overview · Country/Rating · Growth · Trends (the original four tabs) |

The shared sidebar filters (year added, type, country, rating, genre) stay live
on every page.

### Find Similar Titles

A second, deliberately simpler path: pick a **seed title**, optionally narrow it
with **genre**, **country** and **mood**, and get six neighbours back.

```
score = cosine(seed, candidate)                 # the existing content model
      + 0.10  if the candidate shares a selected genre
      + 0.06  if it shares the selected country
      + 0.08  if it matches the mood's genres or keywords
```

- The boosts **re-rank, they never filter** — an over-specified profile still
  returns a full grid instead of an empty one.
- Moods are a fixed editorial map (`src/similar.py`): `suspenseful`,
  `light-hearted`, `dark`, `feel-good`, `intense` → genres + prose keywords.
- A seed with too little text to vectorize falls back to **genre + country**
  matching rather than returning nothing.
- Stateless: the last requested selection lives in the **URL**
  (`?sim_seed=…&sim_genres=…`), so results survive reruns and can be
  bookmarked. Same inputs always draw the same grid.

### How the ranking is built

```
final = 0.55 · content + 0.30 · collaborative + 0.15 · popularity
```

- **Content (0.55)** — TF-IDF over description, genres, director, cast,
  country and rating, plus multi-hot genre/country/rating blocks, cosine over a
  *sparse* matrix (no `n_titles × n_titles` dense object).
- **Collaborative (0.30)** — item-item cosine on a column-normalized sparse
  ratings matrix from MovieLens. The weight scales up as
  `min(1, matched_rated_titles / 3)`, so it contributes only once you've rated
  at least three titles that join to MovieLens.
- **Popularity (0.15)** — log rating count per MovieLens item, used as a prior.
  Titles with no join get a neutral `0.5` rather than being pushed down.

Each card shows **which signal scored that title highest**, so a
collaborative result is visibly distinguishable from a content match.

### Offline evaluation

Leave-one-out over 609 MovieLens users: hide one item a user rated ≥ 3.5, rank
everything they haven't rated, check the top 10.

| Metric | This model | Popularity baseline | Lift |
| --- | ---: | ---: | ---: |
| Precision@10 | 0.0212 | 0.0107 | **1.98×** |
| NDCG@10 | 0.1437 | 0.0586 | **2.45×** |
| Recall@10 | 0.2118 | — | — |
| Catalog coverage | 17.2 % | — | — |

Numbers from `python scripts/smoke_test.py` on `ml-latest-small`.

---

## 🍿 Your watchlist

Your list is keyed by a **random 22-character key that appears in the URL**:

```
http://localhost:8511/?w=<your-key>
```

- Bookmark that URL (or copy it from the **My Watchlist** tab) to get your list
  back on another device — it *is* your identity, there are no accounts.
- **Anyone with the link can read and edit the list.** Treat it like a password.
  There is no recovery: lose the key, lose the list.
- Storage auto-selects: **GitHub Gist** (if configured) → **local SQLite** →
  **this browser tab**. Any failure degrades to the next one and the app tells
  you which backend is live.
- Export/import as JSON from the My Watchlist tab.

### Optional: shared watchlists across devices via a Gist

Add `GITHUB_TOKEN` (scope `gist`) and `GIST_ID` to `.streamlit/secrets.toml`:

```toml
GITHUB_TOKEN = "ghp_..."
GIST_ID = "abc123..."
```

Without them the app still works — it just falls back to SQLite/session
storage.

---

## 📂 Datasets

**Netflix catalog** — `netflix_titles.csv` (Kaggle), 8,807 raw rows and
**8,709 titles** after dropping the 98 rows with no usable `date_added`.

**MovieLens `ml-latest-small`** — committed under `data/movielens/`
(redistributable). Ratings from 610 anonymized users, 9,742 movies, ~100k
ratings, 2018-era. Fetch it fresh with:

```bash
python scripts/download_movielens.py
```

> 📖 **Citation** — F. Maxwell Harper and Joseph A. Konstan. 2015. The
> MovieLens Datasets: History and Context. *ACM Transactions on Interactive
> Intelligent Systems* 5, 4, Article 19 (December 2015), 19 pages.
> https://doi.org/10.1145/2827872

**Larger optional set** — `ml-25m` (~25M ratings) is **not** redistributable
and is gitignored. Drop it at `data/movielens/ml-25m/` locally if you want to
rebuild the recommender against it; derived artifacts from it stay ignored too.

No API keys are required — there is no TMDB/external enrichment step. Matching
is title-only.

---

## 🧠 The title join (why only ~14 % of titles have collaborative signal)

MovieLens and the Netflix catalog don't share an ID, so
`scripts/build_title_index.py` matches them on normalised titles with a
**precision-first** rule set:

- exact normalised title (year window ±3) wins outright — normalisation folds
  accents, drops articles in either position (`The Departed` ≡ `Departed, The`),
  and expands `&` → `and`;
- token-set/token-sort fuzzy scores must clear a higher bar, have a clear margin
  over the runner-up, agree on genre Jaccard, and pass anti-over-match rules
  (title nesting, year mismatch, subset evidence);
- TV shows are never matched — MovieLens has none.

Measured result with `ml-latest-small`:

| Bucket | Matched |
| --- | ---: |
| pre-2000 | 44.3 % |
| 2000–2009 | 41.1 % |
| 2010–2014 | 22.7 % |
| 2015–2018 | 6.4 % |
| 2019+ | 0.4 % |
| **All movies** | **878 / 6,126 = 14.3 %** |

14.3 % is below the 30 % gate set in the plan, so **content leads and CF is
weighted second** (0.30) — the plan's gate, not a guess. Re-run with
`python scripts/build_title_index.py`; borderline cases land in
`data/movielens/title_review.csv` for manual review, and hand-checked decisions
go in `title_overrides.csv`.

---

## 📦 Installation

```bash
git clone https://github.com/singharoysarthak17/netflix-catalog-dashboard.git
cd netflix-catalog-dashboard
pip install -r requirements.txt
streamlit run app.py
```

Dependencies are **pinned** — Streamlit Cloud rebuilds from this file, and an
unpinned `streamlit`/`plotly` pair is how a working app breaks itself.

---

## ✅ Testing

```bash
pytest                          # 46 tests: join, content, CF, blend, storage, similarity
pytest -m slow                  # + the full leave-one-out evaluation (~20s)
python scripts/smoke_test.py    # headless end-to-end: models, all rec paths, eval
python scripts/ui_smoke_test.py # renders both pages via streamlit AppTest
```

---

## 🗂️ Project structure

```
app.py                    entry point: sidebar + st.navigation
views/                    thin page scripts (Streamlit executes these)
ui/                       import-safe renderers (recommend.py, dashboard.py,
                          similar.py)
src/
  data.py  filters.py     load + filter the catalog
  features.py             TF-IDF / multi-hot content model (sparse)
  similar.py              seed-title similarity + soft genre/country/mood boost
  cf.py                   item-item CF over MovieLens (sparse, column-normalized)
  blend.py                weights, ranking, reasons, dominant-signal badges
  catalog.py              Netflix ↔ MovieLens join helpers
  storage.py              Gist / SQLite / session watchlist backends
  explain.py  evaluate.py reason strings, leave-one-out evaluation
scripts/
  download_movielens.py   fetch ml-latest-small (certifi TLS workaround)
  build_title_index.py    title join → title_mapping.csv
  smoke_test.py           headless pipeline check
  ui_smoke_test.py        headless UI check (streamlit.testing.v1)
tests/                    pytest suite
data/movielens/           ml-latest-small + join artifacts
```

---

## ⚠️ Limitations

- Ratings come from **610 anonymized MovieLens users (through 2018)** — not
  Netflix viewing data, which isn't public. Popularity ≠ "popular on Netflix".
- ~86 % of the catalog has no collaborative signal and is scored on content
  alone. The title join is deliberately precision-first: a few hundred
  borderline titles were *rejected* rather than guessed.
- MovieLens covers **movies only**, so TV shows never receive a CF score.
- Watchlist keys are bearer secrets — no accounts, no auth, no recovery.
- Not affiliated with Netflix or GroupLens.

---

## 📸 Dashboard Preview

### Overview

![Overview](images/overview.png)

### Country Rating

![Country Rating](images/country-rating.png)

### Growth

![Growth](images/growth-trend.png)

### Trends

![Trends](images/movie-tv-trend.png)

---

## 👤 Author

Sarthak Singha Roy

B.Tech CSE | KIIT University

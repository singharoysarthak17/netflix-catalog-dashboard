"""Build the Netflix <-> MovieLens title mapping used by the recommender.

Fully offline: no API keys, no network calls.

    python scripts/build_title_index.py

Stages
    1. normalise titles on both sides (case, punctuation, unicode, articles)
    2. block candidates to a +/-3 year window around release_year
    3. score with rapidfuzz token-set ratio
    4. validate with a genre-overlap check (Netflix listed_in vs MovieLens genres)
    5. reject ambiguous matches (thin margin between best and runner-up)
    6. apply the hand-curated overrides file
    7. emit a coverage report + a review queue for the rejected titles

Outputs (all committed)
    data/movielens/title_mapping.csv    show_id -> movieId (auto + overrides)
    data/movielens/title_review.csv     rejected/ambiguous titles for a human pass
    data/movielens/title_overrides.csv  hand-curated show_id -> movieId fixes
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

REPO_ROOT = Path(__file__).resolve().parents[1]
NETFLIX_CSV = REPO_ROOT / "netflix_titles.csv"
MOVIELENS_DIR = REPO_ROOT / "data" / "movielens" / "ml-latest-small"

MAPPING_CSV = REPO_ROOT / "data" / "movielens" / "title_mapping.csv"
REVIEW_CSV = REPO_ROOT / "data" / "movielens" / "title_review.csv"
OVERRIDES_CSV = REPO_ROOT / "data" / "movielens" / "title_overrides.csv"

# Tunables ---------------------------------------------------------------
YEAR_WINDOW = 3      # +/- years of slack when blocking candidates
STRONG_SCORE = 95    # genre agreement not required above this
ACCEPT_SCORE = 90    # minimum fuzz score with genre agreement
MARGIN = 5           # minimum gap between best and runner-up (>= 8 below STRONG)
REVIEW_SCORE = 75    # below this we do not even queue a suggestion

# --- genre vocabularies --------------------------------------------------
# MovieLens uses a fixed 20-genre vocabulary; Netflix uses free-text labels.
ML_GENRES = {
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
}

TOKEN_TO_GENRE = {
    "action": "Action", "adventure": "Adventure", "animation": "Animation",
    "anime": "Animation", "children": "Children", "child": "Children",
    "kid": "Children", "kids": "Children", "family": "Children",
    "comedy": "Comedy", "comical": "Comedy", "stand": "Comedy",
    "crime": "Crime", "criminal": "Crime",
    "documentary": "Documentary", "documentaries": "Documentary",
    "docuseries": "Documentary", "drama": "Drama",
    "fantasy": "Fantasy", "noir": "Film-Noir", "horror": "Horror",
    "musical": "Musical", "music": "Musical", "mystery": "Mystery",
    "romance": "Romance", "romantic": "Romance",
    "scifi": "Sci-Fi", "sci": "Sci-Fi", "thriller": "Thriller",
    "war": "War", "western": "Western",
}

ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
TRAILING_ARTICLE = re.compile(r"\s+(the|a|an)$", re.IGNORECASE)
YEAR_SUFFIX = re.compile(r"\s*\((\d{4})\)\s*$")
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(raw: object) -> str:
    """Aggressive normalisation so 'The Bipolar Express' ~ 'Bipolar Express, The'.

    Handles MovieLens's inverted-article convention ('Departed, The (2006)')
    as well as Netflix's plain form ('The Departed'); both must land on the
    same key or those titles never take the exact-match path.
    """
    if not isinstance(raw, str):
        return ""
    text = unicodedata.normalize("NFKD", raw)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = YEAR_SUFFIX.sub(" ", text)
    text = text.replace("&", " and ")
    text = NON_ALNUM.sub(" ", text).strip()
    text = ARTICLES.sub("", text)                  # leading article
    text = TRAILING_ARTICLE.sub("", text).strip()  # 'departed, the'
    return " ".join(text.split())


def netflix_genres(label: object) -> set[str]:
    """Map a Netflix listed_in string onto the MovieLens genre vocabulary.

    Handles the plural forms Netflix uses ('Dramas', 'Comedies', 'Thrillers')
    by falling back to the singular before giving up on a token.
    """
    if not isinstance(label, str) or not label.strip():
        return set()
    found: set[str] = set()
    for token in re.split(r"[^a-z]+", label.lower()):
        if not token:
            continue
        genre = TOKEN_TO_GENRE.get(token)
        if genre is None and token.endswith("s"):
            genre = TOKEN_TO_GENRE.get(token[:-1])
        if genre:
            found.add(genre)
    return found


def movielens_genres(label: object) -> set[str]:
    if not isinstance(label, str):
        return set()
    return {g for g in label.split("|") if g in ML_GENRES}


def genre_jaccard(a: set[str], b: set[str]) -> float | None:
    """None means 'no evidence' (one side has no mappable genres) -> do not penalise."""
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def extract_year(title: str) -> int | None:
    match = YEAR_SUFFIX.search(title if isinstance(title, str) else "")
    return int(match.group(1)) if match else None


def load_netflix() -> pd.DataFrame:
    df = pd.read_csv(NETFLIX_CSV)
    df["norm_title"] = df["title"].map(normalize_title)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")
    df["genre_set"] = df["listed_in"].map(netflix_genres)
    return df[df["norm_title"] != ""].dropna(subset=["norm_title"]).reset_index(drop=True)


def load_movielens() -> pd.DataFrame:
    df = pd.read_csv(MOVIELENS_DIR / "movies.csv")
    df["norm_title"] = df["title"].map(normalize_title)
    df["ml_year"] = df["title"].map(extract_year)
    df["genre_set"] = df["genres"].map(movielens_genres)
    return df[df["norm_title"] != ""].dropna(subset=["norm_title"]).reset_index(drop=True)


def match(nf: pd.DataFrame, ml: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (accepted, review_queue).

    Only Netflix *movies* are scored: MovieLens contains no TV shows, so
    scanning them would produce nothing but noise in the review queue.
    """
    # Position-based arrays: ~40x faster than repeated .loc in the hot loop.
    def _yr(v) -> int | None:
        if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)

    ml_titles: list[str] = ml["norm_title"].tolist()
    ml_years: list[int | None] = [_yr(v) for v in ml["ml_year"].tolist()]
    ml_genres: list[set[str]] = ml["genre_set"].tolist()
    ml_ids: list[int] = ml["movieId"].tolist()
    ml_display: list[str] = ml["title"].tolist()

    ml_by_title: dict[str, list[int]] = {}
    ml_by_year: dict[int | None, list[int]] = {}
    for pos, title in enumerate(ml_titles):
        ml_by_title.setdefault(title, []).append(pos)
        ml_by_year.setdefault(ml_years[pos], []).append(pos)
    undated = ml_by_year.get(None, ())

    def candidate_positions(title: str, year: int | None) -> list[int]:
        """Exact-title hits from any year, plus a +/-YEAR_WINDOW scan."""
        ids: set[int] = set(ml_by_title.get(title, ()))
        if year is not None:
            for y in range(int(year) - YEAR_WINDOW, int(year) + YEAR_WINDOW + 1):
                ids.update(ml_by_year.get(y, ()))
            ids.update(undated)
        return list(ids)

    accepted_rows: list[dict] = []
    review_rows: list[dict] = []

    movies = nf[nf["type"] == "Movie"]
    for _, n in movies.iterrows():
        n_title = n["norm_title"]
        n_year = n["release_year"]
        n_year_int = int(n_year) if n_year is not None and not pd.isna(n_year) else None
        n_genres: set[str] = n["genre_set"]

        candidates: dict[int, float] = {}
        subset_flags: dict[int, bool] = {}
        exact_year_mismatch = False
        for pos in candidate_positions(n_title, n_year_int):
            m_year = ml_years[pos]
            exact = ml_titles[pos] == n_title
            if n_year_int is not None and m_year is not None:
                if abs(m_year - n_year_int) > YEAR_WINDOW:
                    # Remake / re-issue protection: same name, different film.
                    if exact:
                        exact_year_mismatch = True
                    continue
            m_title = ml_titles[pos]
            if exact:
                score, is_subset = 100.0, False
            else:
                # Ensemble: token_sort catches reordered titles
                # ('The Departed' vs 'Departed, The'), token_set catches a
                # short Netflix title contained in a longer MovieLens one
                # ('Borat' vs 'Borat: Cultural Learnings...'). token_set alone
                # would score 100 on any token subset, so anything it lifts
                # above token_sort is flagged as a subset match and held to a
                # stricter acceptance bar below.
                sort_score = float(fuzz.token_sort_ratio(n_title, m_title))
                set_score = float(fuzz.token_set_ratio(n_title, m_title))
                is_subset = set_score > sort_score + 1
                score = max(sort_score, set_score)
            if score >= REVIEW_SCORE:
                candidates[pos] = score
                subset_flags[pos] = is_subset

        if not candidates:
            review_rows.append({
                "show_id": n["show_id"], "netflix_title": n["title"],
                "release_year": n_year_int, "listed_in": n["listed_in"],
                "type": n["type"], "best_movieId": "", "best_title": "",
                "best_score": "", "runner_up_score": "", "genre_jaccard": "",
                "reason": "year mismatch on exact title" if exact_year_mismatch
                          else "no candidate",
            })
            continue

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        best_pos, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        is_exact = ml_titles[best_pos] == n_title
        subset_flag = subset_flags.get(best_pos, False)
        ml_tokens = set(ml_titles[best_pos].split())
        nf_tokens = set(n_title.split())
        # MovieLens entries carry the official/complete name, so a MovieLens
        # title strictly nested inside a longer Netflix one is the wrong film
        # ('Bad Genius' (2017) vs MovieLens 'Genius', 'Nowhere Boy' vs 'Boy').
        # The reverse direction (Netflix 'Borat' inside MovieLens
        # 'Borat: Cultural Learnings...') is legitimate and stays allowed.
        ml_nested_in_nf = bool(ml_tokens) and ml_tokens < nf_tokens
        jac = genre_jaccard(n_genres, ml_genres[best_pos])
        margin = best_score - second_score
        conflict = jac == 0.0

        record = {
            "show_id": n["show_id"], "netflix_title": n["title"],
            "release_year": n_year_int, "listed_in": n["listed_in"],
            "type": n["type"],
            "best_movieId": ml_ids[best_pos], "best_title": ml_display[best_pos],
            "best_score": round(best_score, 1),
            "runner_up_score": round(second_score, 1),
            "genre_jaccard": "" if jac is None else round(jac, 2),
        }

        if is_exact:
            # Exact normalised title inside the year window is decisive:
            # Netflix and MovieLens use different genre labels, so a genre
            # disagreement or a duplicated MovieLens entry must not veto it.
            record["source"] = "auto"
            accepted_rows.append(record)
        elif conflict:
            record["reason"] = "genre conflict"
            review_rows.append(record)
        elif ml_nested_in_nf:
            record["reason"] = "MovieLens title nested inside Netflix title (likely different film)"
            review_rows.append(record)
        elif subset_flag and not (best_score >= STRONG_SCORE and margin >= MARGIN * 2):
            # A strict token subset ('Green Lantern' inside
            # 'Green Lantern: First Flight') is how token_set manufactures
            # 100s for different films -- demand a near-perfect score *and*
            # a wide margin before believing it.
            record["reason"] = f"subset match, insufficient evidence " \
                               f"(score {best_score:.0f}, margin {margin:.0f})"
            review_rows.append(record)
        elif best_score >= STRONG_SCORE and margin >= MARGIN:
            record["source"] = "auto"
            accepted_rows.append(record)
        elif best_score >= ACCEPT_SCORE and margin >= MARGIN + 3:
            record["source"] = "auto"
            accepted_rows.append(record)
        elif best_score < ACCEPT_SCORE:
            record["reason"] = f"score {best_score:.0f} < {ACCEPT_SCORE}"
            review_rows.append(record)
        else:
            record["reason"] = f"ambiguous (margin {margin:.0f})"
            review_rows.append(record)

    return pd.DataFrame(accepted_rows), pd.DataFrame(review_rows)


def apply_overrides(accepted: pd.DataFrame, ml: pd.DataFrame, nf: pd.DataFrame) -> pd.DataFrame:
    """Merge hand-curated show_id -> movieId fixes; overrides always win."""
    valid_ids = set(ml["movieId"].astype(int))
    valid_shows = set(nf["show_id"])

    if not OVERRIDES_CSV.exists() or OVERRIDES_CSV.stat().st_size == 0:
        return accepted.sort_values("show_id") if not accepted.empty else accepted

    overrides = pd.read_csv(OVERRIDES_CSV)
    required = {"show_id", "movieId"}
    if not required.issubset(overrides.columns):
        raise SystemExit(f"{OVERRIDES_CSV.name} must have columns {sorted(required)}")
    overrides = overrides.dropna(subset=["show_id", "movieId"])
    if overrides.empty:
        return accepted.sort_values("show_id") if not accepted.empty else accepted

    bad_id = sorted(set(overrides["movieId"].astype(int)) - valid_ids)
    if bad_id:
        raise SystemExit(f"{OVERRIDES_CSV.name} references unknown movieId(s): {bad_id}")
    bad_show = sorted(set(overrides["show_id"]) - valid_shows)
    if bad_show:
        raise SystemExit(f"{OVERRIDES_CSV.name} references unknown show_id(s): {bad_show}")

    overrides = overrides.assign(
        show_id=overrides["show_id"].astype(str),
        movieId=overrides["movieId"].astype(int),
    )
    if accepted.empty:
        keep = accepted
    else:
        keep = accepted[~accepted["show_id"].astype(str).isin(overrides["show_id"])]

    patched = overrides[["show_id", "movieId"]].merge(
        ml[["movieId", "title"]].rename(columns={"title": "best_title"}),
        on="movieId", how="left",
    )
    patched["best_movieId"] = patched["movieId"]
    patched = patched.drop(columns=["movieId"])
    patched["source"] = "override"
    patched["netflix_title"] = patched["show_id"].map(dict(zip(nf["show_id"], nf["title"])))
    patched["release_year"] = patched["show_id"].map(
        dict(zip(nf["show_id"], nf["release_year"]))
    )
    patched["best_score"] = ""
    patched["genre_jaccard"] = ""

    merged = pd.concat([keep, patched], ignore_index=True, sort=False)
    return merged.sort_values("show_id")


def report(accepted: pd.DataFrame, review: pd.DataFrame, nf: pd.DataFrame) -> None:
    total = len(nf)
    movies = nf[nf["type"] == "Movie"]
    tv = nf[nf["type"] == "TV Show"]
    matched_ids = set(accepted["show_id"]) if not accepted.empty else set()

    print("\n" + "=" * 64)
    print("TITLE JOIN REPORT")
    print("=" * 64)
    print(f"Netflix titles        : {total:,}")
    print(f"MovieLens titles      : 9,742 (ml-latest-small)")

    n_movies_matched = len(set(movies["show_id"]) & matched_ids)
    n_tv_matched = len(set(tv["show_id"]) & matched_ids)

    print(f"\nNetflix MOVIES matched: {n_movies_matched:,}/{len(movies):,} "
          f"({100 * n_movies_matched / max(len(movies), 1):.1f}%)")
    print(f"Netflix TV  matched   : {n_tv_matched:,}/{len(tv):,} "
          f"({100 * n_tv_matched / max(len(tv), 1):.1f}%)  "
          f"[MovieLens is movies-only: expected 0]")

    print("\nMovies matched by release-year bucket:")
    m = movies.copy()
    m["matched"] = m["show_id"].isin(matched_ids)
    m["bucket"] = pd.cut(
        m["release_year"].astype("Float64"),
        bins=[-1, 1999, 2009, 2014, 2018, 10**5],
        labels=["pre-2000", "2000-2009", "2010-2014", "2015-2018", "2019+"],
    )
    table = m.groupby("bucket", observed=False)["matched"].agg(["sum", "count"])
    for bucket, row in table.iterrows():
        cnt, tot = int(row["sum"]), int(row["count"])
        print(f"  {str(bucket):10s} {cnt:5,}/{tot:5,}  {100 * cnt / max(tot, 1):5.1f}%")

    if not accepted.empty:
        dupes = accepted[accepted.duplicated("best_movieId", keep=False)]
        if not dupes.empty:
            print(f"\nNOTE: {len(dupes)} Netflix rows share a MovieLens id "
                  f"(catalog duplicates): {sorted(dupes['best_movieId'].unique())[:10]}")

    print(f"\nReview queue           : {len(review):,} rows -> {REVIEW_CSV.name}")
    if not review.empty:
        print("  top reasons:")
        for reason, cnt in review["reason"].value_counts().head(5).items():
            print(f"    {cnt:5,}  {reason}")

    rate = 100 * n_movies_matched / max(len(movies), 1)
    print("\nGATE:")
    if rate >= 50:
        print(f"  {rate:.1f}% >= 50%  ->  CF is the headline feature (alpha ~ 0.6)")
    elif rate >= 30:
        print(f"  30% <= {rate:.1f}% < 50%  ->  balanced blend (alpha ~ 0.5)")
    else:
        print(f"  {rate:.1f}% < 30%  ->  content model leads, CF secondary (alpha ~ 0.3)")
    print("=" * 64)


def main() -> int:
    if not MOVIELENS_DIR.is_dir():
        raise SystemExit("MovieLens data missing - run: python scripts/download_movielens.py")

    # Create the (empty) overrides file on first run so it is obvious where
    # hand-curated fixes belong.
    if not OVERRIDES_CSV.exists():
        pd.DataFrame(columns=["show_id", "movieId", "note"]).to_csv(OVERRIDES_CSV, index=False)

    nf = load_netflix()
    ml = load_movielens()
    accepted, review = match(nf, ml)
    accepted = apply_overrides(accepted, ml, nf)

    MAPPING_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["show_id", "best_movieId", "source", "best_title", "netflix_title",
            "best_score", "genre_jaccard"]
    if not accepted.empty:
        accepted = accepted.reindex(columns=cols + [c for c in accepted.columns if c not in cols])
        accepted.to_csv(MAPPING_CSV, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(MAPPING_CSV, index=False)

    review.to_csv(REVIEW_CSV, index=False)

    report(accepted, review, nf)
    print(f"\nWrote {MAPPING_CSV.relative_to(REPO_ROOT)} "
          f"({0 if accepted.empty else len(accepted):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

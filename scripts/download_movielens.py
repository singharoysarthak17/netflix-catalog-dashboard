"""Download the MovieLens datasets used by the recommender.

Default (safe, redistributable):
    python scripts/download_movielens.py
    -> data/movielens/ml-latest-small/   (committed, same-license redistribution allowed)

Optional (NOT redistributable, local experiments only):
    python scripts/download_movielens.py --25m
    -> data/movielens/ml-25m/            (gitignored)

License notes (see data/movielens/ml-latest-small/README.txt):
  * ml-latest-small may be redistributed under the same license conditions,
    with the Harper & Konstan (2015) citation retained.
  * ml-25m may NOT be redistributed without separate permission from GroupLens.
    Everything produced from it (including model artifacts) is gitignored too.

Required citation:
  F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
  History and Context. ACM TiiS 5(4), 19:1-19:19. https://doi.org/10.1145/2827872
"""

from __future__ import annotations

import argparse
import io
import ssl
import sys
import zipfile
from pathlib import Path

import certifi
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "data" / "movielens"

DATASETS = {
    "small": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "zip": "ml-latest-small.zip",
        "folder": "ml-latest-small",
        "redistributable": True,
    },
    "25m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        "zip": "ml-25m.zip",
        "folder": "ml-25m",
        "redistributable": False,
    },
}

CITATION = (
    "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: "
    "History and Context. ACM Transactions on Interactive Intelligent "
    "Systems (TiiS) 5(4), 19:1-19:19. https://doi.org/10.1145/2827872"
)


def _context() -> ssl.SSLContext:
    """System CA store fails on some Windows/Anaconda setups; certifi never does."""
    return ssl.create_default_context(cafile=certifi.where())


def download(spec: dict, force: bool = False) -> Path:
    target_dir = DEST / spec["folder"]
    zip_path = DEST / spec["zip"]

    if target_dir.is_dir() and any(target_dir.iterdir()) and not force:
        print(f"[skip] {spec['folder']} already present at {target_dir}")
        return target_dir

    if not spec["redistributable"]:
        print("=" * 72)
        print("WARNING: ml-25m LICENSE FORBIDS REDISTRIBUTION WITHOUT PERMISSION.")
        print("  - Do not commit this folder or any artifact derived from it.")
        print("  - It is gitignored; use it for local experiments only.")
        print("=" * 72)

    print(f"[get ] {spec['url']}")
    with urllib.request.urlopen(spec["url"], timeout=120, context=_context()) as resp:
        payload = resp.read()
    zip_path.write_bytes(payload)
    print(f"[save] {zip_path} ({len(payload):,} bytes)")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(DEST)
    print(f"[ok  ] extracted to {target_dir}")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--25m", dest="big", action="store_true",
                        help="also download ml-25m (NOT redistributable)")
    parser.add_argument("--force", action="store_true", help="re-download")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    download(DATASETS["small"], force=args.force)
    if args.big:
        download(DATASETS["25m"], force=args.force)

    print(f"\nCitation required when publishing results:\n  {CITATION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

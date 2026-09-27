"""Shared test setup: put the repo root on sys.path and make it the CWD so
relative data paths (netflix_titles.csv, data/movielens/...) resolve."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

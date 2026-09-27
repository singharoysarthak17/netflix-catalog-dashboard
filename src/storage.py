"""Persistent watchlist storage.

Three backends, chosen automatically:

  GistStorage    secret GitHub Gist holding one JSON document keyed by
                 per-user random keys. Used on Streamlit Community Cloud,
                 where the container filesystem is wiped on every deploy.
  SqliteStorage  local file, used when running the app on your machine.
  SessionStorage st.session_state, always available as the last resort so
                 a storage outage can never break the page.

`ResilientStorage` wraps whichever primary backend is active: any failure
falls back to session storage and is reported through `last_error` instead
of raising.

Per-user keys are generated with `secrets.token_urlsafe` and shown to the
visitor as a shareable link. Anyone holding the link can read and edit that
watchlist -- that trade-off is surfaced in the UI, not hidden.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Protocol

import requests
import streamlit as st

GIST_FILENAME = "watchlists.json"
GIST_API = "https://api.github.com/gists"
CACHE_TTL = 60.0

Watchlist = dict[str, dict]


def new_key() -> str:
    return secrets.token_urlsafe(16)


class Storage(Protocol):
    name: str

    def load(self, key: str) -> Watchlist: ...
    def save(self, key: str, data: Watchlist) -> None: ...


class SessionStorage:
    """In-memory, per browser session. Never fails."""

    name = "this browser session"

    def load(self, key: str) -> Watchlist:
        return dict(st.session_state.get(f"watchlist_{key}", {}))

    def save(self, key: str, data: Watchlist) -> None:
        st.session_state[f"watchlist_{key}"] = dict(data)


class SqliteStorage:
    name = "local SQLite file"

    def __init__(self, path: Path = Path(".watchlist/watchlists.sqlite")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS watchlists ("
                " key TEXT PRIMARY KEY,"
                " payload TEXT NOT NULL,"
                " updated_at TEXT NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def load(self, key: str) -> Watchlist:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM watchlists WHERE key = ?", (key,)).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}

    def save(self, key: str, data: Watchlist) -> None:
        payload = json.dumps(data, sort_keys=True)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO watchlists (key, payload, updated_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET payload = excluded.payload, "
                "updated_at = excluded.updated_at",
                (key, payload),
            )


class GistStorage:
    """One secret Gist; per-user key selects a slice of the JSON document."""

    name = "GitHub Gist"

    def __init__(self, token: str, gist_id: str) -> None:
        self.token = token
        self.gist_id = gist_id
        self._cache: dict[str, tuple[float, dict]] = {}

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _document(self, refresh: bool = False) -> dict:
        now = time.monotonic()
        if not refresh:
            hit = self._cache.get("_doc")
            if hit and now - hit[0] < CACHE_TTL:
                return hit[1]
        resp = requests.get(
            f"{GIST_API}/{self.gist_id}", headers=self._headers(), timeout=15)
        resp.raise_for_status()
        files = resp.json().get("files", {})
        entry = files.get(GIST_FILENAME) or {}
        raw = entry.get("content", "") or ""
        try:
            doc = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            doc = {}
        if not isinstance(doc, dict):
            doc = {}
        doc.setdefault("version", 1)
        doc.setdefault("lists", {})
        self._cache["_doc"] = (now, doc)
        return doc

    def load(self, key: str) -> Watchlist:
        doc = self._document()
        entries = doc.get("lists", {}).get(key, {})
        return dict(entries) if isinstance(entries, dict) else {}

    def save(self, key: str, data: Watchlist) -> None:
        # Read-modify-write with a fresh document so two visitors (or two
        # tabs) cannot silently clobber each other's watchlist.
        doc = self._document(refresh=True)
        doc.setdefault("lists", {})[key] = dict(data)
        payload = json.dumps(doc, sort_keys=True)
        resp = requests.patch(
            f"{GIST_API}/{self.gist_id}",
            headers=self._headers(),
            json={"files": {GIST_FILENAME: {"content": payload}}},
            timeout=15,
        )
        resp.raise_for_status()
        self._cache["_doc"] = (time.monotonic(), doc)


class ResilientStorage:
    """Delegates to `primary`, falling back to session storage on failure."""

    def __init__(self, primary: Storage, fallback: SessionStorage | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or SessionStorage()
        self.name = primary.name
        self.last_error: str | None = None

    def load(self, key: str) -> Watchlist:
        try:
            data = self.primary.load(key)
            self.last_error = None
            return data
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.fallback.load(key)

    def save(self, key: str, data: Watchlist) -> None:
        try:
            self.primary.save(key, data)
            self.last_error = None
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.fallback.save(key, data)


def _secrets() -> dict:
    try:
        return dict(st.secrets)                       # type: ignore[arg-type]
    except Exception:                                  # noqa: BLE001
        return {}


def _build_primary() -> Storage:
    secrets_map = _secrets()
    token = secrets_map.get("GITHUB_TOKEN") or secrets_map.get("GH_TOKEN")
    gist_id = secrets_map.get("GIST_ID")
    if token and gist_id:
        return GistStorage(str(token), str(gist_id))
    return SqliteStorage()


_storage: ResilientStorage | None = None


def get_storage() -> ResilientStorage:
    """Built once per process; the SQLite path and secrets are fixed."""
    global _storage
    if _storage is None:
        _storage = ResilientStorage(_build_primary())
    return _storage

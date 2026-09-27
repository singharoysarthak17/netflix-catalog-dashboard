"""Watchlist storage backends."""

from __future__ import annotations

import json

import pytest

from src.storage import ResilientStorage, SqliteStorage, Watchlist, new_key


@pytest.fixture()
def store(tmp_path) -> SqliteStorage:
    return SqliteStorage(tmp_path / "w.sqlite")


def test_roundtrip(store: SqliteStorage):
    key = new_key()
    assert store.load(key) == {}

    data: Watchlist = {"s1": {"rating": 4.5, "added_at": "2026-01-01T00:00:00+00:00"}}
    store.save(key, data)
    assert store.load(key) == data


def test_keys_are_isolated(store: SqliteStorage):
    a, b = new_key(), new_key()
    store.save(a, {"s1": {"rating": 5.0}})
    store.save(b, {"s2": {"rating": 1.0}})
    assert store.load(a) != store.load(b)
    assert a != b


def test_key_is_unpredictable():
    keys = {new_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(len(k) >= 20 for k in keys)


def test_overwrite_stores_one_document_per_key(store: SqliteStorage):
    key = new_key()
    store.save(key, {"s1": {"rating": 1.0}})
    store.save(key, {"s2": {"rating": 5.0}})
    assert store.load(key) == {"s2": {"rating": 5.0}}


def test_falling_back_never_raises(tmp_path):
    """A broken primary must degrade to session storage and report it."""

    class Broken:
        name = "broken"

        def load(self, key):
            raise RuntimeError("gist is down")

        def save(self, key, data):
            raise RuntimeError("gist is down")

    class Memory:
        def __init__(self):
            self.data = {}

        def load(self, key):
            return self.data.get(key, {})

        def save(self, key, data):
            self.data[key] = data

    fallback = Memory()
    storage = ResilientStorage(Broken(), fallback)

    key = new_key()
    assert storage.load(key) == {}
    storage.save(key, {"s1": {"rating": 3.0}})

    assert fallback.load(key) == {"s1": {"rating": 3.0}}
    assert storage.last_error and "gist is down" in storage.last_error


def test_gist_payload_is_valid_json_roundtrip():
    doc = {"version": 1, "lists": {"abc": {"s1": {"rating": 5.0}}}}
    assert json.loads(json.dumps(doc)) == doc

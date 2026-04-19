"""Tests for the owner-pin bootstrap helper in src.main.

The helper seeds an empty store with (a) a "User" entity, (b) an entity
named after the profile owner, and (c) a fact containing the identity
pin — so cold-start queries on Day 1 already have grounding.

Re-running against a non-empty store must be a no-op (idempotency).
"""
from __future__ import annotations

import asyncio

import pytest

from sieve.config import ProfileOwnerConfig, StoreConfig
from sieve.main import _maybe_bootstrap_owner_pin
from sieve.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    cfg = StoreConfig(path=str(tmp_path / "bootstrap.db"), embedding_dimensions=4)
    s = MemoryStore(cfg, passphrase="test-bootstrap")
    s.open()
    s.init_schema()
    yield s
    s.close()


async def _noop_embed(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


async def _fail_embed(_text: str) -> list[float]:
    raise RuntimeError("embedder down")


def _entities(store: MemoryStore) -> list[tuple[str, str | None]]:
    rows = store._conn.execute(
        "SELECT name, type FROM entities ORDER BY name"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _fact_contents(store: MemoryStore) -> list[str]:
    rows = store._conn.execute(
        "SELECT content FROM facts ORDER BY created_at"
    ).fetchall()
    return [r[0] for r in rows]


def test_bootstrap_populates_empty_store(store):
    owner = ProfileOwnerConfig(
        name="Albert Green",
        aliases=["Albert", "I"],
        pin="Albert is a 41-year-old engineer in Bristol.",
    )
    asyncio.run(_maybe_bootstrap_owner_pin(store, owner, _noop_embed))

    names = [n for (n, _t) in _entities(store)]
    assert "User" in names
    assert "Albert Green" in names

    contents = _fact_contents(store)
    assert any("Albert Green" in c and "41-year-old" in c for c in contents), (
        f"pin not inserted as fact: {contents}"
    )


def test_bootstrap_is_idempotent(store):
    owner = ProfileOwnerConfig(
        name="Albert Green",
        pin="Albert is a 41-year-old engineer in Bristol.",
    )
    asyncio.run(_maybe_bootstrap_owner_pin(store, owner, _noop_embed))
    fact_count_after_first = len(_fact_contents(store))

    # Call again — already populated, must not duplicate.
    asyncio.run(_maybe_bootstrap_owner_pin(store, owner, _noop_embed))
    assert len(_fact_contents(store)) == fact_count_after_first


def test_bootstrap_no_pin_is_noop(store):
    owner = ProfileOwnerConfig(name="Albert Green", pin="")
    asyncio.run(_maybe_bootstrap_owner_pin(store, owner, _noop_embed))
    assert _entities(store) == []
    assert _fact_contents(store) == []


def test_bootstrap_no_owner_name_is_noop(store):
    owner = ProfileOwnerConfig(name="", pin="some pin")
    asyncio.run(_maybe_bootstrap_owner_pin(store, owner, _noop_embed))
    assert _entities(store) == []


def test_bootstrap_survives_embedder_failure(store):
    """Embedder outages must not block startup — insert without vector."""
    owner = ProfileOwnerConfig(
        name="Albert Green",
        pin="Albert is a 41-year-old engineer in Bristol.",
    )
    asyncio.run(_maybe_bootstrap_owner_pin(store, owner, _fail_embed))
    contents = _fact_contents(store)
    assert any("Albert Green" in c for c in contents)

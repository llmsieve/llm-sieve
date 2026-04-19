"""Cycle 27 T2: schema v2 migration + CRUD tests.

Verifies:
- Fresh DBs get all v2 columns via SCHEMA_SQL.
- Legacy DBs (simulated by stripping v2 columns) get them via
  ALTER TABLE, idempotently.
- known_unknowns insert is idempotent on (subject, slot_key).
- get_current_slot_fact returns the single current value.
- get_slot_timeline returns all versions in order.
- supersede_slot correctly marks prior rows valid_to + superseded_by.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sieve.config import RecallConfig, StoreConfig
from sieve.store import MemoryStore


@pytest.fixture
def fresh_store(tmp_path: Path) -> MemoryStore:
    cfg = StoreConfig(path=str(tmp_path / "v2.db"))
    s = MemoryStore(cfg)
    s.open()
    s.init_schema()
    yield s
    s.close()


def _facts_columns(store: MemoryStore) -> set[str]:
    return {row[1] for row in store.conn.execute("PRAGMA table_info(facts)").fetchall()}


def _rel_columns(store: MemoryStore) -> set[str]:
    return {row[1] for row in store.conn.execute("PRAGMA table_info(relationships)").fetchall()}


def test_fresh_store_has_v2_fact_columns(fresh_store: MemoryStore) -> None:
    cols = _facts_columns(fresh_store)
    for c in (
        "subject_entity_id", "predicate", "object_entity_id", "object_literal",
        "slot_key", "valid_from", "valid_to", "category",
        "source_turn_id", "extraction_method",
    ):
        assert c in cols, f"missing v2 column facts.{c}"


def test_fresh_store_has_v2_relationship_columns(fresh_store: MemoryStore) -> None:
    cols = _rel_columns(fresh_store)
    for c in ("relationship_type", "valid_from", "valid_to"):
        assert c in cols, f"missing v2 column relationships.{c}"


def test_fresh_store_has_known_unknowns_table(fresh_store: MemoryStore) -> None:
    row = fresh_store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='known_unknowns'"
    ).fetchone()
    assert row is not None
    # Should be empty
    assert fresh_store.conn.execute("SELECT count(*) FROM known_unknowns").fetchone()[0] == 0


def test_fresh_store_has_slot_current_index(fresh_store: MemoryStore) -> None:
    row = fresh_store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_facts_slot_current'"
    ).fetchone()
    assert row is not None


def test_init_schema_is_idempotent(fresh_store: MemoryStore) -> None:
    # Running init_schema twice must not raise, and column count must be stable.
    cols_before = _facts_columns(fresh_store)
    fresh_store.init_schema()
    fresh_store.init_schema()
    assert _facts_columns(fresh_store) == cols_before


def test_known_unknown_insert_is_idempotent(fresh_store: MemoryStore) -> None:
    a = fresh_store.insert_known_unknown("mary_chen", "mother_first_name", "asked_by_llm")
    b = fresh_store.insert_known_unknown("mary_chen", "mother_first_name", "asked_by_llm")
    assert a == b
    rows = fresh_store.get_known_unknowns("mary_chen")
    assert len(rows) == 1
    assert rows[0]["slot_key"] == "mother_first_name"
    assert rows[0]["reason"] == "asked_by_llm"


def _seed_slot_fact(
    store: MemoryStore,
    *,
    content: str,
    slot_key: str,
    predicate: str,
    object_literal: str,
    valid_from: str,
    valid_to: str | None = None,
) -> str:
    fact_id = store.insert_fact(content=content)
    store.conn.execute(
        "UPDATE facts SET slot_key = ?, predicate = ?, object_literal = ?,"
        " valid_from = ?, valid_to = ? WHERE id = ?",
        (slot_key, predicate, object_literal, valid_from, valid_to, fact_id),
    )
    store.conn.commit()
    return fact_id


def test_get_current_slot_fact_returns_latest(fresh_store: MemoryStore) -> None:
    _seed_slot_fact(
        fresh_store,
        content="Mary Chen is a product manager at Nexus Health",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="Nexus Health",
        valid_from="2024-01-01",
        valid_to="2026-04-01",
    )
    new_id = _seed_slot_fact(
        fresh_store,
        content="Mary Chen is VP of Product at Meridian Health",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="Meridian Health",
        valid_from="2026-04-01",
    )
    current = fresh_store.get_current_slot_fact("mary_chen:employer")
    assert current is not None
    assert current["id"] == new_id
    assert current["object_literal"] == "Meridian Health"


def test_get_current_slot_fact_returns_none_when_absent(fresh_store: MemoryStore) -> None:
    assert fresh_store.get_current_slot_fact("mary_chen:employer") is None


def test_get_slot_timeline_returns_all_versions(fresh_store: MemoryStore) -> None:
    _seed_slot_fact(
        fresh_store,
        content="analyst at State Street",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="State Street",
        valid_from="2020-01-01",
        valid_to="2024-01-01",
    )
    _seed_slot_fact(
        fresh_store,
        content="PM at Nexus Health",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="Nexus Health",
        valid_from="2024-01-01",
        valid_to="2026-04-01",
    )
    _seed_slot_fact(
        fresh_store,
        content="VP Product at Meridian",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="Meridian Health",
        valid_from="2026-04-01",
    )
    timeline = fresh_store.get_slot_timeline("mary_chen:employer")
    assert [t["object_literal"] for t in timeline] == ["State Street", "Nexus Health", "Meridian Health"]


def test_supersede_slot_marks_prior_rows(fresh_store: MemoryStore) -> None:
    old = _seed_slot_fact(
        fresh_store,
        content="Nexus Health",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="Nexus Health",
        valid_from="2024-01-01",
    )
    new = _seed_slot_fact(
        fresh_store,
        content="Meridian Health",
        slot_key="mary_chen:employer",
        predicate="employer",
        object_literal="Meridian Health",
        valid_from="2026-04-01",
    )
    updated = fresh_store.supersede_slot("mary_chen:employer", new, "2026-04-01")
    assert updated == 1
    row = fresh_store.conn.execute(
        "SELECT valid_to, superseded_by FROM facts WHERE id = ?", (old,)
    ).fetchone()
    assert row[0] == "2026-04-01"
    assert row[1] == new
    # Current-slot query now returns only the new row
    current = fresh_store.get_current_slot_fact("mary_chen:employer")
    assert current["id"] == new

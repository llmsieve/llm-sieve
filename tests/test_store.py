"""Tests for Phase 2: Encrypted memory store with vector search."""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import pytest
import sqlcipher3

from sieve.config import StoreConfig
from sieve.store import MemoryStore, serialize_float32, deserialize_float32


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "test_memory.db"


@pytest.fixture
def store_config(tmp_db) -> StoreConfig:
    return StoreConfig(path=str(tmp_db), embedding_dimensions=4)


@pytest.fixture
def store(store_config) -> MemoryStore:
    ms = MemoryStore(store_config, passphrase="test-passphrase-123")
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


# --- Serialization ---


def test_serialize_deserialize_float32():
    vec = [1.0, 0.5, -0.3, 0.0]
    blob = serialize_float32(vec)
    assert len(blob) == 4 * 4  # 4 floats * 4 bytes
    result = deserialize_float32(blob, 4)
    for a, b in zip(vec, result):
        assert abs(a - b) < 1e-6


# --- Schema initialization ---


def test_schema_creates_all_tables(store):
    tables = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {t[0] for t in tables}
    expected = {
        "facts", "entities", "relationships", "episodes",
        "preferences", "sessions", "fingerprints", "audit_log",
    }
    assert expected.issubset(table_names)


def test_vec_facts_virtual_table_exists(store):
    # vec0 tables show up differently in sqlite_master
    row = store.conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE name='vec_facts'"
    ).fetchone()
    assert row[0] > 0


def test_is_initialized(store):
    assert store.is_initialized() is True


def test_is_initialized_false_on_empty_db(tmp_path):
    config = StoreConfig(path=str(tmp_path / "empty.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test")
    ms.open()
    assert ms.is_initialized() is False
    ms.close()


# --- Encryption verification ---


def test_db_encrypted_unreadable_without_passphrase(store, tmp_db):
    """The database file should be unreadable with a standard sqlite3 connection."""
    store.insert_fact("User lives in Springfield", embedding=[1.0, 0.0, 0.0, 0.0])
    store.close()

    # Try reading with wrong passphrase
    conn = sqlcipher3.connect(str(tmp_db))
    conn.execute("PRAGMA key='wrong-passphrase'")
    with pytest.raises(sqlcipher3.dbapi2.DatabaseError):
        conn.execute("SELECT count(*) FROM sqlite_master")
    conn.close()


def test_db_readable_with_correct_passphrase(store, tmp_db):
    """The database should be readable with the correct passphrase."""
    store.insert_fact("User lives in Springfield", embedding=[1.0, 0.0, 0.0, 0.0])
    store.close()

    conn = sqlcipher3.connect(str(tmp_db))
    conn.execute("PRAGMA key='test-passphrase-123'")
    count = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    assert count == 1
    conn.close()


def test_db_file_not_plaintext(store, tmp_db):
    """Raw file bytes should not contain plaintext content."""
    store.insert_fact("User lives in Springfield", embedding=[1.0, 0.0, 0.0, 0.0])
    store.close()

    raw = tmp_db.read_bytes()
    assert b"User lives in Springfield" not in raw
    assert b"SQLite" not in raw  # encrypted header


# --- Passphrase management ---


def test_passphrase_auto_generated(tmp_path):
    """First open should auto-generate a keyfile."""
    db_path = tmp_path / "auto.db"
    config = StoreConfig(path=str(db_path), embedding_dimensions=4)
    ms = MemoryStore(config)
    ms.open()
    ms.init_schema()

    keyfile = tmp_path / ".sieve_key"
    assert keyfile.exists()
    passphrase = keyfile.read_text().strip()
    assert len(passphrase) == 64  # 32 bytes hex
    # Verify permissions
    assert oct(keyfile.stat().st_mode)[-3:] == "600"
    ms.close()

    # Reopen with auto-detected passphrase
    ms2 = MemoryStore(config)
    ms2.open()
    assert ms2.is_initialized()
    ms2.close()


# --- Facts CRUD ---


def test_insert_and_get_fact(store):
    fact_id = store.insert_fact(
        "User lives in Springfield",
        embedding=[1.0, 0.0, 0.0, 0.0],
        entity_ids=["entity1"],
        source="conversation",
        confidence=0.9,
        fact_type="objective",
    )

    fact = store.get_fact(fact_id)
    assert fact is not None
    assert fact["content"] == "User lives in Springfield"
    assert fact["confidence"] == 0.9
    assert fact["fact_type"] == "objective"
    assert fact["status"] == "current"
    assert fact["status_detail"] == "provisional"


def test_get_facts_by_status(store):
    store.insert_fact("Fact 1", embedding=[1.0, 0.0, 0.0, 0.0])
    store.insert_fact("Fact 2", embedding=[0.0, 1.0, 0.0, 0.0])

    facts = store.get_facts(status="current")
    assert len(facts) == 2


def test_update_fact_retrieval(store):
    fact_id = store.insert_fact("Test fact", embedding=[1.0, 0.0, 0.0, 0.0])
    store.update_fact_retrieval(fact_id)
    store.update_fact_retrieval(fact_id)

    fact = store.get_fact(fact_id)
    assert fact["retrieval_count"] == 2
    assert fact["last_retrieved_at"] is not None


# --- Vector similarity search ---


def _normalize(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length for cosine-like distance."""
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0 else vec


def test_vector_search_returns_closest(store):
    """Insert several facts with known vectors, query with a similar vector."""
    store.insert_fact("User lives in Springfield", embedding=_normalize([1.0, 0.0, 0.0, 0.0]))
    store.insert_fact("User works at Acme", embedding=_normalize([0.0, 1.0, 0.0, 0.0]))
    store.insert_fact("User likes coffee", embedding=_normalize([0.0, 0.0, 1.0, 0.0]))

    # Query close to "Springfield" vector
    results = store.search_facts_by_vector(
        query_embedding=_normalize([0.9, 0.1, 0.0, 0.0]),
        limit=2,
    )
    assert len(results) == 2
    assert results[0]["content"] == "User lives in Springfield"
    assert "distance" in results[0]
    assert results[0]["distance"] < results[1]["distance"]


def test_vector_search_with_min_distance(store):
    """min_distance should filter out distant results."""
    store.insert_fact("Close fact", embedding=_normalize([1.0, 0.0, 0.0, 0.0]))
    store.insert_fact("Far fact", embedding=_normalize([0.0, 0.0, 0.0, 1.0]))

    results = store.search_facts_by_vector(
        query_embedding=_normalize([1.0, 0.0, 0.0, 0.0]),
        limit=10,
        min_distance=0.5,
    )
    assert len(results) == 1
    assert results[0]["content"] == "Close fact"


def test_vector_search_empty_store(store):
    results = store.search_facts_by_vector(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        limit=5,
    )
    assert results == []


# --- Entities CRUD ---


def test_insert_and_get_entity(store):
    entity_id = store.insert_entity("Springfield", type="place", description="City in UAE")
    entity = store.get_entity(entity_id)
    assert entity is not None
    assert entity["name"] == "Springfield"
    assert entity["type"] == "place"


def test_find_entity_by_name(store):
    store.insert_entity("Springfield", type="place")
    entity = store.find_entity_by_name("Springfield")
    assert entity is not None
    assert entity["name"] == "Springfield"


def test_find_entity_by_name_not_found(store):
    assert store.find_entity_by_name("Nonexistent") is None


# --- Relationships & graph traversal ---


def test_relationships_and_graph_traversal(store):
    user_id = store.insert_entity("User", type="person")
    springfield_id = store.insert_entity("Springfield", type="place")
    acme_id = store.insert_entity("Acme Corp", type="project")

    store.insert_relationship(user_id, "lives_in", springfield_id)
    store.insert_relationship(user_id, "works_at", acme_id)

    related = store.get_related_entities(user_id)
    assert len(related) == 2
    names = {e["name"] for e in related}
    assert names == {"Springfield", "Acme Corp"}


def test_graph_traversal_bidirectional(store):
    """Traversal should work from either direction."""
    user_id = store.insert_entity("User", type="person")
    springfield_id = store.insert_entity("Springfield", type="place")
    store.insert_relationship(user_id, "lives_in", springfield_id)

    # From Springfield side, should find User
    related = store.get_related_entities(springfield_id)
    assert len(related) == 1
    assert related[0]["name"] == "User"


# --- Episodes ---


def test_insert_episode(store):
    ep_id = store.insert_episode(
        "User asked about weather in Springfield",
        entities_involved=["entity1"],
        decisions_made=["provided forecast"],
        session_id="session1",
    )
    assert ep_id is not None


# --- Preferences ---


def test_insert_preference(store):
    pref_id = store.insert_preference("communication", "User prefers concise responses", 0.8)
    assert pref_id is not None


# --- Sessions ---


def test_session_lifecycle(store):
    session_id = store.insert_session()
    store.end_session(session_id, coherence_score=0.95)

    row = store.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row is not None
    # ended_at should be set
    assert row[4] is not None  # ended_at column


# --- Stats ---


def test_stats(store):
    store.insert_fact("Fact 1", embedding=[1.0, 0.0, 0.0, 0.0])
    store.insert_entity("Entity 1", type="person")
    store.insert_session()

    s = store.stats()
    assert s["facts_count"] == 1
    assert s["entities_count"] == 1
    assert s["sessions_count"] == 1
    assert s["vec_facts_count"] == 1
    assert "db_size_bytes" in s


# --- Audit log ---


def test_audit_log_written_on_insert(store):
    store.insert_fact("Test", embedding=[1.0, 0.0, 0.0, 0.0])
    rows = store.conn.execute("SELECT * FROM audit_log").fetchall()
    assert len(rows) >= 1
    # Should have an 'extract' operation for the fact
    ops = [r[1] for r in rows]
    assert "extract" in ops


# --- Tool Registry ---


def test_tool_registry_table_exists(store):
    """The tool_registry table must be created by init_schema."""
    # Verify the table exists and has all expected columns
    cols = {row[1] for row in store.conn.execute(
        "PRAGMA table_info(tool_registry)"
    ).fetchall()}
    expected = {
        "id", "name", "description", "full_schema", "lean_schema",
        "embedding", "category", "usage_count", "last_used_at",
        "hash", "active", "created_at", "updated_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"
    # Verify the indexes exist
    idx_names = {row[1] for row in store.conn.execute(
        "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='tool_registry'"
    ).fetchall()}
    assert "idx_tool_registry_name" in idx_names
    assert "idx_tool_registry_category" in idx_names
    assert "idx_tool_registry_hash" in idx_names
    assert "idx_tool_registry_active" in idx_names


# --- Embedding dimension mismatch guard ---


def test_dimension_mismatch_raises_hard_error(tmp_path):
    """Switching embedding providers between runs changes the vector
    dimension (e.g. 768 Ollama → 384 FastEmbed). Every existing vector
    in the store is then incompatible with the active embedder, so
    retrieval would silently return garbage. Ensure we fail cold with
    a clear error instead."""
    from sieve.store import EmbeddingDimensionMismatchError

    db_path = tmp_path / "mixed_dim.db"

    cfg_a = StoreConfig(path=str(db_path), embedding_dimensions=8)
    store_a = MemoryStore(cfg_a, passphrase="pw")
    store_a.open()
    store_a.init_schema()
    store_a.insert_fact(
        content="seed fact",
        embedding=[0.1] * 8,
    )
    store_a.close()

    # Re-open with a different configured dimension (simulating a
    # provider switch). The check must raise.
    cfg_b = StoreConfig(path=str(db_path), embedding_dimensions=4)
    store_b = MemoryStore(cfg_b, passphrase="pw")
    store_b.open()
    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        store_b.check_embedding_dimensions()
    assert excinfo.value.stored_dim == 8
    assert excinfo.value.configured_dim == 4
    store_b.close()


def test_dimension_check_noop_on_empty_store(store):
    """Empty store must not raise regardless of configured dimension."""
    store.check_embedding_dimensions()  # must not raise

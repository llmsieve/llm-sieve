"""Tests for src/tool_registry.py — ToolRegistry CRUD + ingest."""
import asyncio
import json

import pytest

from sieve.config import StoreConfig
from sieve.store import MemoryStore, deserialize_float32
from sieve.tool_registry import ToolRegistry, ToolRecord


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current weather and news.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

FS_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Remember a note for later recall.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}


def _fake_embed_factory(dim: int = 768):
    import hashlib
    import struct
    async def _embed(text: str) -> list[float]:
        # Deterministic, collision-resistant: SHA-256 of the text, expanded
        # into `dim` float values. Any change to the text produces a
        # completely different vector.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the digest bytes enough times to fill `dim` floats
        needed = dim * 4  # 4 bytes per float32
        buf = (digest * ((needed // len(digest)) + 1))[:needed]
        # Interpret each 4-byte chunk as an unsigned int, normalise to [0, 1]
        floats = [
            struct.unpack("<I", buf[i:i+4])[0] / 2**32
            for i in range(0, needed, 4)
        ]
        return floats
    return _embed


@pytest.fixture
def store(tmp_path):
    cfg = StoreConfig(path=str(tmp_path / "memory.db"), embedding_dimensions=768)
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


@pytest.fixture
def registry(store):
    return ToolRegistry(store, embed_fn=_fake_embed_factory(), compression="moderate")


def test_ingest_new_tools_creates_rows(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL, MEMORY_TOOL]))
    rows = store.conn.execute("SELECT name FROM tool_registry ORDER BY name").fetchall()
    names = [r[0] for r in rows]
    assert names == ["read_file", "save_note", "web_search"]


def test_ingest_stores_embeddings(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    row = store.conn.execute(
        "SELECT embedding FROM tool_registry WHERE name='web_search'"
    ).fetchone()
    assert row[0] is not None
    vec = deserialize_float32(row[0], 768)
    assert len(vec) == 768


def test_ingest_stores_lean_schema(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    row = store.conn.execute(
        "SELECT full_schema, lean_schema FROM tool_registry WHERE name='web_search'"
    ).fetchone()
    full = json.loads(row[0])
    lean = json.loads(row[1])
    assert full == WEATHER_TOOL  # stored verbatim
    # Moderate compression: description preserved (already one sentence)
    assert lean["function"]["description"] == "Search the web for current weather and news."
    # And the property has been trimmed — no "description" key on query (it had none anyway)
    assert lean["function"]["parameters"]["properties"]["query"] == {"type": "string"}


def test_ingest_categorises_tools(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL, MEMORY_TOOL]))
    cats = dict(store.conn.execute(
        "SELECT name, category FROM tool_registry"
    ).fetchall())
    assert cats["web_search"] == "web"
    assert cats["read_file"] == "filesystem"
    assert cats["save_note"] == "memory"


def test_ingest_assigns_other_for_unknown_tools(registry, store):
    unknown_tool = {
        "type": "function",
        "function": {
            "name": "image_generate",
            "description": "Create an image from a prompt.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        },
    }
    asyncio.run(registry.ingest([unknown_tool]))
    cat = store.conn.execute(
        "SELECT category FROM tool_registry WHERE name='image_generate'"
    ).fetchone()[0]
    assert cat == "other"


def test_ingest_skips_unchanged_hash(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    row1 = store.conn.execute(
        "SELECT updated_at FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0]
    # Ingest again with identical tool — should be a no-op (no updated_at change)
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    row2 = store.conn.execute(
        "SELECT updated_at FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0]
    assert row1 == row2


def test_ingest_reembeds_on_description_change(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    orig = store.conn.execute(
        "SELECT embedding FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0]

    changed = json.loads(json.dumps(WEATHER_TOOL))  # deep copy
    changed["function"]["description"] = "Look up things on the internet."
    asyncio.run(registry.ingest([changed]))

    new = store.conn.execute(
        "SELECT embedding FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0]
    assert new != orig


def test_ingest_drops_recall_collision(registry, store, caplog):
    colliding = {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Agent-defined recall tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    with caplog.at_level("WARNING"):
        asyncio.run(registry.ingest([colliding, WEATHER_TOOL]))

    rows = store.conn.execute("SELECT name FROM tool_registry").fetchall()
    assert [r[0] for r in rows] == ["web_search"]
    assert any("recall" in rec.message.lower() for rec in caplog.records)


def test_ingest_marks_removed_tools_inactive(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL]))
    # Second ingest drops FS_TOOL
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    rows = dict(store.conn.execute(
        "SELECT name, active FROM tool_registry"
    ).fetchall())
    assert rows["web_search"] == 1
    assert rows["read_file"] == 0


def test_ingest_reactivates_tool_on_readd(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL]))
    asyncio.run(registry.ingest([WEATHER_TOOL]))  # FS removed
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL]))  # FS back
    active = store.conn.execute(
        "SELECT active FROM tool_registry WHERE name='read_file'"
    ).fetchone()[0]
    assert active == 1


def test_record_usage_increments_count(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    before = store.conn.execute(
        "SELECT usage_count FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0]
    registry.record_usage("web_search")
    registry.record_usage("web_search")
    after = store.conn.execute(
        "SELECT usage_count, last_used_at FROM tool_registry WHERE name='web_search'"
    ).fetchone()
    assert after[0] == before + 2
    assert after[1] is not None


def test_record_usage_unknown_tool_is_noop(registry, store):
    # Should not raise
    registry.record_usage("never_registered")


def test_get_active_records_returns_only_active(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL]))
    asyncio.run(registry.ingest([WEATHER_TOOL]))  # mark FS inactive
    active = registry.get_active_records()
    names = [r.name for r in active]
    assert names == ["web_search"]


def test_recompute_all_lean_schemas_applies_new_mode(registry, store):
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    orig_lean = json.loads(store.conn.execute(
        "SELECT lean_schema FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0])
    # Swap compression mode
    registry._compression = "aggressive"
    registry.recompute_all_lean_schemas()
    new_lean = json.loads(store.conn.execute(
        "SELECT lean_schema FROM tool_registry WHERE name='web_search'"
    ).fetchone()[0])
    # Moderate kept description; aggressive drops it
    assert "description" in orig_lean["function"]
    assert "description" not in new_lean["function"]

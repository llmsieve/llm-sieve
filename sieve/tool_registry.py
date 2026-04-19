"""Tool registry — stores agent tools, their embeddings, and usage stats.

Hash-based ingest: unchanged tools are skipped. Changed tools are re-embedded
and re-compressed. Removed tools are marked inactive but retained so usage
history survives a re-add.

Categorisation uses a static keyword map (see CATEGORY_KEYWORDS below). Tools
that don't match any keyword are tagged "other" and fall through to L1
similarity + fallback selection in ToolClassifier.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import xxhash

from sieve.store import MemoryStore, serialize_float32, deserialize_float32
from sieve.tool_compression import compress_schema

logger = logging.getLogger("recall.tool_registry")

EmbedFn = Callable[[str], Awaitable[list[float]]]


# ─── Category heuristic ───────────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "web":        ["weather", "search", "news", "look up", "google",
                   "find online", "fetch url", "http"],
    "filesystem": ["file", "read file", "open file", "save to",
                   "directory", "folder", "write to"],
    "memory":     ["remember", "save note", "store this", "don't forget"],
    "code":       ["execute", "script", "python", "bash", "compile"],
    "comms":      ["email", "slack", "notify", "send message"],
}

OTHER_CATEGORY = "other"


def _categorize(name: str, description: str) -> str:
    """Map a tool to a category based on name + description keywords."""
    haystack = f"{name} {description}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in haystack:
                return category
    return OTHER_CATEGORY


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_name(tool: dict) -> str:
    """Extract name from either OpenAI-shape or Ollama flat shape."""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("name", "")
    return tool.get("name", "")


def _tool_description(tool: dict) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("description", "")
    return tool.get("description", "")


def _hash_tool(tool: dict) -> str:
    """Stable hash of a tool schema, order-insensitive."""
    return xxhash.xxh64(json.dumps(tool, sort_keys=True)).hexdigest()


# ─── Data record ──────────────────────────────────────────────────────────────

@dataclass
class ToolRecord:
    id: str
    name: str
    description: str
    full_schema: dict
    lean_schema: dict
    category: str
    embedding: list[float] | None
    usage_count: int
    last_used_at: str | None
    hash: str
    active: bool


# ─── Registry ────────────────────────────────────────────────────────────────

class ToolRegistry:
    """CRUD + ingest for tool schemas."""

    def __init__(
        self,
        store: MemoryStore,
        embed_fn: EmbedFn,
        compression: str = "moderate",
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._compression = compression

    @property
    def compression(self) -> str:
        return self._compression

    async def ingest(self, tools: list[dict]) -> None:
        """Upsert tools by name. Drops 'recall' collisions. Marks removed tools inactive."""
        if self._store._conn is None:
            logger.debug("ingest called with closed store — skipping")
            return

        # Drop recall collisions first
        filtered: list[dict] = []
        for t in tools:
            if not isinstance(t, dict):
                logger.warning("Skipping non-dict tool entry: %r", t)
                continue
            if _tool_name(t) == "recall":
                logger.warning(
                    "Agent tool named 'recall' collides with Recall's own tool — dropping it"
                )
                continue
            filtered.append(t)

        seen_names: set[str] = set()
        for tool in filtered:
            name = _tool_name(tool)
            if not name:
                logger.warning("Tool with no name — skipping: %r", tool)
                continue
            seen_names.add(name)
            await self._upsert_one(tool, name)

        # Mark tools not in this ingest as inactive (preserves their usage_count)
        self._deactivate_missing(seen_names)

    async def _upsert_one(self, tool: dict, name: str) -> None:
        h = _hash_tool(tool)
        existing = self._store.conn.execute(
            "SELECT id, hash, active FROM tool_registry WHERE name = ?", (name,)
        ).fetchone()

        now = _now_iso()
        if existing is not None:
            existing_id, existing_hash, existing_active = existing
            if existing_hash == h and existing_active == 1:
                # Unchanged and already active — nothing to do
                return
            if existing_hash == h and existing_active == 0:
                # Hash unchanged but tool was previously inactive — reactivate
                self._store.conn.execute(
                    "UPDATE tool_registry SET active = 1, updated_at = ? WHERE id = ?",
                    (now, existing_id),
                )
                self._store.conn.commit()
                return
            # Hash changed: re-embed, re-compress, update
            row_id = existing_id
        else:
            row_id = uuid.uuid4().hex

        description = _tool_description(tool)
        category = _categorize(name, description)

        # Embed the description (fail gracefully — store NULL on error)
        embedding_blob: bytes | None = None
        if description.strip():
            try:
                vec = await self._embed_fn(description)
                embedding_blob = serialize_float32(vec)
            except Exception as exc:
                logger.warning("embed failed for tool %r: %s", name, exc)

        lean = compress_schema(tool, mode=self._compression)

        self._store.conn.execute(
            """INSERT INTO tool_registry
                 (id, name, description, full_schema, lean_schema, embedding,
                  category, usage_count, last_used_at, hash, active,
                  created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, 1, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                 description=excluded.description,
                 full_schema=excluded.full_schema,
                 lean_schema=excluded.lean_schema,
                 embedding=excluded.embedding,
                 category=excluded.category,
                 hash=excluded.hash,
                 active=1,
                 updated_at=excluded.updated_at""",
            (
                row_id, name, description,
                json.dumps(tool, sort_keys=True),
                json.dumps(lean, sort_keys=True),
                embedding_blob,
                category,
                h,
                now, now,
            ),
        )
        self._store.conn.commit()

    def _deactivate_missing(self, seen_names: set[str]) -> None:
        """Set active=0 on rows whose name is NOT in seen_names."""
        if not seen_names:
            # If no tools in this payload, do nothing (don't purge existing)
            return
        placeholders = ",".join(["?"] * len(seen_names))
        self._store.conn.execute(
            f"UPDATE tool_registry SET active = 0 WHERE name NOT IN ({placeholders})",  # noqa: S608
            tuple(seen_names),
        )
        self._store.conn.commit()

    def record_usage(self, tool_name: str) -> None:
        """Increment usage_count + set last_used_at for a tool. Best-effort."""
        if self._store._conn is None:
            return
        try:
            self._store.conn.execute(
                """UPDATE tool_registry
                   SET usage_count = usage_count + 1,
                       last_used_at = ?
                   WHERE name = ?""",
                (_now_iso(), tool_name),
            )
            self._store.conn.commit()
        except Exception as exc:
            logger.debug("record_usage(%r) failed: %s", tool_name, exc)

    def get_active_records(self) -> list[ToolRecord]:
        """Load all active tools into memory as ToolRecord objects."""
        if self._store._conn is None:
            return []
        rows = self._store.conn.execute(
            """SELECT id, name, description, full_schema, lean_schema, embedding,
                      category, usage_count, last_used_at, hash, active
               FROM tool_registry
               WHERE active = 1
               ORDER BY name"""
        ).fetchall()
        records: list[ToolRecord] = []
        for row in rows:
            embedding: list[float] | None = None
            if row[5] is not None:
                dim = self._store.config.embedding_dimensions
                try:
                    embedding = deserialize_float32(row[5], dim)
                except Exception:
                    embedding = None
            records.append(ToolRecord(
                id=row[0],
                name=row[1],
                description=row[2] or "",
                full_schema=json.loads(row[3]),
                lean_schema=json.loads(row[4]) if row[4] else {},
                category=row[6] or OTHER_CATEGORY,
                embedding=embedding,
                usage_count=row[7] or 0,
                last_used_at=row[8],
                hash=row[9],
                active=bool(row[10]),
            ))
        return records

    def recompute_all_lean_schemas(self) -> None:
        """Re-compute lean_schema for every tool using the current compression mode."""
        if self._store._conn is None:
            return
        rows = self._store.conn.execute(
            "SELECT id, full_schema FROM tool_registry"
        ).fetchall()
        now = _now_iso()
        for row_id, full_json in rows:
            try:
                full = json.loads(full_json)
                lean = compress_schema(full, mode=self._compression)
            except Exception as exc:
                logger.warning("recompute lean_schema failed for id=%s: %s", row_id, exc)
                continue
            self._store.conn.execute(
                "UPDATE tool_registry SET lean_schema = ?, updated_at = ? WHERE id = ?",
                (json.dumps(lean, sort_keys=True), now, row_id),
            )
        self._store.conn.commit()
        logger.info("recomputed lean_schemas for %d tools (mode=%s)",
                    len(rows), self._compression)

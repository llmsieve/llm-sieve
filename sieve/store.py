"""Encrypted memory store using SQLCipher + sqlite-vec.

Single encrypted SQLite file holding facts, entities, relationships,
episodes, preferences, sessions, fingerprints, and audit log.
Vector similarity search via sqlite-vec extension.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sqlcipher3
import sqlite_vec

from sieve.config import StoreConfig

logger = logging.getLogger("recall.store")


class EmbeddingDimensionMismatchError(RuntimeError):
    """Raised when the store's on-disk embeddings don't match the active
    embedding backend's dimension. A switch between FastEmbed (384) and
    a 768-dim Ollama model, for example, leaves every stored vector
    incompatible with new queries — cosine similarity across different
    dimensions is nonsense. The operator must either sterilise the store
    or revert the provider change."""

    def __init__(self, stored_dim: int, configured_dim: int, store_path: str):
        self.stored_dim = stored_dim
        self.configured_dim = configured_dim
        self.store_path = store_path
        super().__init__(
            f"Embedding dimension mismatch at {store_path}: "
            f"store was built with {stored_dim}-dim vectors but the active "
            f"embedding provider produces {configured_dim}-dim vectors. "
            f"Options: (1) sterilise the store and reseed with the new "
            f"provider, or (2) set embeddings.provider back to the previous "
            f"backend. Continuing would silently corrupt retrieval."
        )


# --- Schema ---

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,
    entity_ids TEXT,
    source TEXT,
    confidence REAL DEFAULT 0.7,
    fact_type TEXT DEFAULT 'objective',
    status TEXT DEFAULT 'current',
    status_detail TEXT DEFAULT 'provisional',
    resolution_status TEXT DEFAULT 'resolved',
    ambiguity_note TEXT,
    superseded_by TEXT,
    session_coherence_score REAL,
    retrieval_count INTEGER DEFAULT 0,
    usage_count INTEGER DEFAULT 0,
    last_retrieved_at TEXT,
    created_at TEXT NOT NULL,
    last_confirmed_at TEXT,
    -- Schema v2 fields (populated only when ablation.schema_v2 is on).
    -- NULL on legacy rows; slot_key index is partial and ignores NULLs.
    subject_entity_id TEXT,
    predicate TEXT,
    object_entity_id TEXT,
    object_literal TEXT,
    slot_key TEXT,
    valid_from TEXT,
    valid_to TEXT,
    category TEXT,
    source_turn_id TEXT,
    extraction_method TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,
    description TEXT,
    embedding BLOB,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_entity TEXT NOT NULL REFERENCES entities(id),
    relationship TEXT NOT NULL,
    target_entity TEXT NOT NULL REFERENCES entities(id),
    confidence REAL DEFAULT 0.7,
    status TEXT DEFAULT 'current',
    created_at TEXT NOT NULL,
    -- Schema v2 fields
    relationship_type TEXT,
    valid_from TEXT,
    valid_to TEXT
);

CREATE TABLE IF NOT EXISTS known_unknowns (
    -- Explicit absence signals. When retrieval looks up a slot and
    -- finds nothing, or the profile_owner config declares a gap, a row
    -- lands here so the context formatter can emit [NOT PRESENT: X].
    id TEXT PRIMARY KEY,
    subject_entity_id TEXT NOT NULL,
    slot_key TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(subject_entity_id, slot_key)
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    embedding BLOB,
    entities_involved TEXT,
    decisions_made TEXT,
    session_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preferences (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    strength REAL DEFAULT 0.5,
    observation_count INTEGER DEFAULT 1,
    last_observed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    coherence_score REAL,
    message_count INTEGER DEFAULT 0,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS fingerprints (
    section_key TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    content TEXT,
    token_count INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    session_id TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_registry (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    full_schema TEXT NOT NULL,
    lean_schema TEXT,
    embedding BLOB,
    category TEXT,
    usage_count INTEGER DEFAULT 0,
    last_used_at TEXT,
    hash TEXT NOT NULL,
    active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_fact_type ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_created_at ON facts(created_at);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_operation ON audit_log(operation);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_registry_name ON tool_registry(name);
CREATE INDEX IF NOT EXISTS idx_tool_registry_category ON tool_registry(category);
CREATE INDEX IF NOT EXISTS idx_tool_registry_hash ON tool_registry(hash);
CREATE INDEX IF NOT EXISTS idx_tool_registry_active ON tool_registry(active);
CREATE INDEX IF NOT EXISTS idx_known_unknowns_subject ON known_unknowns(subject_entity_id);
"""

# Schema v2 indexes run AFTER the ALTER TABLE migration in
# init_schema, because on legacy DBs the referenced columns don't
# exist until the ALTERs complete.
_V2_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_facts_slot_current
    ON facts(slot_key)
    WHERE slot_key IS NOT NULL
      AND valid_to IS NULL
      AND status IN ('current', 'provisional');
CREATE INDEX IF NOT EXISTS idx_facts_valid_from ON facts(valid_from);
"""

# Schema v2: ALTER TABLE columns for existing DBs that were created
# before v2 landed. Each entry is (table, column, type). Applied
# idempotently in init_schema() — errors "duplicate column" are
# swallowed.
_V2_ALTERS: list[tuple[str, str, str]] = [
    ("facts", "subject_entity_id", "TEXT"),
    ("facts", "predicate", "TEXT"),
    ("facts", "object_entity_id", "TEXT"),
    ("facts", "object_literal", "TEXT"),
    ("facts", "slot_key", "TEXT"),
    ("facts", "valid_from", "TEXT"),
    ("facts", "valid_to", "TEXT"),
    ("facts", "category", "TEXT"),
    ("facts", "source_turn_id", "TEXT"),
    ("facts", "extraction_method", "TEXT"),
    ("relationships", "relationship_type", "TEXT"),
    ("relationships", "valid_from", "TEXT"),
    ("relationships", "valid_to", "TEXT"),
]


def _now_iso() -> str:
    from sieve.clock import get_clock
    return get_clock().now().isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def serialize_float32(vec: list[float]) -> bytes:
    """Serialize a list of floats to little-endian float32 bytes for sqlite-vec."""
    return struct.pack(f"<{len(vec)}f", *vec)


def deserialize_float32(data: bytes, dimensions: int) -> list[float]:
    """Deserialize little-endian float32 bytes back to a list of floats."""
    return list(struct.unpack(f"<{dimensions}f", data))


# --- Passphrase management ---

def _passphrase_path(db_path: Path) -> Path:
    """Path to the keyfile stored alongside the database."""
    return db_path.parent / ".sieve_key"


def get_or_create_passphrase(db_path: Path) -> str:
    """Get existing passphrase or generate one on first run.

    Stores a random 64-char hex passphrase in a restricted keyfile
    next to the database.
    """
    keyfile = _passphrase_path(db_path)
    if keyfile.exists():
        return keyfile.read_text().strip()

    passphrase = os.urandom(32).hex()
    keyfile.parent.mkdir(parents=True, exist_ok=True)
    keyfile.write_text(passphrase)
    keyfile.chmod(0o600)
    return passphrase


# --- MemoryStore ---

class MemoryStore:
    """Encrypted SQLite store with vector search support."""

    def __init__(self, config: StoreConfig, passphrase: str | None = None):
        self.config = config
        self.db_path = Path(config.path).expanduser()
        self._passphrase = passphrase
        self._conn: sqlcipher3.Connection | None = None

    @property
    def conn(self) -> sqlcipher3.Connection:
        assert self._conn is not None, "Store not opened — call open() first"
        return self._conn

    def open(self) -> None:
        """Open the encrypted database and load extensions."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self._passphrase is None:
            self._passphrase = get_or_create_passphrase(self.db_path)

        self._conn = sqlcipher3.connect(str(self.db_path))
        self._conn.execute(f"PRAGMA key='{self._passphrase}'")

        # Verify encryption is working — this will fail if key is wrong
        self._conn.execute("SELECT count(*) FROM sqlite_master")

        # Enable WAL mode for better concurrent read performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Load sqlite-vec extension
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        """Create all tables, indexes, and the vector virtual table.

        Also runs schema v2 ALTER TABLE migrations idempotently —
        existing DBs created before v2 get the new columns; fresh DBs
        already have them from SCHEMA_SQL and the ALTERs no-op
        (duplicate-column errors are swallowed).
        """
        self.conn.executescript(SCHEMA_SQL)

        # Idempotent v2 column adds for legacy DBs.
        for table, column, type_ in _V2_ALTERS:
            try:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_}")
            except Exception as exc:
                msg = str(exc).lower()
                if "duplicate column" in msg or "already exists" in msg:
                    continue
                logger.warning("v2 migration ALTER %s.%s failed: %s", table, column, exc)

        # v2 indexes (must run after ALTERs so the columns exist)
        self.conn.executescript(_V2_INDEXES_SQL)

        # Create the vec0 virtual table for fact embeddings
        dim = self.config.embedding_dimensions
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts "
            f"USING vec0(fact_id TEXT PRIMARY KEY, embedding float[{dim}])"
        )
        # Phase-3 Fix 2: vec index over episodes so follow-up queries
        # ("going back to the mortgage...") can retrieve the compressed
        # summary of the relevant prior turn.
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes "
            f"USING vec0(episode_id TEXT PRIMARY KEY, embedding float[{dim}])"
        )
        self.conn.commit()

    def is_initialized(self) -> bool:
        """Check if the schema has been created."""
        try:
            row = self.conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='facts'"
            ).fetchone()
            return row[0] > 0
        except Exception:
            return False

    def check_embedding_dimensions(self) -> None:
        """Verify stored embeddings match the configured dimension.

        Raises ``EmbeddingDimensionMismatchError`` when the store on disk
        was built with a different embedding dimension than the currently
        configured backend. A mismatch means every vector in the store is
        incompatible with the active embedder — cosine similarity is
        meaningless across dimensions — and retrieval will silently
        return garbage. The safe action is to stop cold so the operator
        either sterilises the store or switches the provider back.

        Checks both stored-vector bytes and the vec_facts DDL. The DDL
        check catches fresh stores that were initialised at the wrong
        dim (e.g. wizard built the schema before the FastEmbed override
        was applied) — those stores are empty, so the row-based check
        passes, but every subsequent write fails with a dimension
        mismatch against the vec0 virtual table.
        """
        configured_dim = self.config.embedding_dimensions

        # Row-based check — catches drift after data was written.
        try:
            row = self.conn.execute(
                "SELECT embedding FROM facts WHERE embedding IS NOT NULL LIMIT 1"
            ).fetchone()
        except Exception:
            row = None
        if row and row[0]:
            stored_dim = len(row[0]) // 4  # float32 = 4 bytes each
            if stored_dim != configured_dim:
                raise EmbeddingDimensionMismatchError(
                    stored_dim=stored_dim,
                    configured_dim=configured_dim,
                    store_path=str(self.db_path),
                )

        # DDL-based check — catches empty stores with a mismatched
        # vec_facts schema. Parses "embedding float[N]" out of the
        # CREATE VIRTUAL TABLE statement.
        try:
            ddl_row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'vec_facts'"
            ).fetchone()
        except Exception:
            return
        if not ddl_row or not ddl_row[0]:
            return
        import re
        m = re.search(r"embedding\s+float\[(\d+)\]", ddl_row[0], re.IGNORECASE)
        if not m:
            return
        schema_dim = int(m.group(1))
        if schema_dim != configured_dim:
            raise EmbeddingDimensionMismatchError(
                stored_dim=schema_dim,
                configured_dim=configured_dim,
                store_path=str(self.db_path),
            )

    # --- Facts CRUD ---

    def insert_fact(
        self,
        content: str,
        embedding: list[float] | None = None,
        *,
        entity_ids: list[str] | None = None,
        source: str | None = None,
        confidence: float = 0.7,
        fact_type: str = "objective",
        session_coherence_score: float | None = None,
        # ── Schema v2 fields (all optional, NULL on legacy inserts) ──
        subject_entity_id: str | None = None,
        predicate: str | None = None,
        object_entity_id: str | None = None,
        object_literal: str | None = None,
        slot_key: str | None = None,
        valid_from: str | None = None,
        valid_to: str | None = None,
        category: str | None = None,
        source_turn_id: str | None = None,
        extraction_method: str | None = None,
    ) -> str:
        """Insert a fact and its embedding vector. Returns the fact ID.

        Schema v2 fields are optional; when NULL the row is
        indistinguishable from a legacy row and is not served by the
        slot_lookup / timeline retrieval paths.
        """
        fact_id = _new_id()
        now = _now_iso()
        emb_blob = serialize_float32(embedding) if embedding else None

        self.conn.execute(
            """INSERT INTO facts
               (id, content, embedding, entity_ids, source, confidence,
                fact_type, status, status_detail, created_at, last_confirmed_at,
                subject_entity_id, predicate, object_entity_id, object_literal,
                slot_key, valid_from, valid_to, category, source_turn_id,
                extraction_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'current', 'provisional', ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact_id, content, emb_blob,
                json.dumps(entity_ids) if entity_ids else None,
                source, confidence, fact_type, now, now,
                subject_entity_id, predicate, object_entity_id, object_literal,
                slot_key, valid_from, valid_to, category, source_turn_id,
                extraction_method,
            ),
        )

        if embedding:
            self.conn.execute(
                "INSERT INTO vec_facts(fact_id, embedding) VALUES (?, ?)",
                (fact_id, serialize_float32(embedding)),
            )

        self.conn.commit()
        self._audit("extract", "fact", fact_id)
        return fact_id

    def get_fact(self, fact_id: str) -> dict[str, Any] | None:
        """Get a fact by ID."""
        row = self.conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict("facts", row)

    def count_current_facts(self) -> int:
        """Return the number of ``status='current'`` facts in the store.

        Used by the progressive-activation phase detector. Cheap (single
        indexed SELECT COUNT) and safe to call per-request; the proxy
        queries it at the top of every intercepted chat to pick OBSERVE
        / ACCUMULATE / ACTIVATE.
        """
        row = self.conn.execute(
            "SELECT count(*) FROM facts WHERE status = 'current'"
        ).fetchone()
        return int(row[0]) if row else 0

    def get_facts(self, status: str = "current", limit: int = 100) -> list[dict[str, Any]]:
        """Get facts filtered by status."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [self._row_to_dict("facts", r) for r in rows]

    def search_facts_by_vector(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        """Find facts by vector similarity. Returns facts with distance scores."""
        rows = self.conn.execute(
            """SELECT f.*, v.distance
               FROM vec_facts v
               JOIN facts f ON f.id = v.fact_id
               WHERE v.embedding MATCH ?
                 AND k = ?
               ORDER BY v.distance""",
            (serialize_float32(query_embedding), limit),
        ).fetchall()

        results = []
        for row in rows:
            fact = self._row_to_dict("facts", row[:-1])  # last col is distance
            fact["distance"] = row[-1]
            if min_distance is not None and fact["distance"] > min_distance:
                continue
            results.append(fact)

        return results

    def update_fact_retrieval(self, fact_id: str) -> None:
        """Increment retrieval count and update last_retrieved_at."""
        now = _now_iso()
        self.conn.execute(
            """UPDATE facts SET retrieval_count = retrieval_count + 1,
               last_retrieved_at = ? WHERE id = ?""",
            (now, fact_id),
        )
        self.conn.commit()
        self._audit("retrieve", "fact", fact_id)

    def update_fact_status(
        self,
        fact_id: str,
        status: str,
        status_detail: str | None = None,
        superseded_by: str | None = None,
        ambiguity_note: str | None = None,
    ) -> None:
        """Update a fact's status and optional metadata."""
        self.conn.execute(
            """UPDATE facts
               SET status = ?, status_detail = COALESCE(?, status_detail),
                   superseded_by = COALESCE(?, superseded_by),
                   ambiguity_note = COALESCE(?, ambiguity_note)
               WHERE id = ?""",
            (status, status_detail, superseded_by, ambiguity_note, fact_id),
        )
        self.conn.commit()
        self._audit("update_status", "fact", fact_id)

    def boost_fact_confidence(self, fact_id: str, boost: float = 0.05) -> None:
        """Boost confidence and update last_confirmed_at for re-confirmed facts."""
        now = _now_iso()
        self.conn.execute(
            """UPDATE facts
               SET confidence = MIN(1.0, confidence + ?),
                   last_confirmed_at = ?,
                   usage_count = usage_count + 1
               WHERE id = ?""",
            (boost, now, fact_id),
        )
        self.conn.commit()

    def find_similar_facts(
        self,
        embedding: list[float],
        limit: int = 5,
        max_distance: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find facts within max_distance of the given embedding. Like search_facts_by_vector but filtered."""
        results = self.search_facts_by_vector(embedding, limit=limit)
        return [r for r in results if r.get("distance", 999) < max_distance]

    def get_fact_confirmation_count(self, fact_id: str) -> int:
        """Return the usage_count for a fact (how many times it's been re-confirmed)."""
        row = self.conn.execute(
            "SELECT usage_count FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return row[0] if row else 0

    # --- Entities CRUD ---

    def insert_entity(
        self,
        name: str,
        type: str | None = None,
        description: str | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Insert an entity. Returns the entity ID."""
        entity_id = _new_id()
        emb_blob = serialize_float32(embedding) if embedding else None
        self.conn.execute(
            """INSERT INTO entities (id, name, type, description, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (entity_id, name, type, description, emb_blob, _now_iso()),
        )
        self.conn.commit()
        self._audit("extract", "entity", entity_id)
        return entity_id

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Get an entity by ID."""
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict("entities", row)

    def find_entity_by_name(self, name: str) -> dict[str, Any] | None:
        """Find an entity by case-insensitive name match.

        D38: prior exact-match behaviour fragmented "Mum" vs "mum" vs
        "MUM" into 3 separate entities. Case-folded LIKE retains an
        existing entity across surface-form variation.
        """
        row = self.conn.execute(
            "SELECT * FROM entities WHERE LOWER(name) = LOWER(?) "
            "ORDER BY created_at ASC LIMIT 1", (name,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict("entities", row)

    # --- Relationships CRUD ---

    def insert_relationship(
        self,
        source_entity: str,
        relationship: str,
        target_entity: str,
        confidence: float = 0.7,
    ) -> str:
        """Insert a relationship between two entities. Returns relationship ID.

        D40: if an identical current edge already exists (same source,
        relationship, target), reuse it instead of creating a duplicate.
        The first 30-day run accumulated 2-3 duplicate edges per
        relationship (e.g. User → sister → Amy appeared 3 times with
        different confidences). Dedup at insert time with confidence
        merged via max().
        """
        existing = self.conn.execute(
            """SELECT id, confidence FROM relationships
               WHERE source_entity = ? AND LOWER(relationship) = LOWER(?)
                 AND target_entity = ? AND status = 'current'
               LIMIT 1""",
            (source_entity, relationship, target_entity),
        ).fetchone()
        if existing is not None:
            existing_id, existing_conf = existing[0], existing[1]
            merged = max(existing_conf or 0.0, confidence)
            if merged != existing_conf:
                self.conn.execute(
                    "UPDATE relationships SET confidence = ? WHERE id = ?",
                    (merged, existing_id),
                )
                self.conn.commit()
            return existing_id
        rel_id = _new_id()
        self.conn.execute(
            """INSERT INTO relationships
               (id, source_entity, relationship, target_entity, confidence, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'current', ?)""",
            (rel_id, source_entity, relationship, target_entity, confidence, _now_iso()),
        )
        self.conn.commit()
        return rel_id

    def get_related_entities(self, entity_id: str) -> list[dict[str, Any]]:
        """Get all entities related to the given entity (1-hop traversal)."""
        rows = self.conn.execute(
            """SELECT e.*, r.relationship, r.confidence as rel_confidence
               FROM relationships r
               JOIN entities e ON (e.id = r.target_entity OR e.id = r.source_entity)
               WHERE (r.source_entity = ? OR r.target_entity = ?)
                 AND e.id != ?
                 AND r.status = 'current'""",
            (entity_id, entity_id, entity_id),
        ).fetchall()

        results = []
        for row in rows:
            entity = self._row_to_dict("entities", row[:6])
            entity["relationship"] = row[6]
            entity["rel_confidence"] = row[7]
            results.append(entity)
        return results

    # --- Episodes CRUD ---

    def insert_episode(
        self,
        summary: str,
        embedding: list[float] | None = None,
        *,
        entities_involved: list[str] | None = None,
        decisions_made: list[str] | None = None,
        session_id: str | None = None,
    ) -> str:
        """Insert an episode. Returns the episode ID."""
        episode_id = _new_id()
        emb_blob = serialize_float32(embedding) if embedding else None
        self.conn.execute(
            """INSERT INTO episodes
               (id, summary, embedding, entities_involved, decisions_made, session_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                episode_id, summary, emb_blob,
                json.dumps(entities_involved) if entities_involved else None,
                json.dumps(decisions_made) if decisions_made else None,
                session_id, _now_iso(),
            ),
        )
        if embedding:
            try:
                self.conn.execute(
                    "INSERT INTO vec_episodes(episode_id, embedding) VALUES (?, ?)",
                    (episode_id, serialize_float32(embedding)),
                )
            except Exception as exc:
                logger.warning("vec_episodes insert failed for %s: %s", episode_id, exc)
        self.conn.commit()
        return episode_id

    def search_episodes_by_vector(
        self,
        query_embedding: list[float],
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        """Find episodes by vector similarity. Returns episodes with distance.

        Phase-3 Fix 2: used by ContextRetriever when include_episodes=True
        so follow-up queries can surface the compressed prior-turn
        summary that matches their topic.
        """
        try:
            rows = self.conn.execute(
                """SELECT e.id, e.summary, e.entities_involved,
                          e.decisions_made, e.session_id, e.created_at,
                          v.distance
                   FROM vec_episodes v
                   JOIN episodes e ON e.id = v.episode_id
                   WHERE v.embedding MATCH ?
                     AND k = ?
                   ORDER BY v.distance""",
                (serialize_float32(query_embedding), limit),
            ).fetchall()
        except Exception as exc:
            logger.warning("vec_episodes search failed: %s", exc)
            return []

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({
                "id": row[0],
                "summary": row[1],
                "entities_involved": json.loads(row[2]) if row[2] else None,
                "decisions_made": json.loads(row[3]) if row[3] else None,
                "session_id": row[4],
                "created_at": row[5],
                "distance": row[6],
            })
        return results

    # --- Preferences CRUD ---

    def insert_preference(
        self,
        category: str,
        content: str,
        strength: float = 0.5,
    ) -> str:
        """Insert a preference. Returns the preference ID."""
        pref_id = _new_id()
        self.conn.execute(
            """INSERT INTO preferences (id, category, content, strength, observation_count, last_observed_at)
               VALUES (?, ?, ?, ?, 1, ?)""",
            (pref_id, category, content, strength, _now_iso()),
        )
        self.conn.commit()
        return pref_id

    def upsert_preference(
        self,
        category: str,
        content: str,
        strength: float = 0.5,
    ) -> str:
        """Insert or update a preference. Returns the preference ID."""
        row = self.conn.execute(
            "SELECT id FROM preferences WHERE category = ? AND content = ?",
            (category, content),
        ).fetchone()
        if row:
            self.conn.execute(
                """UPDATE preferences
                   SET strength = ?, observation_count = observation_count + 1,
                       last_observed_at = ?
                   WHERE id = ?""",
                (strength, _now_iso(), row[0]),
            )
            self.conn.commit()
            return row[0]
        return self.insert_preference(category, content, strength)

    def get_preferences(self, category: str | None = None) -> list[dict[str, Any]]:
        """Get preferences, optionally filtered by category."""
        if category:
            rows = self.conn.execute(
                "SELECT * FROM preferences WHERE category = ? ORDER BY strength DESC",
                (category,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM preferences ORDER BY category, strength DESC",
            ).fetchall()
        return [self._row_to_dict("preferences", r) for r in rows]

    # --- Learning queries ---

    def get_all_facts_with_usage(self) -> list[dict[str, Any]]:
        """Return all current facts with retrieval/usage stats for the tuning loop."""
        rows = self.conn.execute("""
            SELECT id, content, confidence, fact_type, status,
                   retrieval_count, usage_count, created_at
            FROM facts
            WHERE status IN ('current', 'provisional')
            ORDER BY created_at DESC
        """).fetchall()
        cols = ["id", "content", "confidence", "fact_type", "status",
                "retrieval_count", "usage_count", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    def get_interaction_count(self) -> int:
        """Return total number of audit log entries for 'extract' operations (proxy for interaction count)."""
        row = self.conn.execute(
            "SELECT count(*) FROM audit_log WHERE operation = 'interaction'"
        ).fetchone()
        return row[0] if row else 0

    def log_interaction(self, session_id: str | None = None) -> None:
        """Log an interaction for the tuning loop counter."""
        self._audit("interaction", "session", session_id or "unknown", session_id)

    # --- Sessions CRUD ---

    def insert_session(self) -> str:
        """Start a new session. Returns the session ID."""
        session_id = _new_id()
        self.conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
            (session_id, _now_iso()),
        )
        self.conn.commit()
        return session_id

    def end_session(self, session_id: str, coherence_score: float | None = None) -> None:
        """Mark a session as ended."""
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?, coherence_score = ? WHERE id = ?",
            (_now_iso(), coherence_score, session_id),
        )
        self.conn.commit()

    # --- Fingerprints CRUD ---

    def get_fingerprint(self, section_key: str) -> dict[str, Any] | None:
        """Get a stored fingerprint by section key."""
        row = self.conn.execute(
            "SELECT * FROM fingerprints WHERE section_key = ?", (section_key,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict("fingerprints", row)

    def upsert_fingerprint(
        self,
        section_key: str,
        hash_value: str,
        content: str | None = None,
        token_count: int | None = None,
    ) -> None:
        """Insert or update a fingerprint hash for a payload section."""
        now = _now_iso()
        self.conn.execute(
            """INSERT INTO fingerprints (section_key, hash, content, token_count, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(section_key)
               DO UPDATE SET hash=excluded.hash, content=excluded.content,
                            token_count=excluded.token_count, updated_at=excluded.updated_at""",
            (section_key, hash_value, content, token_count, now),
        )
        self.conn.commit()

    def get_all_fingerprints(self) -> dict[str, str]:
        """Get all stored fingerprints as {section_key: hash}."""
        rows = self.conn.execute("SELECT section_key, hash FROM fingerprints").fetchall()
        return {row[0]: row[1] for row in rows}

    # --- Stats ---

    def stats(self) -> dict[str, Any]:
        """Return counts for all memory types."""
        tables = ["facts", "entities", "relationships", "episodes", "preferences", "sessions", "known_unknowns"]
        result: dict[str, Any] = {}
        for table in tables:
            count = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608
            result[f"{table}_count"] = count

        # Vec index count
        vec_count = self.conn.execute("SELECT count(*) FROM vec_facts").fetchone()[0]
        result["vec_facts_count"] = vec_count

        # DB file size
        if self.db_path.exists():
            result["db_size_bytes"] = self.db_path.stat().st_size

        return result

    # --- known_unknowns CRUD ---

    def insert_known_unknown(
        self,
        subject_entity_id: str,
        slot_key: str,
        reason: str | None = None,
    ) -> str:
        """Record that a slot was asked-about and is not in the store.

        Idempotent on (subject_entity_id, slot_key) — a second insert is a
        no-op. Returns the row id (existing or new).
        """
        row = self.conn.execute(
            "SELECT id FROM known_unknowns WHERE subject_entity_id = ? AND slot_key = ?",
            (subject_entity_id, slot_key),
        ).fetchone()
        if row:
            return row[0]
        ku_id = _new_id()
        self.conn.execute(
            "INSERT INTO known_unknowns (id, subject_entity_id, slot_key, reason, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (ku_id, subject_entity_id, slot_key, reason, _now_iso()),
        )
        self.conn.commit()
        return ku_id

    def get_known_unknowns(self, subject_entity_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, subject_entity_id, slot_key, reason, created_at"
            " FROM known_unknowns WHERE subject_entity_id = ? ORDER BY created_at DESC",
            (subject_entity_id,),
        ).fetchall()
        return [
            {"id": r[0], "subject_entity_id": r[1], "slot_key": r[2], "reason": r[3], "created_at": r[4]}
            for r in rows
        ]

    def get_current_slot_fact(self, slot_key: str) -> dict | None:
        """Return the single current fact for a slot, or None.

        Drives deterministic slot_lookup retrieval: uses the partial
        index idx_facts_slot_current (slot_key IS NOT NULL AND
        valid_to IS NULL AND status IN ('current','provisional')).
        """
        row = self.conn.execute(
            "SELECT id, content, slot_key, predicate, object_literal, valid_from,"
            "       category, confidence, status, source_turn_id"
            "  FROM facts"
            " WHERE slot_key = ? AND valid_to IS NULL"
            "   AND status IN ('current', 'provisional')"
            " ORDER BY valid_from DESC NULLS LAST, created_at DESC"
            " LIMIT 1",
            (slot_key,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "content": row[1], "slot_key": row[2], "predicate": row[3],
            "object_literal": row[4], "valid_from": row[5], "category": row[6],
            "confidence": row[7], "status": row[8], "source_turn_id": row[9],
        }

    def get_slot_timeline(self, slot_key: str) -> list[dict]:
        """Return all facts (including superseded) for a slot, in time order."""
        rows = self.conn.execute(
            "SELECT id, content, slot_key, predicate, object_literal,"
            "       valid_from, valid_to, category, status, superseded_by"
            "  FROM facts"
            " WHERE slot_key = ?"
            " ORDER BY valid_from ASC NULLS FIRST, created_at ASC",
            (slot_key,),
        ).fetchall()
        return [
            {"id": r[0], "content": r[1], "slot_key": r[2], "predicate": r[3],
             "object_literal": r[4], "valid_from": r[5], "valid_to": r[6],
             "category": r[7], "status": r[8], "superseded_by": r[9]}
            for r in rows
        ]

    def supersede_slot(self, slot_key: str, new_fact_id: str, as_of: str) -> int:
        """Mark all current rows for a slot as superseded by new_fact_id.

        Returns the number of rows updated. Called from the writer's S3
        stage when schema_v2 is enabled.
        """
        cursor = self.conn.execute(
            "UPDATE facts SET valid_to = ?, superseded_by = ?"
            " WHERE slot_key = ? AND id != ? AND valid_to IS NULL"
            "   AND status IN ('current', 'provisional')",
            (as_of, new_fact_id, slot_key, new_fact_id),
        )
        self.conn.commit()
        return cursor.rowcount

    # --- Audit ---

    def _audit(self, operation: str, target_type: str, target_id: str, session_id: str | None = None) -> None:
        """Write an audit log entry (operation only, never content)."""
        self.conn.execute(
            "INSERT INTO audit_log (id, operation, target_type, target_id, session_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (_new_id(), operation, target_type, target_id, session_id, _now_iso()),
        )
        self.conn.commit()

    # --- Helpers ---

    def _row_to_dict(self, table: str, row: tuple) -> dict[str, Any]:
        """Convert a database row to a dict using column names."""
        cols = [desc[0] for desc in self.conn.execute(f"SELECT * FROM {table} LIMIT 0").description]  # noqa: S608
        return dict(zip(cols, row[:len(cols)]))

"""Store inspection helpers for `sieve store facts/entities/...`.

The CLI layer in cli.py opens a MemoryStore, hands the connection to the
pure query functions here, and renders rich tables from their output.
This module does zero I/O beyond SQL reads on an already-open connection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# ── Queries ────────────────────────────────────────────────────────────────

def list_facts(conn, limit: int = 50, search: str | None = None) -> list[dict]:
    if search:
        rows = conn.execute(
            "SELECT id, content, confidence, source, created_at "
            "FROM facts "
            "WHERE status = 'current' AND content LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (f"%{search}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content, confidence, source, created_at "
            "FROM facts WHERE status = 'current' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    cols = ["id", "content", "confidence", "source", "created_at"]
    return [dict(zip(cols, r)) for r in rows]


def list_entities(conn, limit: int = 50, search: str | None = None) -> list[dict]:
    if search:
        rows = conn.execute(
            "SELECT e.id, e.name, e.type, "
            "(SELECT count(*) FROM facts f WHERE f.entity_ids LIKE '%' || e.id || '%') AS fact_count "
            "FROM entities e WHERE e.name LIKE ? "
            "ORDER BY e.created_at DESC LIMIT ?",
            (f"%{search}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT e.id, e.name, e.type, "
            "(SELECT count(*) FROM facts f WHERE f.entity_ids LIKE '%' || e.id || '%') AS fact_count "
            "FROM entities e ORDER BY e.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    cols = ["id", "name", "type", "fact_count"]
    return [dict(zip(cols, r)) for r in rows]


def list_relationships(conn, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT r.id, s.name AS source_name, r.relationship, "
        "       t.name AS target_name, r.confidence, r.status "
        "FROM relationships r "
        "JOIN entities s ON s.id = r.source_entity "
        "JOIN entities t ON t.id = r.target_entity "
        "ORDER BY r.created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    cols = ["id", "source_name", "relationship", "target_name", "confidence", "status"]
    return [dict(zip(cols, r)) for r in rows]


def list_episodes(conn, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT id, summary, entities_involved, created_at "
        "FROM episodes ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    cols = ["id", "summary", "entities_involved", "created_at"]
    return [dict(zip(cols, r)) for r in rows]


# ── Export ─────────────────────────────────────────────────────────────────

def export_json(conn, out: Path) -> None:
    payload = {
        "facts": list_facts(conn, limit=1_000_000),
        "entities": list_entities(conn, limit=1_000_000),
        "relationships": list_relationships(conn, limit=1_000_000),
        "episodes": list_episodes(conn, limit=1_000_000),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))


def export_csv(conn, out_dir: Path) -> None:
    """Write one CSV per section under out_dir/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = {
        "facts": list_facts(conn, limit=1_000_000),
        "entities": list_entities(conn, limit=1_000_000),
        "relationships": list_relationships(conn, limit=1_000_000),
        "episodes": list_episodes(conn, limit=1_000_000),
    }
    for name, rows in sections.items():
        path = out_dir / f"{name}.csv"
        if not rows:
            path.write_text("")
            continue
        with open(path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ── Wipe ───────────────────────────────────────────────────────────────────

_WIPE_TABLES = (
    # Children first so FKs (relationships → entities) don't block.
    "facts", "relationships", "episodes", "known_unknowns",
    "preferences", "sessions", "audit_log", "fingerprints",
    "entities",
)


def wipe_store(conn) -> None:
    """Delete all data rows but preserve the schema. Also clears the vec
    virtual table so embeddings don't outlive their facts."""
    for table in _WIPE_TABLES:
        try:
            conn.execute(f"DELETE FROM {table}")  # noqa: S608
        except Exception:
            # tables like audit_log may not exist on very old DBs
            pass
    try:
        conn.execute("DELETE FROM vec_facts")
    except Exception:
        pass
    conn.commit()


# ── Detailed stats ────────────────────────────────────────────────────────

def detailed_stats(conn, db_path: Path) -> dict[str, Any]:
    """Return a richer stats dict than MemoryStore.stats(): per-table row
    counts, average facts per entity, vec index size."""
    rows: dict[str, Any] = {}
    for table in _WIPE_TABLES + ("vec_facts",):
        try:
            rows[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # noqa: S608
        except Exception:
            rows[table] = 0
    rows["db_size_bytes"] = db_path.stat().st_size if db_path.exists() else 0
    entities = rows.get("entities", 0) or 1
    facts = rows.get("facts", 0)
    rows["avg_facts_per_entity"] = round(facts / entities, 2) if entities else 0
    return rows

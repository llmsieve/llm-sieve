#!/usr/bin/env python3
"""Seed test — validates the full Stage 1 writer pipeline without live Ollama.

Simulates a conversation containing personal facts and prints the resulting
store contents: facts, entities, relationships with confidence scores.

Usage:
    source .venv/bin/activate
    python tests/seed_test.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.writer import MemoryWriter, extract_facts_s1

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def _fake_embed(text: str) -> list[float]:
    """Deterministic 768-dim unit embedding. No Ollama required."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    raw = [(b - 128) / 128.0 for b in h]
    vec = (raw * (768 // len(raw) + 1))[:768]
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0 else vec


async def _async_embed(text: str) -> list[float]:
    return _fake_embed(text)


# ── conversation to seed ─────────────────────────────────────────────────────
CONVERSATION = [
    # (role, text)
    ("user",      "Hi there! I'm a pilot based in Dubai."),
    ("assistant", "Great to meet you! How long have you been flying?"),
    ("user",      "About 12 years. My partner is Madeline and I have two children."),
    ("assistant", "That's wonderful. Do your kids enjoy travelling?"),
    ("user",      "They love it. I work for Emirates Airlines. I'm 38 years old."),
    ("assistant", "Emirates is an amazing airline. Where did you grow up?"),
    ("user",      "I'm originally from Melbourne, Australia. Moved to Dubai five years ago."),
    ("assistant", "What a journey! What do you enjoy doing in Dubai?"),
    ("user",      "I love the food here. My best friend is Marcus, he's also a pilot."),
]


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="recall_seed_") as tmp_dir:
        db_path = Path(tmp_dir) / "seed.db"
        config = StoreConfig(
            path=str(db_path),
            embedding_dimensions=768,
        )

        store = MemoryStore(config, passphrase="seed-test-pass")
        store.open()
        store.init_schema()

        writer = MemoryWriter(store, embed_fn=_async_embed)

        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}  Recall — Stage 1 Writer Seed Test{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}")
        print(f"\n{DIM}Processing {len(CONVERSATION)} conversation turns...{RESET}\n")

        # Process each user turn
        session_id = "seed-session-001"
        for role, text in CONVERSATION:
            if role != "user":
                continue

            print(f"  {CYAN}User:{RESET} {text}")
            candidates = extract_facts_s1(text)
            if candidates:
                print(f"  {DIM}  → S1 extracted {len(candidates)} candidate(s):{RESET}")
                for c in candidates:
                    print(f"  {DIM}    [{c.category}] {c.content}{RESET}")

            result = await writer.process(text, session_id=session_id)
            if result.facts_written or result.facts_skipped:
                print(f"  {GREEN}  ✓ Written: {result.facts_written}  Skipped (dedup): {result.facts_skipped}  "
                      f"Entities: {result.entities_written}  Rels: {result.relationships_written}  "
                      f"({result.elapsed_ms:.1f}ms){RESET}")
            print()

        # ── Print store contents ──────────────────────────────────────────────
        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}  What's in the store after seeding:{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}")

        facts = store.get_facts()
        print(f"\n  {BOLD}Facts ({len(facts)} total):{RESET}")
        for f in facts:
            conf_bar = "█" * int(f["confidence"] * 10)
            conf_pct = f"{f['confidence']:.0%}"
            status = f["status_detail"]
            ftype = f["fact_type"]
            has_vec = "🔢" if f["embedding"] else "  "
            print(f"    {has_vec} [{ftype:12s}] {f['content']}")
            print(f"       {DIM}confidence={conf_pct} {conf_bar:10s}  status={status}  id={f['id'][:10]}...{RESET}")

        entities = store.conn.execute(
            "SELECT id, name, type FROM entities ORDER BY name"
        ).fetchall()
        print(f"\n  {BOLD}Entities ({len(entities)} total):{RESET}")
        for eid, name, etype in entities:
            print(f"    • {CYAN}{name}{RESET} ({etype or '?'})  {DIM}id={eid[:10]}...{RESET}")

        rels = store.conn.execute("""
            SELECT e1.name, r.relationship, e2.name, r.confidence
            FROM relationships r
            JOIN entities e1 ON e1.id = r.source_entity
            JOIN entities e2 ON e2.id = r.target_entity
            WHERE r.status = 'current'
        """).fetchall()
        print(f"\n  {BOLD}Relationships ({len(rels)} total):{RESET}")
        for src, rel, tgt, conf in rels:
            print(f"    • {CYAN}{src}{RESET} --[{YELLOW}{rel}{RESET}]--> {CYAN}{tgt}{RESET}  "
                  f"{DIM}(confidence={conf:.0%}){RESET}")

        stats = store.stats()
        print(f"\n  {BOLD}Store stats:{RESET}")
        print(f"    {DIM}Facts: {stats['facts_count']}  Entities: {stats['entities_count']}  "
              f"Relationships: {stats['relationships_count']}  Vectors: {stats['vec_facts_count']}{RESET}")
        print(f"    {DIM}DB size: {stats.get('db_size_bytes', 0) / 1024:.1f} KB (encrypted){RESET}")

        store.close()
        print(f"\n  {DIM}(temp store cleaned up){RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())

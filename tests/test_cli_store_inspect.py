"""Tests for `sieve store facts/entities/relationships/episodes/stats/export/wipe`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    """Build a real encrypted store with a handful of rows, redirect the
    CLI to use it via SIEVE_CONFIG env.

    This is a small integration fixture — store inspection has to exercise
    the real MemoryStore because we're testing that SQL reads return the
    right shape. We keep it tiny (one fact, one entity, one episode)."""
    from sieve.config import StoreConfig
    from sieve.store import MemoryStore

    db = tmp_path / "memory.db"
    cfg = tmp_path / "sieve.yaml"
    cfg.write_text(yaml.safe_dump({
        "listen": {"port": 11435},
        "provider": {"base_url": "http://127.0.0.1:11434",
                     "default_model": "qwen3.5:9b"},
        "store": {"path": str(db)},
    }))
    monkeypatch.setenv("SIEVE_CONFIG", str(cfg))

    ms = MemoryStore(StoreConfig(path=str(db)))
    ms.open()
    ms.init_schema()

    # Seed: two entities, two facts, one relationship, one episode
    e1 = ms.insert_entity(name="Casey", type="person")
    e2 = ms.insert_entity(name="Mabel", type="pet")
    ms.insert_fact(
        content="Casey works as a landscape architect.",
        entity_ids=[e1],
        source="writer",
    )
    ms.insert_fact(
        content="Mabel is a border terrier.",
        entity_ids=[e2],
        source="writer",
    )
    ms.insert_relationship(
        source_entity=e1, relationship="owns", target_entity=e2,
    )
    ms.insert_episode(summary="First conversation with Casey.", entities_involved=[e1])
    ms.close()

    return {"db": db, "config": cfg, "e1": e1, "e2": e2}


def test_store_facts_lists_rows(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "facts"])
    assert result.exit_code == 0, result.output
    assert "Casey works as a landscape architect" in result.output
    assert "Mabel is a border terrier" in result.output


def test_store_facts_respects_limit(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "facts", "--limit", "1"])
    assert result.exit_code == 0
    # Only one of the two facts should appear
    has_casey = "Casey works" in result.output
    has_mabel = "Mabel is a border terrier" in result.output
    assert has_casey ^ has_mabel  # exactly one


def test_store_facts_search_filter(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "facts", "--search", "Mabel"])
    assert result.exit_code == 0
    assert "Mabel" in result.output
    assert "landscape architect" not in result.output


def test_store_entities_lists_rows(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "entities"])
    assert result.exit_code == 0
    assert "Casey" in result.output
    assert "Mabel" in result.output


def test_store_relationships_lists_rows(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "relationships"])
    assert result.exit_code == 0
    assert "owns" in result.output


def test_store_episodes_lists_rows(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "episodes"])
    assert result.exit_code == 0
    assert "First conversation with Casey" in result.output


def test_store_stats_detailed_output(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "stats"])
    assert result.exit_code == 0
    # Should show the fact count (2) and entity count (2) somewhere
    assert "facts" in result.output.lower()
    assert "2" in result.output


def test_store_export_json(populated_store, tmp_path):
    from sieve import cli as cli_mod
    out = tmp_path / "dump.json"
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli, ["store", "export", "--format", "json",
                      "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    # Must contain the three top-level sections
    assert "facts" in data and "entities" in data and "relationships" in data
    assert len(data["facts"]) == 2
    assert len(data["entities"]) == 2


def test_store_export_csv_writes_directory(populated_store, tmp_path):
    from sieve import cli as cli_mod
    out = tmp_path / "csv_out"
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli, ["store", "export", "--format", "csv",
                      "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "facts.csv").exists()
    assert (out / "entities.csv").exists()


def test_store_wipe_requires_typed_confirmation(populated_store):
    """Wipe is destructive — anything other than the literal 'WIPE'
    must abort."""
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "wipe"], input="wipe\n")  # lowercase
    assert result.exit_code != 0
    # Data should still be present
    from sieve.store import MemoryStore
    from sieve.config import StoreConfig
    ms = MemoryStore(StoreConfig(path=str(populated_store["db"])))
    ms.open()
    assert ms.stats()["facts_count"] == 2
    ms.close()


def test_store_wipe_with_correct_confirmation(populated_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["store", "wipe"], input="WIPE\n")
    assert result.exit_code == 0, result.output

    # All tables should now be empty
    from sieve.store import MemoryStore
    from sieve.config import StoreConfig
    ms = MemoryStore(StoreConfig(path=str(populated_store["db"])))
    ms.open()
    stats = ms.stats()
    assert stats["facts_count"] == 0
    assert stats["entities_count"] == 0
    assert stats["relationships_count"] == 0
    ms.close()

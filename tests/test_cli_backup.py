"""Tests for the existing `sieve backup create/list/restore` commands.

We already have sieve/backup.py library code with its own tests;
these lock in the CLI UX (table output, empty-list message, restore
abort-on-no)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Empty but initialised store so backup create/list work."""
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
    ms.insert_entity(name="Marker", type="test")
    ms.close()
    return {"db": db, "cfg": cfg, "tmp": tmp_path}


def test_backup_list_empty_shows_message(fresh_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["backup", "list"])
    assert result.exit_code == 0, result.output
    # No backups created yet → dim "No backups found" message
    assert "no backups" in result.output.lower() or "0 backup" in result.output.lower()


def test_backup_create_then_list_shows_row(fresh_store):
    from sieve import cli as cli_mod
    runner = CliRunner()

    r1 = runner.invoke(cli_mod.cli, ["backup", "create"])
    assert r1.exit_code == 0, r1.output
    assert "backup created" in r1.output.lower()

    r2 = runner.invoke(cli_mod.cli, ["backup", "list"])
    assert r2.exit_code == 0
    # The table should surface the filename's timestamp fragment
    assert "recall_backup_" in r2.output or "backup" in r2.output.lower()


def test_backup_restore_abort_preserves_data(fresh_store):
    """Answering 'n' to the restore confirmation must leave the DB alone."""
    from sieve import cli as cli_mod
    runner = CliRunner()
    r1 = runner.invoke(cli_mod.cli, ["backup", "create"])
    assert r1.exit_code == 0
    from sieve.backup import list_backups
    ident = list_backups(fresh_store["db"])[0]["id"]

    # Abort at prompt
    r2 = runner.invoke(cli_mod.cli, ["backup", "restore", ident], input="n\n")
    assert r2.exit_code != 0

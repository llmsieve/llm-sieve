"""Tests for `sieve key show/rotate/export/import`."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def keyed_store(tmp_path, monkeypatch):
    """Create a real encrypted store with a known passphrase, wire up the
    CLI to find it via SIEVE_CONFIG."""
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

    keyfile = tmp_path / ".sieve_key"
    return {"db": db, "cfg": cfg, "keyfile": keyfile, "tmp": tmp_path}


# ── Pure helpers ──────────────────────────────────────────────────────────

def test_fingerprint_is_hex_and_stable():
    from sieve.cli_keys import fingerprint
    fp1 = fingerprint("abc123")
    fp2 = fingerprint("abc123")
    assert fp1 == fp2
    assert all(c in "0123456789abcdef" for c in fp1)
    assert len(fp1) >= 16


def test_fingerprint_differs_per_key():
    from sieve.cli_keys import fingerprint
    assert fingerprint("a") != fingerprint("b")


def test_verify_key_rejects_wrong(keyed_store):
    from sieve.cli_keys import verify_key
    assert verify_key(keyed_store["db"], "wrong-key") is False


def test_verify_key_accepts_correct(keyed_store):
    from sieve.cli_keys import verify_key
    correct = keyed_store["keyfile"].read_text().strip()
    assert verify_key(keyed_store["db"], correct) is True


def test_rotate_key_re_encrypts_store(keyed_store):
    """After rotation, the old passphrase must NO LONGER open the DB
    and the new passphrase must succeed."""
    from sieve.cli_keys import rotate_key, verify_key

    old = keyed_store["keyfile"].read_text().strip()
    new_pp = "a" * 64  # any 64-char hex-ish string

    rotate_key(keyed_store["db"], old_key=old, new_key=new_pp)

    assert verify_key(keyed_store["db"], old) is False
    assert verify_key(keyed_store["db"], new_pp) is True

    # Keyfile on disk should now hold the new key
    assert keyed_store["keyfile"].read_text().strip() == new_pp
    assert oct(keyed_store["keyfile"].stat().st_mode)[-3:] == "600"


def test_rotate_key_rejects_wrong_old_key(keyed_store):
    from sieve.cli_keys import rotate_key
    with pytest.raises(ValueError, match="current key"):
        rotate_key(keyed_store["db"], old_key="wrong", new_key="irrelevant")


# ── CLI surface ───────────────────────────────────────────────────────────

def test_key_show(keyed_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["key", "show"])
    assert result.exit_code == 0, result.output
    assert str(keyed_store["keyfile"]) in result.output
    # The actual key must not leak
    key_contents = keyed_store["keyfile"].read_text().strip()
    assert key_contents not in result.output


def test_key_export_prints_key_with_warning(keyed_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["key", "export"], input="y\n")
    assert result.exit_code == 0
    key = keyed_store["keyfile"].read_text().strip()
    assert key in result.output
    assert "warning" in result.output.lower() or "store" in result.output.lower()


def test_key_export_aborts_without_confirmation(keyed_store):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["key", "export"], input="n\n")
    assert result.exit_code != 0
    key = keyed_store["keyfile"].read_text().strip()
    assert key not in result.output


def test_key_import_happy_path(keyed_store, tmp_path):
    """Moving the keyfile from one location to another — simulates
    a user who stored a backup of their key."""
    from sieve import cli as cli_mod

    # Pretend the user has a keyfile at a different path.
    current_key = keyed_store["keyfile"].read_text().strip()
    external = tmp_path / "my-key.txt"
    external.write_text(current_key)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["key", "import", str(external)])
    assert result.exit_code == 0, result.output
    # After import the canonical keyfile should still open the DB
    from sieve.cli_keys import verify_key
    assert verify_key(keyed_store["db"], keyed_store["keyfile"].read_text().strip())


def test_key_import_rejects_wrong_key(keyed_store, tmp_path):
    from sieve import cli as cli_mod
    bad = tmp_path / "bad-key.txt"
    bad.write_text("not-the-real-key")
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["key", "import", str(bad)])
    assert result.exit_code != 0


def test_key_rotate_requires_confirmation(keyed_store):
    """Rotate is destructive — anything other than 'ROTATE' aborts."""
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["key", "rotate"], input="rotate\n")
    assert result.exit_code != 0
    # Old key should still open the DB
    from sieve.cli_keys import verify_key
    assert verify_key(keyed_store["db"], keyed_store["keyfile"].read_text().strip())


def test_key_rotate_with_confirmation(keyed_store):
    from sieve import cli as cli_mod

    old = keyed_store["keyfile"].read_text().strip()
    runner = CliRunner()
    # Answer: ROTATE (confirmation) + "y" to auto-generate new key
    result = runner.invoke(cli_mod.cli, ["key", "rotate"],
                            input="ROTATE\ny\n")
    assert result.exit_code == 0, result.output

    new = keyed_store["keyfile"].read_text().strip()
    assert new != old
    from sieve.cli_keys import verify_key
    assert verify_key(keyed_store["db"], new)
    assert not verify_key(keyed_store["db"], old)

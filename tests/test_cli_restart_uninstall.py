"""Tests for `sieve restart` and `sieve uninstall --soft/--hard`."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_sieve_dir(tmp_path, monkeypatch):
    """Point all uninstall / restart paths at a tmp ~/.sieve."""
    d = tmp_path / ".sieve"
    d.mkdir()
    (d / "sieve.yaml").write_text("listen:\n  port: 11435\n")
    (d / "memory.db").write_bytes(b"fake db")
    (d / ".sieve_key").write_text("a" * 64)
    (d / "pid").write_text("99999")

    monkeypatch.setattr("sieve.cli_uninstall.SIEVE_DIR", d)
    monkeypatch.setattr("sieve.cli.SIEVE_DIR", d)
    return d


def test_uninstall_soft_preserves_data(fake_sieve_dir):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["uninstall", "--soft"])
    assert result.exit_code == 0, result.output

    # Data must still be there
    assert fake_sieve_dir.exists()
    assert (fake_sieve_dir / "memory.db").exists()
    assert (fake_sieve_dir / ".sieve_key").exists()
    # User should be told how to remove it and how to uninstall the pkg
    assert "pip uninstall" in result.output.lower() or "rm -rf" in result.output


def test_uninstall_default_is_soft(fake_sieve_dir):
    """Plain `sieve uninstall` (no flag) should behave like --soft."""
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["uninstall"])
    assert result.exit_code == 0, result.output
    assert fake_sieve_dir.exists()
    assert (fake_sieve_dir / "memory.db").exists()


def test_uninstall_hard_requires_typed_confirmation(fake_sieve_dir):
    """Type anything other than 'DELETE' → abort, data preserved."""
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["uninstall", "--hard"], input="delete\n")
    assert result.exit_code != 0
    assert fake_sieve_dir.exists()
    assert (fake_sieve_dir / "memory.db").exists()


def test_uninstall_hard_with_confirmation_removes_dir(fake_sieve_dir):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["uninstall", "--hard"], input="DELETE\n")
    assert result.exit_code == 0, result.output
    assert not fake_sieve_dir.exists()
    # Still remind the user how to drop the package
    assert "pip uninstall" in result.output.lower()


def test_uninstall_soft_and_hard_mutually_exclusive(fake_sieve_dir):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["uninstall", "--soft", "--hard"])
    assert result.exit_code != 0


# ── Restart ────────────────────────────────────────────────────────────────

def test_restart_calls_stop_then_execs_start(monkeypatch, fake_sieve_dir):
    """Restart should fully stop the running proxy, then hand off to
    start via os.execvp (the current process is replaced)."""
    from sieve import cli as cli_mod

    called: dict[str, int] = {"stop": 0, "execvp": 0}

    def fake_stop(pid_file=None):
        called["stop"] += 1
    def fake_execvp(prog, args):
        called["execvp"] += 1

    monkeypatch.setattr(cli_mod, "_stop_proxy", fake_stop)
    monkeypatch.setattr("os.execvp", fake_execvp)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["restart"])
    assert result.exit_code == 0, result.output
    assert called["stop"] == 1
    assert called["execvp"] == 1


def test_restart_passes_port_override(monkeypatch, fake_sieve_dir):
    from sieve import cli as cli_mod

    captured: dict[str, list] = {"argv": []}

    monkeypatch.setattr(cli_mod, "_stop_proxy", lambda pid_file=None: None)
    def fake_execvp(prog, args):
        captured["argv"] = list(args)
    monkeypatch.setattr("os.execvp", fake_execvp)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["restart", "--port", "12345"])
    assert result.exit_code == 0
    assert "--port" in captured["argv"]
    assert "12345" in captured["argv"]

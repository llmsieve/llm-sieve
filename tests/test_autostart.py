"""Tests for the autostart module (systemd user service, Linux-only).

Subprocesses are mocked — we verify the module calls the right
systemctl commands in the right order and produces the right unit
file, not that systemd itself behaves correctly (that's kernel
business).
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from sieve import _autostart


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def fake_unit_dir(tmp_path, monkeypatch):
    """Redirect ~/.config/systemd/user to a tmpdir so the test leaves
    nothing behind."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config" / "systemd" / "user"


@pytest.fixture
def linux_with_systemd(monkeypatch):
    """Pretend we're on Linux with systemctl + sieve available."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    def _which(cmd):
        return f"/usr/bin/{cmd}" if cmd in ("systemctl", "sieve") else None

    monkeypatch.setattr(_autostart.shutil, "which", _which)


# ── autostart_supported ────────────────────────────────────────────────


def test_not_supported_on_non_linux(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    assert _autostart.autostart_supported() is False


def test_not_supported_without_systemctl(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(_autostart.shutil, "which", lambda c: None)
    assert _autostart.autostart_supported() is False


def test_not_supported_when_systemctl_version_fails(linux_with_systemd, monkeypatch):
    def _fake_run(cmd, *a, **kw):
        # Simulate systemctl --user returning nonzero (no user bus).
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="bus err")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    assert _autostart.autostart_supported() is False


def test_supported_when_everything_present(linux_with_systemd, monkeypatch):
    def _fake_run(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="systemd 254", stderr="")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    assert _autostart.autostart_supported() is True


# ── autostart_status ──────────────────────────────────────────────────


def test_status_enabled(linux_with_systemd, monkeypatch):
    def _fake_run(cmd, *a, **kw):
        # systemctl --version → ok; is-enabled → enabled.
        if "is-enabled" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "enabled\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    assert _autostart.autostart_status() == "enabled"


def test_status_disabled_returns_disabled(linux_with_systemd, monkeypatch):
    def _fake_run(cmd, *a, **kw):
        if "is-enabled" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "disabled\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    assert _autostart.autostart_status() == "disabled"


def test_status_not_supported_without_linux(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert _autostart.autostart_status() == "not supported"


# ── enable_autostart ──────────────────────────────────────────────────


def test_sieve_binary_resolves_from_venv_bin(monkeypatch, tmp_path):
    """When the user runs sieve from a venv whose bin isn't on PATH,
    _sieve_binary must still find the executable via sys.executable's
    sibling directory. This is the exact scenario on BattleTest:
    /root/test-sieve/bin/python invokes sieve, PATH lookup fails."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_sieve = fake_bin / "sieve"
    fake_python.write_text("#!/bin/sh\n")
    fake_sieve.write_text("#!/bin/sh\n")
    fake_python.chmod(0o755)
    fake_sieve.chmod(0o755)
    import sys
    monkeypatch.setattr(sys, "executable", str(fake_python))
    monkeypatch.setattr(sys, "argv", ["/not/on/path"])
    monkeypatch.setattr(_autostart.shutil, "which", lambda c: None)
    result = _autostart._sieve_binary()
    assert result == str(fake_sieve)


def test_enable_writes_unit_file_and_calls_systemctl(
    linux_with_systemd, fake_unit_dir, monkeypatch
):
    calls: list[list[str]] = []

    def _fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    # Pin the resolved binary path so the assertion is stable
    # regardless of where the test suite runs.
    monkeypatch.setattr(_autostart, "_sieve_binary", lambda: "/usr/bin/sieve")
    ok, msg = _autostart.enable_autostart()
    assert ok, msg
    unit = fake_unit_dir / "sieve.service"
    assert unit.exists()
    text = unit.read_text()
    # Right shape + our binary path gets embedded.
    assert "[Unit]" in text and "[Service]" in text and "[Install]" in text
    assert "/usr/bin/sieve start --foreground" in text
    # We called daemon-reload then enable --now.
    cmd_strings = [" ".join(c) for c in calls]
    assert any("daemon-reload" in s for s in cmd_strings)
    assert any("enable --now sieve.service" in s for s in cmd_strings)


def test_enable_reports_error_when_systemctl_fails(
    linux_with_systemd, fake_unit_dir, monkeypatch
):
    def _fake_run(cmd, *a, **kw):
        # daemon-reload succeeds; enable fails.
        if "enable" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, "", "Unit sieve.service not found."
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    ok, msg = _autostart.enable_autostart()
    assert not ok
    assert "not found" in msg


# ── disable_autostart ─────────────────────────────────────────────────


def test_disable_idempotent_when_unit_missing(
    linux_with_systemd, fake_unit_dir, monkeypatch
):
    def _fake_run(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    # No unit file exists.
    ok, msg = _autostart.disable_autostart()
    assert ok
    assert "already disabled" in msg.lower()


def test_disable_removes_unit_file(linux_with_systemd, fake_unit_dir, monkeypatch):
    fake_unit_dir.mkdir(parents=True)
    unit = fake_unit_dir / "sieve.service"
    unit.write_text("dummy")
    def _fake_run(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(_autostart.subprocess, "run", _fake_run)
    ok, msg = _autostart.disable_autostart()
    assert ok
    assert not unit.exists()


# ── Uninstall hook ────────────────────────────────────────────────────


def test_remove_autostart_on_uninstall_swallows_errors(monkeypatch):
    """Uninstall must NEVER fail because systemd got grumpy."""
    monkeypatch.setattr(_autostart, "autostart_status", lambda: (_ for _ in ()).throw(RuntimeError("systemd exploded")))
    # Should not raise.
    _autostart.remove_autostart_on_uninstall()


def test_wipe_sieve_dir_runs_autostart_cleanup(monkeypatch, tmp_path):
    """The `sieve uninstall` path imports _autostart and calls the
    teardown hook, so removing autostart happens before we wipe
    the config dir."""
    from sieve import cli_uninstall
    called = []
    def _cleanup():
        called.append(True)
    monkeypatch.setattr(
        "sieve._autostart.remove_autostart_on_uninstall",
        _cleanup,
    )
    d = tmp_path / "fake-sieve-dir"
    d.mkdir()
    (d / "foo").write_text("bar")
    cli_uninstall.wipe_sieve_dir(d)
    assert called == [True]
    assert not d.exists()

"""Tests for the sandboxed-proxy infrastructure.

The sandbox wraps a full uvicorn child + encrypted store, so these
tests mock the expensive parts (subprocess, httpx health probe) and
exercise the scaffolding: config cloning, free-port picking, orphan
detection, lifecycle cleanup, signal handler registration.

End-to-end "sandbox really runs a benchmark" coverage lives in the
BattleTest verification step, not here — those tests would need a
working LLM endpoint to be meaningful.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from unittest import mock

import pytest
import yaml

from sieve import _sandbox
from sieve._sandbox import (
    SANDBOX_PREFIX,
    SandboxedProxy,
    _pick_free_port,
    _pid_alive,
    _write_sandbox_yaml,
    sweep_orphan_sandboxes,
)
from sieve.config import RecallConfig


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_sandbox_root(tmp_path, monkeypatch):
    """Point SANDBOX_ROOT at a tmpdir so tests never touch ~/.cache/sieve."""
    root = tmp_path / "sandbox_root"
    monkeypatch.setattr(_sandbox, "SANDBOX_ROOT", root)
    return root


@pytest.fixture
def main_config(tmp_path):
    """A minimal RecallConfig with a throwaway store path."""
    yaml_path = tmp_path / "main-sieve.yaml"
    yaml_path.write_text(
        "listen: {host: 127.0.0.1, port: 11435}\n"
        "provider:\n"
        "  type: ollama\n"
        "  base_url: http://127.0.0.1:11434\n"
        "  default_model: qwen3.5:9b\n"
        "embeddings: {provider: fastembed}\n"
        f"store: {{path: {tmp_path / 'main.db'}}}\n"
    )
    return RecallConfig.load(str(yaml_path))


# ── _pick_free_port / _pid_alive ────────────────────────────────────────


def test_pick_free_port_returns_bindable_port():
    port = _pick_free_port()
    assert 1024 <= port <= 65535
    # Should be bindable right now (modulo a race window).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


def test_pid_alive_true_for_self():
    assert _pid_alive(os.getpid())


def test_pid_alive_false_for_missing_pid():
    # PIDs above 2**22 don't exist on Linux.
    assert not _pid_alive(2**30)


# ── _write_sandbox_yaml — config cloning ────────────────────────────────


def test_write_sandbox_yaml_preserves_provider_and_overrides_port(main_config, tmp_path):
    """The sandbox YAML keeps provider/model settings but overrides listen + store + auth."""
    target = tmp_path / "sandbox.yaml"
    store = tmp_path / "sb_memory.db"
    _write_sandbox_yaml(
        main_config=main_config,
        target_path=target,
        store_path=store,
        port=54321,
        auth_token_file=tmp_path / "tok",
        key_file=tmp_path / "key",
    )
    doc = yaml.safe_load(target.read_text())
    assert doc["listen"]["port"] == 54321
    assert doc["store"]["path"] == str(store)
    # Provider settings inherited.
    assert doc["provider"]["base_url"] == "http://127.0.0.1:11434"
    assert doc["provider"]["default_model"] == "qwen3.5:9b"
    # Auth disabled — sandbox is localhost only.
    assert doc["security"]["auth_token"] is None
    # FastEmbed inherited.
    assert doc["embeddings"]["provider"] == "fastembed"


def test_write_sandbox_yaml_reloads_cleanly(main_config, tmp_path):
    """The emitted YAML must round-trip through RecallConfig.load without errors.

    Guards against missing required fields and misnamed keys.
    """
    target = tmp_path / "sandbox.yaml"
    _write_sandbox_yaml(
        main_config=main_config,
        target_path=target,
        store_path=tmp_path / "sb.db",
        port=54321,
        auth_token_file=tmp_path / "tok",
        key_file=tmp_path / "key",
    )
    reloaded = RecallConfig.load(str(target))
    assert reloaded.listen.port == 54321
    assert reloaded.provider.base_url == "http://127.0.0.1:11434"
    assert reloaded.provider.default_model == "qwen3.5:9b"
    assert reloaded.store.path == str(tmp_path / "sb.db")


# ── Orphan sweeper ──────────────────────────────────────────────────────


def test_sweep_removes_sandbox_with_dead_pidfile(isolated_sandbox_root):
    root = isolated_sandbox_root
    root.mkdir(parents=True)
    sb = root / f"{SANDBOX_PREFIX}dead"
    sb.mkdir()
    (sb / "proxy.pid").write_text(str(2**30))  # definitely dead
    removed = sweep_orphan_sandboxes(root=root)
    assert sb in removed
    assert not sb.exists()


def test_sweep_spares_sandbox_with_live_pidfile(isolated_sandbox_root):
    root = isolated_sandbox_root
    root.mkdir(parents=True)
    sb = root / f"{SANDBOX_PREFIX}live"
    sb.mkdir()
    (sb / "proxy.pid").write_text(str(os.getpid()))  # this process is live
    removed = sweep_orphan_sandboxes(root=root)
    assert sb not in removed
    assert sb.exists()


def test_sweep_spares_recent_sandbox_without_pidfile(isolated_sandbox_root, monkeypatch):
    """A sandbox that hasn't written its pidfile yet (startup race) is spared."""
    root = isolated_sandbox_root
    root.mkdir(parents=True)
    sb = root / f"{SANDBOX_PREFIX}recent"
    sb.mkdir()
    # No pidfile, but mtime is now — we shouldn't clobber it.
    removed = sweep_orphan_sandboxes(root=root)
    assert sb not in removed
    assert sb.exists()


def test_sweep_removes_old_sandbox_without_pidfile(isolated_sandbox_root):
    """A sandbox dir older than the orphan age, with no pidfile, is cleaned."""
    root = isolated_sandbox_root
    root.mkdir(parents=True)
    sb = root / f"{SANDBOX_PREFIX}stale"
    sb.mkdir()
    # Backdate mtime past the orphan age.
    ancient = time.time() - (_sandbox._ORPHAN_AGE_S + 60)
    os.utime(sb, (ancient, ancient))
    removed = sweep_orphan_sandboxes(root=root)
    assert sb in removed
    assert not sb.exists()


def test_sweep_ignores_unrelated_directories(isolated_sandbox_root):
    """Only directories matching SANDBOX_PREFIX are touched."""
    root = isolated_sandbox_root
    root.mkdir(parents=True)
    (root / "unrelated-thing").mkdir()
    (root / "something-else").write_text("x")
    removed = sweep_orphan_sandboxes(root=root)
    assert removed == []
    assert (root / "unrelated-thing").exists()
    assert (root / "something-else").exists()


def test_sweep_handles_missing_root(tmp_path):
    """Sweeping a non-existent root is a no-op, not an error."""
    nope = tmp_path / "does-not-exist"
    removed = sweep_orphan_sandboxes(root=nope)
    assert removed == []


# ── Full SandboxedProxy lifecycle (subprocess mocked) ──────────────────


class _FakeProc:
    """Stand-in for subprocess.Popen used by the sandbox."""

    def __init__(self, pid: int = 99999, exit_delay_s: float | None = None):
        self.pid = pid
        self.returncode: int | None = None
        self._exit_delay_s = exit_delay_s
        self._terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self._terminated = True
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Replace subprocess.Popen with a fake that never really spawns."""
    calls: list = []

    def _popen(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return _FakeProc()

    monkeypatch.setattr(_sandbox.subprocess, "Popen", _popen)
    return calls


@pytest.fixture
def fake_healthy(monkeypatch):
    """_wait_for_health returns immediately without hitting HTTP."""
    monkeypatch.setattr(_sandbox, "_wait_for_health", lambda *a, **kw: None)


@pytest.fixture
def fake_store_init(monkeypatch):
    """_init_sandbox_store becomes a no-op (no real sqlcipher)."""
    monkeypatch.setattr(_sandbox, "_init_sandbox_store", lambda cfg_path: None)


def test_sandbox_creates_and_removes_directory(
    main_config, isolated_sandbox_root, fake_subprocess, fake_healthy, fake_store_init,
):
    assert not isolated_sandbox_root.exists() or not any(isolated_sandbox_root.iterdir())
    with SandboxedProxy.from_main_config(main_config) as sb:
        assert sb.directory.exists()
        assert sb.directory.parent == isolated_sandbox_root
        assert sb.directory.name.startswith(SANDBOX_PREFIX)
        assert (sb.directory / "sieve.yaml").exists()
        assert (sb.directory / ".sieve_key").exists()
        assert (sb.directory / "proxy.pid").exists()
    # Directory removed on exit.
    assert not sb.directory.exists()


def test_sandbox_health_failure_propagates_and_cleans_up(
    main_config, isolated_sandbox_root, fake_subprocess, fake_store_init, monkeypatch,
):
    def _boom(*a, **kw):
        raise TimeoutError("simulated health failure")
    monkeypatch.setattr(_sandbox, "_wait_for_health", _boom)

    with pytest.raises(TimeoutError):
        with SandboxedProxy.from_main_config(main_config):
            pytest.fail("should not reach body")
    # Even though __enter__ raised, the sandbox dir should be gone.
    children = list(isolated_sandbox_root.iterdir()) if isolated_sandbox_root.exists() else []
    assert children == [], f"orphan sandbox left behind: {children}"


def test_sandbox_passes_config_env_to_child(
    main_config, isolated_sandbox_root, fake_subprocess, fake_healthy, fake_store_init,
):
    with SandboxedProxy.from_main_config(main_config) as sb:
        call = fake_subprocess[-1]
        # The config path must match the sandbox's yaml.
        env = call["kwargs"]["env"]
        assert env["SIEVE_CONFIG"] == str(sb.config_path)
        # start_new_session is critical so Ctrl-C doesn't nuke the child
        # before our cleanup runs.
        assert call["kwargs"]["start_new_session"] is True


def test_sandbox_handle_exposes_main_provider_url(
    main_config, isolated_sandbox_root, fake_subprocess, fake_healthy, fake_store_init,
):
    """Handle exposes the main provider URL so graders can bypass the sandbox proxy."""
    with SandboxedProxy.from_main_config(main_config) as sb:
        assert sb.provider_base_url == "http://127.0.0.1:11434"
        # And the sandbox's own base URL is different (it's a sandbox port).
        assert sb.base_url != sb.provider_base_url


def test_sandbox_teardown_is_idempotent(
    main_config, isolated_sandbox_root, fake_subprocess, fake_healthy, fake_store_init,
):
    sb_ref = SandboxedProxy.from_main_config(main_config)
    handle = sb_ref.__enter__()
    assert handle.directory.exists()
    sb_ref._teardown()
    sb_ref._teardown()  # second call should be a no-op, not a crash
    assert not handle.directory.exists()


def test_sandbox_restores_prior_signal_handlers(
    main_config, isolated_sandbox_root, fake_subprocess, fake_healthy, fake_store_init,
):
    """Signal handlers installed by the sandbox are restored on exit."""
    sentinel = lambda *a: None
    prior = signal.signal(signal.SIGINT, sentinel)
    try:
        with SandboxedProxy.from_main_config(main_config):
            current = signal.getsignal(signal.SIGINT)
            assert current is not sentinel  # sandbox installed its own
        # After exit, our sentinel should be back.
        assert signal.getsignal(signal.SIGINT) is sentinel
    finally:
        signal.signal(signal.SIGINT, prior)

"""Per-port PID file behavior — required for parallel daemons.

Verifies that two daemons listening on different ports get different
PID file paths, while the default port keeps the legacy ~/.sieve/sieve.pid
name (backward compatibility).
"""
from __future__ import annotations

from sieve.cli import PID_FILE, SIEVE_DIR, _pid_file_for_port, _DEFAULT_PORT


def test_default_port_returns_legacy_pid_file():
    # Existing installs that use the default port must keep the
    # ~/.sieve/sieve.pid path so upgrades don't orphan running daemons.
    assert _pid_file_for_port(_DEFAULT_PORT) == PID_FILE


def test_non_default_port_uses_suffixed_pid_file():
    p = _pid_file_for_port(11436)
    assert p == SIEVE_DIR / "sieve-11436.pid"


def test_two_different_ports_get_different_pid_files():
    # The whole point: parallel daemons on different ports never
    # collide on the PID file.
    assert _pid_file_for_port(11436) != _pid_file_for_port(11437)
    assert _pid_file_for_port(11436) != _pid_file_for_port(_DEFAULT_PORT)


def test_default_port_constant_matches_config_default():
    # _DEFAULT_PORT must stay in sync with config.RecallConfig.listen.port
    # default — otherwise upgrades silently switch which file the daemon writes.
    from sieve.config import RecallConfig
    cfg = RecallConfig.load() if False else None  # avoid filesystem dep
    # Read the dataclass default directly:
    from sieve.config import ListenConfig
    assert ListenConfig().port == _DEFAULT_PORT

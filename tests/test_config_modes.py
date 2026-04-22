"""Tests for SIEVE_MODE detection + mode-aware YAML loader."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest


def test_mode_defaults_to_production_when_env_unset(monkeypatch):
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    from sieve.config_modes import current_mode, Mode
    assert current_mode() == Mode.PRODUCTION


def test_mode_defaults_to_production_when_env_empty(monkeypatch):
    monkeypatch.setenv("SIEVE_MODE", "")
    from sieve.config_modes import current_mode, Mode
    assert current_mode() == Mode.PRODUCTION


def test_mode_test_when_env_set(monkeypatch):
    monkeypatch.setenv("SIEVE_MODE", "test")
    from sieve.config_modes import current_mode, Mode
    assert current_mode() == Mode.TEST


def test_mode_test_case_insensitive(monkeypatch):
    monkeypatch.setenv("SIEVE_MODE", "TEST")
    from sieve.config_modes import current_mode, Mode
    assert current_mode() == Mode.TEST


def test_mode_invalid_raises(monkeypatch):
    monkeypatch.setenv("SIEVE_MODE", "staging")
    from sieve.config_modes import current_mode
    with pytest.raises(ValueError, match="SIEVE_MODE"):
        current_mode()


def test_production_mode_accepts_production_key(tmp_path, monkeypatch):
    """writer.model is in _SETTABLE (production surface); must load fine."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  model: qwen3:14b\n")
    from sieve.config_modes import load_config_for_mode
    raw = load_config_for_mode(yaml_path=yaml_path)
    assert raw == {"writer": {"model": "qwen3:14b"}}


def test_production_mode_rejects_advanced_key(tmp_path, monkeypatch):
    """retrieval.mmr_lambda is advanced-only; loading in production must fail."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("retrieval:\n  mmr_lambda: 0.9\n")
    from sieve.config_modes import load_config_for_mode, ProductionKeyViolation
    with pytest.raises(ProductionKeyViolation) as exc:
        load_config_for_mode(yaml_path=yaml_path)
    assert "retrieval.mmr_lambda" in str(exc.value)
    assert "SIEVE_MODE=test" in str(exc.value)


def test_production_mode_rejects_unknown_key(tmp_path, monkeypatch):
    """Typo-ed keys not in either surface are rejected in production."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  num_ctxxx: 8192\n")
    from sieve.config_modes import load_config_for_mode, ProductionKeyViolation
    with pytest.raises(ProductionKeyViolation) as exc:
        load_config_for_mode(yaml_path=yaml_path)
    assert "writer.num_ctxxx" in str(exc.value)


def test_test_mode_accepts_advanced_key_via_test_yaml(tmp_path, monkeypatch, caplog):
    """SIEVE_MODE=test + advanced key in sieve.test.yaml must load + warn."""
    monkeypatch.setenv("SIEVE_MODE", "test")
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  model: qwen3:14b\n")
    test_yaml = tmp_path / "sieve.test.yaml"
    test_yaml.write_text("retrieval:\n  mmr_lambda: 0.9\n")
    from sieve.config_modes import load_config_for_mode
    with caplog.at_level(logging.WARNING):
        raw = load_config_for_mode(yaml_path=yaml_path, test_yaml_path=test_yaml)
    # Merged raw dict contains both.
    assert raw["writer"]["model"] == "qwen3:14b"
    assert raw["retrieval"]["mmr_lambda"] == 0.9
    # Warning mentions the active advanced override.
    assert any("retrieval.mmr_lambda" in r.message for r in caplog.records)


def test_test_mode_rejects_unknown_key_even_in_test_yaml(tmp_path, monkeypatch):
    """Typo-ed key is rejected in BOTH modes (fail-loud beats silent-drop)."""
    monkeypatch.setenv("SIEVE_MODE", "test")
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("")
    test_yaml = tmp_path / "sieve.test.yaml"
    test_yaml.write_text("retrieval:\n  mmr_lambdaaa: 0.9\n")
    from sieve.config_modes import load_config_for_mode, ProductionKeyViolation
    with pytest.raises(ProductionKeyViolation) as exc:
        load_config_for_mode(yaml_path=yaml_path, test_yaml_path=test_yaml)
    assert "retrieval.mmr_lambdaaa" in str(exc.value)


def test_test_mode_test_yaml_overrides_base_yaml(tmp_path, monkeypatch):
    """When both files set a production key, test yaml wins."""
    monkeypatch.setenv("SIEVE_MODE", "test")
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  model: qwen3:14b\n")
    test_yaml = tmp_path / "sieve.test.yaml"
    test_yaml.write_text("writer:\n  model: qwen3:4b\n")
    from sieve.config_modes import load_config_for_mode
    raw = load_config_for_mode(yaml_path=yaml_path, test_yaml_path=test_yaml)
    assert raw["writer"]["model"] == "qwen3:4b"


def test_missing_yaml_file_returns_empty_dict(tmp_path, monkeypatch):
    """No sieve.yaml at the given path = empty config, no error."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    yaml_path = tmp_path / "nonexistent.yaml"
    from sieve.config_modes import load_config_for_mode
    raw = load_config_for_mode(yaml_path=yaml_path)
    assert raw == {}


def test_none_yaml_path_returns_empty_dict(monkeypatch):
    """None yaml_path = empty config, no error."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    from sieve.config_modes import load_config_for_mode
    raw = load_config_for_mode(yaml_path=None)
    assert raw == {}


def test_test_mode_no_test_yaml_still_works(tmp_path, monkeypatch):
    """SIEVE_MODE=test with no sieve.test.yaml file works; just uses base yaml."""
    monkeypatch.setenv("SIEVE_MODE", "test")
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  model: qwen3:14b\n")
    from sieve.config_modes import load_config_for_mode
    raw = load_config_for_mode(
        yaml_path=yaml_path,
        test_yaml_path=tmp_path / "nonexistent_test.yaml",
    )
    assert raw == {"writer": {"model": "qwen3:14b"}}


def test_log_config_drift_logs_known_override(caplog):
    """A production-key override vs dataclass default emits CONFIG_DRIFT."""
    from sieve.config import RecallConfig
    from sieve.config_modes import log_config_drift
    import logging

    config = RecallConfig()
    config.writer.model = "qwen3.5:4b"  # dataclass default — should NOT drift
    config.provider.default_model = "llama3.2:3b"  # override — should drift

    with caplog.at_level(logging.INFO, logger="recall.config_modes"):
        count = log_config_drift(config)

    assert count >= 1
    drift_lines = [r.message for r in caplog.records if "CONFIG_DRIFT" in r.message]
    assert any("provider.default_model" in line for line in drift_lines)
    assert any("llama3.2:3b" in line for line in drift_lines)


def test_log_config_drift_silent_on_defaults(caplog):
    """A config matching dataclass defaults emits zero drift lines."""
    from sieve.config import RecallConfig
    from sieve.config_modes import log_config_drift
    import logging

    config = RecallConfig()  # pure defaults

    with caplog.at_level(logging.INFO, logger="recall.config_modes"):
        count = log_config_drift(config)

    drift_lines = [r.message for r in caplog.records if "CONFIG_DRIFT" in r.message]
    assert count == 0, f"expected 0 drift lines on defaults, got: {drift_lines}"


def test_log_config_drift_allowlist_suppresses_host_specific(caplog):
    """Allowlisted keys (store.path, security.auth_token) don't produce drift."""
    from sieve.config import RecallConfig
    from sieve.config_modes import log_config_drift
    import logging

    config = RecallConfig()
    config.store.path = "/tmp/custom-path.db"  # host-specific, allowlisted
    config.security.auth_token = "some-random-token"  # per-install, allowlisted

    with caplog.at_level(logging.INFO, logger="recall.config_modes"):
        count = log_config_drift(config)

    drift_lines = [r.message for r in caplog.records if "CONFIG_DRIFT" in r.message]
    assert not any("store.path" in line for line in drift_lines)
    assert not any("security.auth_token" in line for line in drift_lines)


def test_log_config_drift_exception_does_not_crash_callers(caplog, monkeypatch):
    """Regression guard: create_app wraps log_config_drift in try/except so
    a property that raises cannot make sieve unbootable. This test verifies
    log_config_drift itself will surface errors (callers handle them) — the
    create_app wrapping is tested implicitly by the full suite not crashing."""
    from sieve.config import RecallConfig
    from sieve.config_modes import log_config_drift

    config = RecallConfig()
    # Replace _get_dotted to raise on the first call, simulating a schema regression.
    import sieve.config_modes as cm
    original = cm._get_dotted

    calls = {"n": 0}

    def exploding(obj, key):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated schema regression")
        return original(obj, key)

    monkeypatch.setattr(cm, "_get_dotted", exploding)

    # log_config_drift surfaces the exception; the wrapper in main.py catches it.
    import pytest
    with pytest.raises(RuntimeError, match="simulated schema regression"):
        log_config_drift(config)

"""Verify that RecallConfig.load() goes through the mode-aware loader.

The goal: in production mode (SIEVE_MODE unset), loading a sieve.yaml
with an advanced-only key must raise ProductionKeyViolation.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from sieve.config import RecallConfig
from sieve.config_modes import ProductionKeyViolation


def test_recall_config_load_rejects_advanced_key_in_production(
    tmp_path, monkeypatch
):
    """Production-mode RecallConfig.load() on a YAML with an advanced-only
    key must raise ProductionKeyViolation."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("retrieval:\n  mmr_lambda: 0.9\n")
    monkeypatch.setenv("SIEVE_CONFIG", str(yaml_path))

    with pytest.raises(ProductionKeyViolation) as exc:
        RecallConfig.load()
    assert "retrieval.mmr_lambda" in str(exc.value)


def test_recall_config_load_accepts_production_key_in_production(
    tmp_path, monkeypatch
):
    """A production-surface key loads cleanly."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  model: qwen3:14b\n")
    monkeypatch.setenv("SIEVE_CONFIG", str(yaml_path))

    c = RecallConfig.load()
    assert c.writer.model == "qwen3:14b"


def test_recall_config_load_test_mode_accepts_advanced(
    tmp_path, monkeypatch
):
    """Under SIEVE_MODE=test + sieve.test.yaml overlay, advanced keys load."""
    monkeypatch.setenv("SIEVE_MODE", "test")
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text("writer:\n  model: qwen3:14b\n")
    test_yaml = tmp_path / "sieve.test.yaml"
    test_yaml.write_text("retrieval:\n  mmr_lambda: 0.9\n")
    monkeypatch.setenv("SIEVE_CONFIG", str(yaml_path))
    monkeypatch.setenv("SIEVE_TEST_CONFIG", str(test_yaml))

    c = RecallConfig.load()
    # Production key survives the merge.
    assert c.writer.model == "qwen3:14b"
    # Advanced key is accepted (we don't assert how it's plumbed into
    # RecallConfig; Task 2.3 scope only needs the YAML to load without
    # raising. Later tasks may expose mmr_lambda via a typed field).


def test_recall_config_load_empty_when_no_yaml(tmp_path, monkeypatch):
    """With no YAML at all, defaults apply, no error."""
    monkeypatch.delenv("SIEVE_MODE", raising=False)
    # Point SIEVE_CONFIG at a nonexistent path in tmp.
    monkeypatch.setenv("SIEVE_CONFIG", str(tmp_path / "nonexistent.yaml"))

    c = RecallConfig.load()
    # Ship-safe defaults kick in.
    assert c.listen.host == "127.0.0.1"

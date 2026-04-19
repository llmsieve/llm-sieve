"""Tests for `sieve config show/set/reset/edit`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


# --- Pure helpers ---

class TestSetPath:
    def test_sets_nested_key(self):
        from sieve.cli_config import set_path
        data: dict = {"provider": {"base_url": "old"}}
        set_path(data, "provider.base_url", "new")
        assert data == {"provider": {"base_url": "new"}}

    def test_creates_missing_sections(self):
        from sieve.cli_config import set_path
        data: dict = {}
        set_path(data, "listen.port", 12345)
        assert data == {"listen": {"port": 12345}}

    def test_coerces_port_to_int(self):
        from sieve.cli_config import set_path
        data: dict = {}
        set_path(data, "listen.port", "11500")
        assert data["listen"]["port"] == 11500
        assert isinstance(data["listen"]["port"], int)

    def test_coerces_bool_strings(self):
        from sieve.cli_config import set_path
        data: dict = {}
        set_path(data, "ablation.absence_signal", "false")
        assert data["ablation"]["absence_signal"] is False

    def test_rejects_invalid_enum(self):
        from sieve.cli_config import set_path
        data: dict = {}
        with pytest.raises(ValueError, match="context_format"):
            set_path(data, "pipeline.context_format", "nonsense")

    def test_rejects_unknown_path(self):
        from sieve.cli_config import set_path
        data: dict = {}
        with pytest.raises(ValueError, match="unknown config path"):
            set_path(data, "does.not.exist", "x")


class TestDiffFromDefaults:
    def test_flags_non_default_values(self):
        from sieve.cli_config import diff_from_defaults
        from sieve.config import RecallConfig
        cfg = RecallConfig()
        cfg.listen.port = 99999
        diffs = diff_from_defaults(cfg)
        assert ("listen.port", 99999, 11435) in diffs

    def test_empty_when_all_defaults(self):
        from sieve.cli_config import diff_from_defaults
        from sieve.config import RecallConfig
        assert diff_from_defaults(RecallConfig()) == []


# --- CLI surface ---

@pytest.fixture
def sieve_cfg(tmp_path, monkeypatch):
    """Point the config-management commands at a tmp sieve.yaml."""
    d = tmp_path / ".sieve"
    d.mkdir()
    cfg = d / "sieve.yaml"
    cfg.write_text(yaml.safe_dump({
        "listen": {"host": "127.0.0.1", "port": 11435},
        "provider": {"base_url": "http://127.0.0.1:11434", "default_model": "qwen3.5:9b"},
        "store": {"path": str(d / "memory.db")},
    }))
    monkeypatch.setattr("sieve.cli_config.SIEVE_CFG", cfg)
    return cfg


def test_config_show_runs(sieve_cfg):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "listen" in result.output
    assert "11435" in result.output


def test_config_show_highlights_non_default(sieve_cfg):
    sieve_cfg.write_text(yaml.safe_dump({
        "listen": {"host": "127.0.0.1", "port": 22222},
        "provider": {"base_url": "http://127.0.0.1:11434", "default_model": "qwen3.5:9b"},
        "store": {"path": str(sieve_cfg.parent / "memory.db")},
    }))
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["config", "show"])
    assert "22222" in result.output
    # non-default values are tagged (we emit a "(non-default)" marker to be
    # robust across terminal widths)
    assert "non-default" in result.output.lower() or "•" in result.output


def test_config_set_updates_yaml(sieve_cfg):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli, ["config", "set", "provider.default_model", "qwen3.5:27b"]
    )
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(sieve_cfg.read_text())
    assert data["provider"]["default_model"] == "qwen3.5:27b"


def test_config_set_invalid_key_errors(sieve_cfg):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli, ["config", "set", "pipeline.context_format", "nonsense"]
    )
    assert result.exit_code != 0
    assert "context_format" in result.output.lower()


def test_config_set_unknown_path_errors(sieve_cfg):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli, ["config", "set", "does.not.exist", "x"]
    )
    assert result.exit_code != 0


def test_config_reset_with_confirmation(sieve_cfg):
    from sieve import cli as cli_mod
    # Seed with non-default
    sieve_cfg.write_text(yaml.safe_dump({
        "listen": {"port": 22222},
        "provider": {"base_url": "http://127.0.0.1:11434"},
    }))
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["config", "reset"], input="y\n")
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(sieve_cfg.read_text())
    # After reset, port should be back to the default 11435
    assert data["listen"]["port"] == 11435


def test_config_reset_preserves_provider_url(sieve_cfg):
    """Reset should keep whatever provider URL the user set — people don't
    want to re-configure Ollama location after a reset."""
    from sieve import cli as cli_mod
    sieve_cfg.write_text(yaml.safe_dump({
        "listen": {"port": 22222},
        "provider": {"base_url": "http://192.168.1.50:11434",
                     "default_model": "qwen3.5:9b"},
    }))
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["config", "reset"], input="y\n")
    assert result.exit_code == 0
    data = yaml.safe_load(sieve_cfg.read_text())
    assert data["provider"]["base_url"] == "http://192.168.1.50:11434"


def test_config_reset_declined_leaves_config_alone(sieve_cfg):
    from sieve import cli as cli_mod
    before = sieve_cfg.read_text()
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["config", "reset"], input="n\n")
    assert result.exit_code != 0  # click.confirm abort returns non-zero
    assert sieve_cfg.read_text() == before


def test_config_edit_invokes_editor(sieve_cfg, monkeypatch):
    """`config edit` should defer to click.edit() and then validate."""
    from sieve import cli as cli_mod

    called = {"edited": False}

    def fake_edit(text=None, filename=None, require_save=True, extension=".yaml"):
        called["edited"] = True
        # Simulate user writing a valid edit
        if filename:
            Path(filename).write_text(
                yaml.safe_dump({"listen": {"port": 33333}})
            )
        return None

    monkeypatch.setattr("click.edit", fake_edit)
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["config", "edit"])
    assert result.exit_code == 0, result.output
    assert called["edited"] is True
    assert yaml.safe_load(sieve_cfg.read_text())["listen"]["port"] == 33333


def test_config_edit_rejects_invalid_yaml(sieve_cfg, monkeypatch):
    from sieve import cli as cli_mod

    def fake_edit(text=None, filename=None, require_save=True, extension=".yaml"):
        if filename:
            Path(filename).write_text("this is: [not valid yaml")
        return None

    monkeypatch.setattr("click.edit", fake_edit)
    runner = CliRunner()
    before = sieve_cfg.read_text()
    result = runner.invoke(cli_mod.cli, ["config", "edit"])
    assert result.exit_code != 0
    # Original config preserved on failure
    assert sieve_cfg.read_text() == before

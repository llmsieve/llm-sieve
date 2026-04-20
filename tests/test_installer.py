"""Tests for the sieve-install command.

Covers the contract each helper is supposed to honour. End-to-end
verification against a live endpoint is in the BattleTest D10 step.

These tests are written to survive the kind of real-world drift the
user called out with "run first time every time" — we mock external
things so the suite doesn't depend on network / filesystem / env.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from sieve import _installer


# ── Probe / URL normalisation ─────────────────────────────────────────


def test_normalise_url_bare_host_gets_default_port():
    assert _installer._normalise_url("192.168.1.100", default_port=11434) == "http://192.168.1.100:11434"


def test_normalise_url_https_not_touched():
    assert (
        _installer._normalise_url("https://api.openai.com/v1", default_port=443)
        == "https://api.openai.com/v1"
    )


def test_normalise_url_keeps_explicit_port():
    assert (
        _installer._normalise_url("ollama.lan:12345", default_port=11434)
        == "http://ollama.lan:12345"
    )


def test_normalise_url_preserves_path():
    assert (
        _installer._normalise_url("http://host.example/api/v1", default_port=11434)
        == "http://host.example:11434/api/v1"
    )


def test_reachable_true_on_200_api_tags(monkeypatch):
    def _get(url, **kw):
        if "/api/tags" in url:
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)
    monkeypatch.setattr(httpx, "get", _get)
    assert _installer._reachable("http://ollama.lan:11434") is True


def test_reachable_tries_v1_models_when_tags_404(monkeypatch):
    def _get(url, **kw):
        if "/api/tags" in url:
            return httpx.Response(404)
        if "/v1/models" in url:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)
    monkeypatch.setattr(httpx, "get", _get)
    assert _installer._reachable("https://api.openai.com/v1") is True


def test_reachable_uses_bearer_auth(monkeypatch):
    """Cloud endpoints 401 without auth — we must pass the API key."""
    seen_headers: list[dict] = []
    def _get(url, headers=None, **kw):
        seen_headers.append(headers or {})
        if "/v1/models" in url:
            if (headers or {}).get("Authorization"):
                return httpx.Response(200, json={"data": [{"id": "x"}]})
            return httpx.Response(401)
        return httpx.Response(404)
    monkeypatch.setattr(httpx, "get", _get)
    assert _installer._reachable("https://api.openai.com", api_key="sk-test") is True
    # And confirm the Authorization header was actually sent.
    assert any(
        h.get("Authorization") == "Bearer sk-test" for h in seen_headers
    )


def test_reachable_false_on_connect_error(monkeypatch):
    def _get(url, **kw):
        raise httpx.ConnectError("refused", request=None)
    monkeypatch.setattr(httpx, "get", _get)
    assert _installer._reachable("http://nope") is False


def test_reachable_false_on_timeout(monkeypatch):
    def _get(url, **kw):
        raise httpx.TimeoutException("slow", request=None)
    monkeypatch.setattr(httpx, "get", _get)
    assert _installer._reachable("http://slow") is False


# ── list_models with api_key ──────────────────────────────────────────


def test_list_models_forwards_api_key_to_v1(monkeypatch):
    from sieve import _wizard_helpers
    seen_headers: list[dict] = []
    def _get(url, headers=None, **kw):
        seen_headers.append(headers or {})
        if "/api/tags" in url:
            return httpx.Response(404)
        if "/v1/models" in url and (headers or {}).get("Authorization"):
            return httpx.Response(
                200,
                json={"data": [{"id": "claude-sonnet-4-6"}, {"id": "claude-opus-4-7"}]},
            )
        return httpx.Response(401)
    monkeypatch.setattr(httpx, "get", _get)
    models = _wizard_helpers.list_models(
        "https://api.anthropic.com/v1",
        api_key="sk-ant-test",
    )
    assert "claude-sonnet-4-6" in models
    assert "claude-opus-4-7" in models
    # Authorization header was sent on the v1/models request.
    assert any(
        h.get("Authorization") == "Bearer sk-ant-test" for h in seen_headers
    )


def test_list_models_no_api_key_still_works_for_local(monkeypatch):
    """Ollama doesn't need auth. api_key=None must not break anything."""
    from sieve import _wizard_helpers
    def _get(url, headers=None, **kw):
        if "/api/tags" in url:
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        return httpx.Response(404)
    monkeypatch.setattr(httpx, "get", _get)
    models = _wizard_helpers.list_models("http://localhost:11434")
    assert models == ["qwen3.5:9b"]


# ── Install state detection ───────────────────────────────────────────


def test_is_already_installed_true_with_valid_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "sieve-home"
    yaml_path.mkdir()
    (yaml_path / "sieve.yaml").write_text(
        "listen: {host: 127.0.0.1, port: 11435}\n"
        "provider:\n"
        "  type: ollama\n"
        "  base_url: http://127.0.0.1:11434\n"
        "  default_model: qwen3.5:9b\n"
        "embeddings: {provider: fastembed}\n"
        f"store: {{path: {tmp_path / 'db.sqlite'}}}\n"
    )
    monkeypatch.setattr(_installer, "SIEVE_DIR", yaml_path)
    monkeypatch.setenv("SIEVE_CONFIG", str(yaml_path / "sieve.yaml"))
    assert _installer._is_already_installed() is True


def test_is_already_installed_false_without_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "does-not-exist")
    assert _installer._is_already_installed() is False


def test_is_already_installed_false_with_corrupt_yaml(tmp_path, monkeypatch):
    """A broken yaml means the installer should re-run, not bail out
    thinking the install is fine."""
    d = tmp_path / "sieve-broken"
    d.mkdir()
    (d / "sieve.yaml").write_text("not: valid: yaml: ]]]\n")
    monkeypatch.setattr(_installer, "SIEVE_DIR", d)
    monkeypatch.setenv("SIEVE_CONFIG", str(d / "sieve.yaml"))
    assert _installer._is_already_installed() is False


# ── Default-model fallback ────────────────────────────────────────────


def test_default_model_for_openai_fallback(monkeypatch):
    """When listing fails, cloud URLs get provider-appropriate fallback."""
    from sieve import _wizard_helpers
    monkeypatch.setattr(_wizard_helpers, "list_models", lambda *a, **k: [])
    assert _installer._default_model_for("https://api.openai.com/v1", "sk-x") == "gpt-4o-mini"


def test_default_model_for_anthropic_fallback(monkeypatch):
    from sieve import _wizard_helpers
    monkeypatch.setattr(_wizard_helpers, "list_models", lambda *a, **k: [])
    assert _installer._default_model_for("https://api.anthropic.com/v1", "sk-x").startswith("claude")


def test_default_model_for_local_fallback(monkeypatch):
    from sieve import _wizard_helpers
    monkeypatch.setattr(_wizard_helpers, "list_models", lambda *a, **k: [])
    assert _installer._default_model_for("http://127.0.0.1:11434", None) == "qwen3.5:9b"


def test_default_model_for_picks_first_chat_model(monkeypatch):
    """Don't accidentally pick an embedding model as the chat default."""
    from sieve import _wizard_helpers
    monkeypatch.setattr(
        _wizard_helpers,
        "list_models",
        lambda *a, **k: ["nomic-embed-text:latest", "qwen3.5:9b", "gemma:2b"],
    )
    assert _installer._default_model_for("http://localhost:11434", None) == "qwen3.5:9b"


# ── Plan preview / redaction ─────────────────────────────────────────


def test_installplan_redacted_hides_api_key():
    p = _installer.InstallPlan(
        provider_url="https://api.openai.com/v1",
        provider_api_key="sk-super-secret-12345",
        model="gpt-4o",
        autostart=True,
        start_now=True,
    )
    r = p.redacted()
    assert r.provider_api_key == "…"
    # Original unchanged.
    assert p.provider_api_key == "sk-super-secret-12345"


def test_installplan_redacted_preserves_none_api_key():
    p = _installer.InstallPlan(
        provider_url="http://127.0.0.1:11434",
        provider_api_key=None,
        model="qwen3.5:9b",
        autostart=False,
        start_now=True,
    )
    assert p.redacted().provider_api_key is None


# ── Autostart default logic ──────────────────────────────────────────


def test_pick_autostart_defaults_yes_for_desktop(monkeypatch):
    """Non-root user on a supported host → default Yes."""
    from sieve import _autostart
    monkeypatch.setattr(_autostart, "autostart_supported", lambda: True)
    monkeypatch.setenv("USER", "alice")
    captured: dict = {}
    import click
    def _confirm(q, default=False):
        captured["default"] = default
        return default
    monkeypatch.setattr(click, "confirm", _confirm)

    from io import StringIO
    from rich.console import Console
    console = Console(file=StringIO(), width=100)
    result = _installer._pick_autostart(console, no_input=False)
    assert captured["default"] is True
    assert result is True


def test_pick_autostart_defaults_no_for_root(monkeypatch):
    """Root user (server / container) → default No."""
    from sieve import _autostart
    monkeypatch.setattr(_autostart, "autostart_supported", lambda: True)
    monkeypatch.setenv("USER", "root")
    captured: dict = {}
    import click
    def _confirm(q, default=False):
        captured["default"] = default
        return default
    monkeypatch.setattr(click, "confirm", _confirm)

    from io import StringIO
    from rich.console import Console
    console = Console(file=StringIO(), width=100)
    result = _installer._pick_autostart(console, no_input=False)
    assert captured["default"] is False
    assert result is False


def test_pick_autostart_skipped_when_unsupported(monkeypatch):
    """Non-Linux / no systemd / no sieve binary → skip silently, return False."""
    from sieve import _autostart
    monkeypatch.setattr(_autostart, "autostart_supported", lambda: False)
    import click
    # confirm must NOT be called in this branch.
    called = {"confirm": False}
    def _confirm(*a, **k):
        called["confirm"] = True
        return True
    monkeypatch.setattr(click, "confirm", _confirm)
    from io import StringIO
    from rich.console import Console
    console = Console(file=StringIO(), width=100)
    result = _installer._pick_autostart(console, no_input=False)
    assert called["confirm"] is False
    assert result is False


def test_pick_autostart_no_input_returns_false(monkeypatch):
    """--no-input is used in CI / scripts; autostart defaults to off."""
    from sieve import _autostart
    monkeypatch.setattr(_autostart, "autostart_supported", lambda: True)
    from io import StringIO
    from rich.console import Console
    console = Console(file=StringIO(), width=100)
    assert _installer._pick_autostart(console, no_input=True) is False


# ── Config write + rollback ──────────────────────────────────────────


def test_write_yaml_without_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "sieve-out")
    plan = _installer.InstallPlan(
        provider_url="http://127.0.0.1:11434",
        provider_api_key=None,
        model="qwen3.5:9b",
        autostart=False,
        start_now=False,
    )
    _installer._write_yaml(plan)
    text = (tmp_path / "sieve-out" / "sieve.yaml").read_text()
    assert "base_url: http://127.0.0.1:11434" in text
    assert "default_model: qwen3.5:9b" in text
    assert "api_key" not in text  # Not written for local/LAN


def test_write_yaml_includes_api_key_for_cloud(tmp_path, monkeypatch):
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "sieve-out")
    plan = _installer.InstallPlan(
        provider_url="https://api.anthropic.com/v1",
        provider_api_key="sk-ant-abc123",
        model="claude-sonnet-4-6",
        autostart=False,
        start_now=False,
    )
    _installer._write_yaml(plan)
    text = (tmp_path / "sieve-out" / "sieve.yaml").read_text()
    assert "api_key: sk-ant-abc123" in text


def test_remove_yaml_rolls_back_write(tmp_path, monkeypatch):
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "rb")
    plan = _installer.InstallPlan(
        provider_url="http://x", provider_api_key=None, model="m",
        autostart=False, start_now=False,
    )
    _installer._write_yaml(plan)
    assert (tmp_path / "rb" / "sieve.yaml").exists()
    _installer._remove_yaml()
    assert not (tmp_path / "rb" / "sieve.yaml").exists()


def test_remove_yaml_idempotent_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "nope")
    # Must not raise even though the file isn't there.
    _installer._remove_yaml()


# ── Entry point: already-installed branch ────────────────────────────


# ── D9: Resilience — "first time every time" ─────────────────────────


def test_init_store_idempotent_on_existing_fresh_store(tmp_path, monkeypatch):
    """Re-running sieve-install with a valid store in place: must not
    re-initialise or lose data. Partial-install recovery path."""
    # Build a minimal yaml pointing at a store inside tmp_path, then
    # call _init_store() twice.
    yaml_path = tmp_path / ".sieve"
    yaml_path.mkdir()
    (yaml_path / "sieve.yaml").write_text(
        "listen: {host: 127.0.0.1, port: 11435}\n"
        "provider:\n"
        "  type: ollama\n"
        "  base_url: http://127.0.0.1:11434\n"
        "  default_model: qwen3.5:9b\n"
        "embeddings: {provider: fastembed}\n"
        f"store: {{path: {yaml_path / 'memory.db'}}}\n"
    )
    monkeypatch.setenv("SIEVE_CONFIG", str(yaml_path / "sieve.yaml"))

    # First init creates the store.
    _installer._init_store()
    db_path = yaml_path / "memory.db"
    assert db_path.exists()
    first_mtime = db_path.stat().st_mtime
    first_size = db_path.stat().st_size

    # Second init must not trash it.
    _installer._init_store()
    # The store file should still exist with at least the original size.
    assert db_path.exists()
    assert db_path.stat().st_size >= first_size


def test_install_flow_exits_cleanly_on_keyboard_interrupt(tmp_path, monkeypatch):
    """Ctrl-C at any prompt must: (a) exit with code 130, (b) roll
    back any config written, (c) print a retry hint."""
    # Force a KeyboardInterrupt at the first real decision: the
    # 'Use the local Ollama?' confirm.
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "fresh")
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    import click
    def _confirm_raises(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(click, "confirm", _confirm_raises)
    # Probe returns True so we reach the confirm.
    monkeypatch.setattr(_installer, "_reachable", lambda *a, **k: True)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(_installer.main, [])
    assert result.exit_code == 130
    # No yaml left behind.
    assert not (tmp_path / "fresh" / "sieve.yaml").exists()


def test_install_flow_surfaces_fastembed_failure(tmp_path, monkeypatch):
    """If FastEmbed fails, the installer surfaces the actual cause
    and rolls back the config write — it doesn't leave a broken yaml."""
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "embed-fail")
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    monkeypatch.setattr(_installer, "_reachable", lambda *a, **k: True)
    # Make the FastEmbed step explode.
    import fastembed
    class _Broken:
        def __init__(self, *a, **kw):
            raise RuntimeError("onnxruntime CPU ISA mismatch")
    monkeypatch.setattr(fastembed, "TextEmbedding", _Broken)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(_installer.main, [
        "--no-input",
        "--provider", "http://127.0.0.1:11434",
        "--model", "qwen3.5:9b",
    ])
    assert result.exit_code == 1
    assert "FastEmbed setup failed" in result.output
    assert "onnxruntime CPU ISA mismatch" in result.output
    # Rollback fired — no yaml left behind.
    assert not (tmp_path / "embed-fail" / "sieve.yaml").exists()


def test_install_flow_no_input_fails_fast_when_default_unreachable(tmp_path, monkeypatch):
    """--no-input with no --provider and local Ollama absent: clear
    error, no config written, exit 1."""
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "fail-fast")
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    monkeypatch.setattr(_installer, "_reachable", lambda *a, **k: False)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(_installer.main, ["--no-input"])
    assert result.exit_code == 1
    assert "--no-input" in result.output
    assert "default Ollama" in result.output
    assert not (tmp_path / "fail-fast" / "sieve.yaml").exists()


def test_install_flow_no_input_fails_fast_when_provider_override_unreachable(
    tmp_path, monkeypatch,
):
    """--provider with an unreachable URL: fail fast so scripted
    installs don't silently write broken configs."""
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "bad-provider")
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    monkeypatch.setattr(_installer, "_reachable", lambda *a, **k: False)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(_installer.main, [
        "--no-input",
        "--provider", "http://bad-host:11434",
        "--model", "qwen3.5:9b",
    ])
    assert result.exit_code == 1
    assert "http://bad-host:11434" in result.output
    assert not (tmp_path / "bad-provider" / "sieve.yaml").exists()


def test_install_flow_tolerates_autostart_failure(tmp_path, monkeypatch):
    """If autostart enable fails (e.g. systemd quirks), install still
    succeeds — we log the reason and keep going. The user can retry
    autostart from the Service menu."""
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "autost-fail")
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    monkeypatch.setattr(_installer, "_reachable", lambda *a, **k: True)
    monkeypatch.setattr(_installer, "_default_model_for", lambda *a, **k: "qwen3.5:9b")
    # Embedding step succeeds.
    import fastembed
    class _Ok:
        def __init__(self, *a, **kw):
            pass
    monkeypatch.setattr(fastembed, "TextEmbedding", _Ok)
    # Autostart enable rejects.
    from sieve import _autostart
    monkeypatch.setattr(
        _autostart, "enable_autostart",
        lambda: (False, "systemctl user bus not available"),
    )
    # Start command succeeds silently.
    import sieve.cli
    class _OkStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            pass
    monkeypatch.setattr(sieve.cli, "start", _OkStart)
    # Store init is a no-op to avoid sqlcipher at test time.
    monkeypatch.setattr(_installer, "_init_store", lambda: None)

    from click.testing import CliRunner
    runner = CliRunner()
    # --no-input still asks about autostart only if desktop; here we
    # force the no-input path where autostart defaults False, so the
    # autostart-failure branch wouldn't fire. Explicitly turn on
    # autostart by driving the interactive path with mocked prompts:
    import click as _click
    monkeypatch.setenv("USER", "alice")
    # First confirm (Use local Ollama?) True; second (autostart) True;
    # third (Start now?) True; fourth (Apply plan?) True.
    confirms = iter([True, True, True, True])
    monkeypatch.setattr(_click, "confirm", lambda *a, **k: next(confirms))
    monkeypatch.setattr(
        _installer, "_pick_llm_location",
        lambda c: ("http://127.0.0.1:11434", None),
    )
    monkeypatch.setattr(
        _installer, "_pick_model_step",
        lambda c, url, k: "qwen3.5:9b",
    )
    from sieve import _autostart
    monkeypatch.setattr(_autostart, "autostart_supported", lambda: True)
    monkeypatch.setattr(_autostart, "autostart_status", lambda: "disabled")

    result = runner.invoke(_installer.main, [])
    # Install completes despite autostart failure.
    assert result.exit_code == 0, result.output
    assert (tmp_path / "autost-fail" / "sieve.yaml").exists()


def test_install_flow_fresh_path_writes_yaml(tmp_path, monkeypatch):
    """Happy path end-to-end: --no-input + --provider that's reachable
    + --model that's named. Yaml gets written with the right values."""
    monkeypatch.setattr(_installer, "SIEVE_DIR", tmp_path / "happy")
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    monkeypatch.setattr(_installer, "_reachable", lambda *a, **k: True)
    import fastembed
    class _Ok:
        def __init__(self, *a, **kw):
            pass
    monkeypatch.setattr(fastembed, "TextEmbedding", _Ok)
    monkeypatch.setattr(_installer, "_init_store", lambda: None)
    import sieve.cli
    class _OkStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            pass
    monkeypatch.setattr(sieve.cli, "start", _OkStart)

    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(_installer.main, [
        "--no-input",
        "--provider", "http://192.168.1.149:11434",
        "--model", "gemma:2b",
    ])
    assert result.exit_code == 0, result.output
    yaml_text = (tmp_path / "happy" / "sieve.yaml").read_text()
    assert "base_url: http://192.168.1.149:11434" in yaml_text
    assert "default_model: gemma:2b" in yaml_text


def test_already_installed_branch_renders_status(tmp_path, monkeypatch, capsys):
    """Running sieve-install against an existing install prints status
    and exits cleanly (exit 0)."""
    d = tmp_path / "existing"
    d.mkdir()
    (d / "sieve.yaml").write_text(
        "listen: {host: 127.0.0.1, port: 11435}\n"
        "provider:\n"
        "  type: ollama\n"
        "  base_url: http://127.0.0.1:11434\n"
        "  default_model: qwen3.5:9b\n"
        "embeddings: {provider: fastembed}\n"
        f"store: {{path: {tmp_path / 'm.db'}}}\n"
    )
    monkeypatch.setattr(_installer, "SIEVE_DIR", d)
    monkeypatch.setenv("SIEVE_CONFIG", str(d / "sieve.yaml"))
    # Mock splash so the test output is clean.
    monkeypatch.setattr(_installer, "render_splash", lambda c: None)
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(_installer.main, ["--no-input"])
    # Some platforms return 0, CliRunner sometimes returns 0 + output.
    assert result.exit_code == 0
    assert "already installed" in result.output.lower()
    assert "qwen3.5:9b" in result.output

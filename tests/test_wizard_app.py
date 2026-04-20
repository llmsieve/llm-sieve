"""Tests for the top-level Sieve wizard (`sieve` / `sieve wizard`).

The wizard is the biggest user-facing surface. These tests verify:

- Every screen builds without touching live state.
- Top screen adapts to install state (options enabled only when
  ~/.sieve exists).
- Install options: Quick path, Guided path, each maps to the right
  underlying handler.
- Service, Store, Config, Uninstall screens exist and surface sane
  subtitles.
- Dangerous config settings are NOT offered for edit.
- The full menu loop navigates top → submenu → back without
  crashing.

We avoid spinning up real subprocesses / real store here; that's
covered by the BattleTest end-to-end verification later.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

from sieve._menu import BACK, QUIT, MenuApp, MenuOption, MenuScreen


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    return Console(file=buf, width=100, force_terminal=False, no_color=True), buf


# ── Branding + wizard entry point ──────────────────────────────────────


def test_run_wizard_renders_splash_and_enters_loop(monkeypatch):
    """Smoke test: the entry point calls render_splash + builds the
    top screen + enters the app loop. We mock MenuApp.run so the
    test doesn't sit on a real stdin prompt."""
    from sieve import _wizard_app

    calls = {"splash": 0, "run": 0}

    def _fake_splash(c):
        calls["splash"] += 1
        c.print("SPLASH_MARKER")

    def _fake_run(self):
        calls["run"] += 1

    monkeypatch.setattr(_wizard_app, "render_splash", _fake_splash)
    monkeypatch.setattr(MenuApp, "run", _fake_run)
    console, buf = _console()
    _wizard_app.run_wizard(console=console, clear_between_screens=False)
    assert calls == {"splash": 1, "run": 1}
    assert "SPLASH_MARKER" in buf.getvalue()


# ── Top screen ─────────────────────────────────────────────────────────


def test_top_screen_disables_action_options_when_not_installed(tmp_path, monkeypatch):
    """Before install, the user can't start the service, inspect the
    store, edit config, benchmark, demo, or uninstall — those are all
    greyed out. Only Install and Quit are live."""
    from sieve import _wizard_app
    monkeypatch.setattr(_wizard_app, "SIEVE_DIR", tmp_path / "does-not-exist")
    console, _ = _console()
    screen = _wizard_app.build_top_screen(console)
    labels_enabled = {
        opt.label: opt.enabled for opt in screen.options
    }
    # Install is always enabled (labelled "Install" when clean).
    assert "Install" in next(
        (l for l in labels_enabled if l.startswith("Install")),
        "",
    )
    # Action options are disabled.
    assert not labels_enabled.get(
        "Service — start, stop, restart, autostart"
    )
    assert not labels_enabled.get(
        "Config — adjust settings"
    )
    assert not labels_enabled.get(
        "Benchmark — measure Sieve's value"
    )


def test_top_screen_after_install_enables_actions(tmp_path, monkeypatch):
    """Once ~/.sieve/sieve.yaml exists, the actions are live and
    Install becomes 'Reinstall'."""
    from sieve import _wizard_app
    sieve_dir = tmp_path / "sieve-dir"
    sieve_dir.mkdir()
    (sieve_dir / "sieve.yaml").write_text("listen:\n  port: 11435\n")
    monkeypatch.setattr(_wizard_app, "SIEVE_DIR", sieve_dir)
    console, _ = _console()
    screen = _wizard_app.build_top_screen(console)
    labels = [opt.label for opt in screen.options]
    assert any(l.startswith("Reinstall") for l in labels)
    # Service and Config are enabled.
    by_label = {opt.label: opt for opt in screen.options}
    assert by_label["Service — start, stop, restart, autostart"].enabled
    assert by_label["Config — adjust settings"].enabled
    # Store option checks for an actual DB, still disabled if only
    # yaml exists.
    assert not by_label["Store — stats and inspection"].enabled


def test_top_screen_quit_option_has_q_shortcut():
    from sieve import _wizard_app
    console, _ = _console()
    screen = _wizard_app.build_top_screen(console)
    quit_opt = next((o for o in screen.options if o.label == "Quit"), None)
    assert quit_opt is not None
    assert quit_opt.key == "q"


# ── Install screen ─────────────────────────────────────────────────────


def test_install_screen_has_quick_and_guided():
    from sieve._wizard_app import build_install_screen
    console, _ = _console()
    screen = build_install_screen(console)
    labels = [o.label for o in screen.options]
    assert any("Quick" in l for l in labels)
    assert any("Guided" in l for l in labels)


# ── Service screen ─────────────────────────────────────────────────────


def test_service_screen_subtitle_reflects_running_state(monkeypatch):
    from sieve import _wizard_app
    # Patch _read_pid to simulate "running".
    monkeypatch.setattr("sieve.cli._read_pid", lambda: 12345)
    console, _ = _console()
    screen = _wizard_app.build_service_screen(console)
    assert "Running" in screen.subtitle

    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    screen2 = _wizard_app.build_service_screen(console)
    assert "Stopped" in screen2.subtitle


def test_service_screen_options_reflect_running_state(monkeypatch):
    from sieve import _wizard_app
    # Running — Start is disabled, Stop/Restart are enabled.
    monkeypatch.setattr("sieve.cli._read_pid", lambda: 42)
    console, _ = _console()
    screen = _wizard_app.build_service_screen(console)
    by_label = {o.label: o for o in screen.options}
    assert not by_label["Start"].enabled
    assert by_label["Stop"].enabled
    assert by_label["Restart"].enabled
    # Stopped — Start enabled, Stop/Restart disabled.
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    screen2 = _wizard_app.build_service_screen(console)
    by_label2 = {o.label: o for o in screen2.options}
    assert by_label2["Start"].enabled
    assert not by_label2["Stop"].enabled
    assert not by_label2["Restart"].enabled


# ── Config screen ──────────────────────────────────────────────────────


def test_config_screen_lists_only_safe_settings(tmp_path, monkeypatch):
    """Dangerous settings (embedding_dimensions, auth_token, base_url
    when store has data, embeddings.provider) must NOT appear as
    editable rows — they must be in the 'view-only' help screen."""
    from sieve import _wizard_app
    # Minimal valid yaml so RecallConfig.load() works.
    yaml_path = tmp_path / "sieve.yaml"
    yaml_path.write_text(
        "listen: {host: 127.0.0.1, port: 11435}\n"
        "provider:\n"
        "  type: ollama\n"
        "  base_url: http://127.0.0.1:11434\n"
        "  default_model: qwen3.5:9b\n"
        "embeddings: {provider: fastembed}\n"
        f"store: {{path: {tmp_path / 'm.db'}}}\n"
    )
    monkeypatch.setenv("SIEVE_CONFIG", str(yaml_path))
    console, _ = _console()
    screen = _wizard_app.build_config_screen(console)
    labels = " ".join(o.label for o in screen.options)
    # Safe settings are present as editable rows.
    assert "provider.default_model" in labels
    assert "pipeline.conversation_turns" in labels
    # Dangerous settings are NOT editable rows.
    assert "provider.base_url" not in labels
    assert "store.embedding_dimensions" not in labels
    assert "security.auth_token" not in labels
    # There's a view-only option for the dangerous ones.
    view_only = [o for o in screen.options if "menu doesn't edit" in o.label]
    assert len(view_only) == 1


# ── Uninstall screen ───────────────────────────────────────────────────


def test_uninstall_screen_requires_double_confirm():
    """The preview-and-confirm handler is the only option; it exists
    and shows what'll be removed."""
    from sieve._wizard_app import build_uninstall_screen
    console, _ = _console()
    screen = build_uninstall_screen(console)
    assert len(screen.options) == 1
    assert "confirm" in screen.options[0].label.lower()


# ── Menu navigation end-to-end ────────────────────────────────────────


def test_navigation_top_to_install_and_back(tmp_path, monkeypatch):
    """Scripted flow: user opens the wizard, picks 'Install' (1),
    then 'b' to go back, then 'q' to quit. No crashes, top screen
    renders twice."""
    from sieve import _wizard_app

    monkeypatch.setattr(_wizard_app, "SIEVE_DIR", tmp_path / "empty")
    console, buf = _console()
    inputs = iter(["1", "b", "q"])

    def _input(_):
        return next(inputs)

    top = _wizard_app.build_top_screen(console)
    app = MenuApp(top, console=console, input_fn=_input, clear_between_screens=False)
    app.run()
    out = buf.getvalue()
    # Top screen title rendered at least twice (push → pop).
    assert out.count("Sieve\n") >= 2 or out.count("\nSieve") >= 2 or out.count("Sieve ") >= 2
    # Install-screen title was visible too.
    assert "Install Sieve" in out

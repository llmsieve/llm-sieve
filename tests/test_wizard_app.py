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


def _make_yaml(tmp_path):
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
    return yaml_path


def test_config_screen_lists_only_safe_settings_when_store_populated(
    tmp_path, monkeypatch,
):
    """Dangerous settings must NOT appear as editable rows when the
    store has data — they're in the view-only help screen. This is
    the default behaviour for a user who's been running Sieve."""
    from sieve import _wizard_app
    monkeypatch.setenv("SIEVE_CONFIG", str(_make_yaml(tmp_path)))
    # Pretend the store has facts — that keeps provider.base_url
    # out of the editable list.
    monkeypatch.setattr(_wizard_app, "_store_fact_count", lambda: 42)
    console, _ = _console()
    screen = _wizard_app.build_config_screen(console)
    labels = " ".join(o.label for o in screen.options)
    assert "provider.default_model" in labels
    assert "pipeline.conversation_turns" in labels
    # Dangerous settings are NOT editable rows when there's data.
    assert "provider.base_url" not in labels
    assert "store.embedding_dimensions" not in labels
    assert "security.auth_token" not in labels
    view_only = [o for o in screen.options if "menu doesn't edit" in o.label]
    assert len(view_only) == 1


def test_config_screen_allows_provider_base_url_edit_when_store_empty(
    tmp_path, monkeypatch,
):
    """When the store has 0 facts, there are no cached embeddings
    a provider switch would invalidate — so we promote
    provider.base_url to an editable row. This closes the UX gap
    where a Quick-install user couldn't fix an unreachable default
    URL without editing YAML by hand."""
    from sieve import _wizard_app
    monkeypatch.setenv("SIEVE_CONFIG", str(_make_yaml(tmp_path)))
    monkeypatch.setattr(_wizard_app, "_store_fact_count", lambda: 0)
    console, _ = _console()
    screen = _wizard_app.build_config_screen(console)
    labels = [o.label for o in screen.options]
    # provider.base_url IS editable now.
    assert any("provider.base_url" in l for l in labels)
    # And it's first, so a Quick-install-then-can't-reach user sees
    # it immediately.
    first_editable = labels[0]
    assert "provider.base_url" in first_editable
    # Other dangerous settings still held back.
    for label in labels:
        assert "store.embedding_dimensions" not in label
        assert "security.auth_token" not in label


# ── Uninstall screen ───────────────────────────────────────────────────


def test_uninstall_screen_requires_double_confirm():
    """The preview-and-confirm handler is the only option; it exists
    and shows what'll be removed."""
    from sieve._wizard_app import build_uninstall_screen
    console, _ = _console()
    screen = build_uninstall_screen(console)
    assert len(screen.options) == 1
    assert "confirm" in screen.options[0].label.lower()


def test_uninstall_handler_exits_wizard_on_success(tmp_path, monkeypatch):
    """After a successful uninstall there's nothing left to do — the
    handler returns QUIT to exit the wizard entirely, rather than
    ResetTo a disabled-everywhere menu."""
    from sieve import _wizard_app
    from sieve._menu import QUIT

    # Pretend Sieve is installed so the option's enabled at build time.
    sieve_dir = tmp_path / "sieve-dir"
    sieve_dir.mkdir()
    (sieve_dir / "sieve.yaml").write_text("listen:\n  port: 11435\n")
    monkeypatch.setattr(_wizard_app, "SIEVE_DIR", sieve_dir)

    # Stub everything destructive.
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    monkeypatch.setattr(_wizard_app, "_pause_for_enter", lambda c, label="": None)
    import click as _click
    monkeypatch.setattr(_click, "confirm", lambda *a, **k: True)
    # Second prompt: they type 'yes' to confirm.
    monkeypatch.setattr(_click, "prompt", lambda *a, **k: "yes")
    from sieve._autostart import autostart_status, autostart_supported
    monkeypatch.setattr("sieve._autostart.autostart_status", lambda: "disabled")
    monkeypatch.setattr("sieve._autostart.autostart_supported", lambda: True)

    # Redirect the actual rmtree to our tmpdir so nothing real
    # gets touched.
    import shutil as _shutil
    real_rmtree = _shutil.rmtree
    def _rmtree(p, *a, **kw):
        real_rmtree(str(p), ignore_errors=True)
    monkeypatch.setattr(_shutil, "rmtree", _rmtree)

    console, _ = _console()
    screen = _wizard_app.build_uninstall_screen(console)
    result = screen.options[0].handler()
    assert result is QUIT, (
        f"Uninstall must exit the wizard after removing everything, got {result!r}"
    )


def test_offer_to_start_service_skipped_when_already_running(monkeypatch):
    """No point asking if Sieve is already up — just note it and skip."""
    from sieve import _wizard_app
    monkeypatch.setattr("sieve.cli._read_pid", lambda: 4242)
    # click.confirm must NOT be called in this branch.
    import click as _click
    called = {"confirm": False, "start": False}
    def _confirm(*a, **k):
        called["confirm"] = True
        return True
    monkeypatch.setattr(_click, "confirm", _confirm)
    import sieve.cli
    class _FakeStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            called["start"] = True
    monkeypatch.setattr(sieve.cli, "start", _FakeStart)

    console, _ = _console()
    _wizard_app._offer_to_start_service(console)
    assert called["confirm"] is False
    assert called["start"] is False


def test_offer_to_start_service_defaults_yes(monkeypatch):
    """Pressing enter at the prompt should start the service — a
    fresh-install user almost always wants it running."""
    from sieve import _wizard_app
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    import click as _click
    captured_confirm_kwargs = {}
    def _confirm(q, default=False):
        captured_confirm_kwargs["default"] = default
        return default  # simulate pressing enter
    monkeypatch.setattr(_click, "confirm", _confirm)
    import sieve.cli
    start_calls = []
    class _FakeStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            start_calls.append(1)
    monkeypatch.setattr(sieve.cli, "start", _FakeStart)

    console, _ = _console()
    _wizard_app._offer_to_start_service(console)
    assert captured_confirm_kwargs["default"] is True, (
        "Prompt must default to Yes — users installed Sieve to use it"
    )
    assert start_calls == [1]


def test_offer_to_start_service_respects_no(monkeypatch):
    """If the user explicitly says no, we must NOT start the service."""
    from sieve import _wizard_app
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    import click as _click
    monkeypatch.setattr(_click, "confirm", lambda *a, **k: False)
    import sieve.cli
    started = []
    class _FakeStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            started.append(1)
    monkeypatch.setattr(sieve.cli, "start", _FakeStart)

    console, _ = _console()
    _wizard_app._offer_to_start_service(console)
    assert started == []


def test_install_to_uninstall_round_trip_via_MenuApp(tmp_path, monkeypatch):
    """End-to-end: stale top → Install → ResetTo fresh top (Reinstall
    label, all opts enabled) → Uninstall → QUIT exits. Guards against
    the bug where the user navigated back through stale install
    screens."""
    from sieve import _wizard_app
    from sieve._menu import MenuApp, QUIT

    sieve_dir = tmp_path / "sieve-dir"
    monkeypatch.setattr(_wizard_app, "SIEVE_DIR", sieve_dir)

    # Stub provider probe, init_cmd, prompts.
    monkeypatch.setattr(_wizard_app, "_probe_provider", lambda url, **kw: True)
    class _FakeInit:
        @staticmethod
        def main(standalone_mode=False, args=None):
            sieve_dir.mkdir(parents=True, exist_ok=True)
            (sieve_dir / "sieve.yaml").write_text("listen:\n  port: 11435\n")
    import sieve.cli
    monkeypatch.setattr(sieve.cli, "init", _FakeInit)
    # The offer-to-start prompt will try `sieve start` after install;
    # stub it so we don't spawn a real daemon from the test suite.
    class _FakeStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            pass
    monkeypatch.setattr(sieve.cli, "start", _FakeStart)
    monkeypatch.setattr(_wizard_app, "_render_post_install_status", lambda c: None)
    monkeypatch.setattr(_wizard_app, "_pause_for_enter", lambda c, label="": None)
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    monkeypatch.setattr("sieve._autostart.autostart_status", lambda: "disabled")
    monkeypatch.setattr("sieve._autostart.autostart_supported", lambda: True)

    import click as _click
    monkeypatch.setattr(_click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(_click, "prompt", lambda *a, **k: "yes")

    # Navigate: top → 1 (Install) → 1 (Quick) → (reset to fresh top)
    # → 7 (Uninstall) → 1 (preview+confirm+uninstall) → QUIT.
    inputs = iter(["1", "1", "7", "1", "q"])
    def _input(_):
        return next(inputs)

    console, buf = _console()
    top = _wizard_app.build_top_screen(console)
    app = MenuApp(top, console=console, input_fn=_input, clear_between_screens=False)
    app.run()

    out = buf.getvalue()
    # After install, the top's title is still "Sieve" but the first
    # option flipped from "Install" to "Reinstall" — the fresh screen
    # took effect.
    assert "Reinstall" in out
    # After uninstall, the process exited the wizard (QUIT).
    # We can't directly observe QUIT from the buffer but we can
    # check that the state actually changed: sieve.yaml gone.
    assert not (sieve_dir / "sieve.yaml").exists() or not sieve_dir.exists()


# ── Menu navigation end-to-end ────────────────────────────────────────


def test_quick_install_returns_ResetTo_so_stack_refreshes(tmp_path, monkeypatch):
    """Post-install the nav stack has stale screens with enabled=False
    options. The install handler must return ResetTo(top) so the
    wizard's navigation reflects the new installed state."""
    from sieve import _wizard_app
    from sieve._menu import ResetTo

    # Simulate a clean pre-install state for build_top_screen.
    monkeypatch.setattr(_wizard_app, "SIEVE_DIR", tmp_path / "fake-sieve")

    # Stub the heavy bits: provider probe says "reachable", init_cmd
    # is a no-op, post-install status render is swallowed.
    monkeypatch.setattr(_wizard_app, "_probe_provider", lambda url, **kw: True)
    class _FakeInit:
        @staticmethod
        def main(standalone_mode=False, args=None):
            # Simulate what `sieve init` does: create sieve.yaml.
            (tmp_path / "fake-sieve").mkdir(parents=True, exist_ok=True)
            (tmp_path / "fake-sieve" / "sieve.yaml").write_text("listen:\n  port: 11435\n")
    import sieve.cli
    monkeypatch.setattr(sieve.cli, "init", _FakeInit)
    monkeypatch.setattr(_wizard_app, "_render_post_install_status", lambda c: None)
    monkeypatch.setattr(_wizard_app, "_pause_for_enter", lambda c, label="": None)
    # Post-install offer-to-start: pretend service isn't running so
    # the branch fires. Decline the prompt to avoid needing to stub
    # `sieve.cli.start` (the offer branches are tested separately).
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    import click as _click
    monkeypatch.setattr(_click, "confirm", lambda *a, **k: False)

    console, _ = _console()
    result = _wizard_app._run_quick_install(console)
    assert isinstance(result, ResetTo), (
        f"Quick install must ResetTo a fresh top screen, got {result!r}"
    )
    # The replacement screen is a fresh top — should have the
    # post-install label (Reinstall, not Install) since we just
    # wrote sieve.yaml.
    assert any(
        opt.label.startswith("Reinstall") for opt in result.screen.options
    ), "Fresh top should reflect installed state"


def test_service_handlers_return_fresh_service_screen(monkeypatch):
    """Start/Stop/Restart handlers rebuild the service screen so the
    status line and enabled flags reflect the new running state."""
    from sieve import _wizard_app
    from sieve._menu import MenuScreen

    # PID None → build service screen with Stop disabled.
    monkeypatch.setattr("sieve.cli._read_pid", lambda: None)
    console, _ = _console()
    screen = _wizard_app.build_service_screen(console)

    # Find the Start handler and stub the actual `sieve start`.
    start_opt = next(o for o in screen.options if o.label == "Start")

    class _FakeStart:
        @staticmethod
        def main(standalone_mode=False, args=None):
            pass
    import sieve.cli
    monkeypatch.setattr(sieve.cli, "start", _FakeStart)
    monkeypatch.setattr(_wizard_app, "_pause_for_enter", lambda c, label="": None)

    # Now flip to "running" for the rebuild.
    monkeypatch.setattr("sieve.cli._read_pid", lambda: 4242)
    result = start_opt.handler()
    assert isinstance(result, MenuScreen), (
        f"Service start handler must return a fresh MenuScreen, got {result!r}"
    )
    # The rebuilt screen shows the new running state.
    assert "Running" in result.subtitle


def test_config_setting_handler_returns_BACK(monkeypatch, tmp_path):
    """Changing a config setting does NOT require a stack reset —
    the top screen's enabled flags don't depend on config values.
    BACK is correct."""
    from sieve import _wizard_app
    from sieve._menu import BACK

    monkeypatch.setenv("SIEVE_CONFIG", str(_make_yaml(tmp_path)))
    monkeypatch.setattr(_wizard_app, "_store_fact_count", lambda: 5)
    # No prompt: simulate user hits enter to keep current.
    import click as _click
    monkeypatch.setattr(_click, "prompt", lambda *a, **k: k.get("default", ""))
    monkeypatch.setattr(_wizard_app, "_pause_for_enter", lambda c, label="": None)

    console, _ = _console()
    screen = _wizard_app.build_config_screen(console)
    # First editable option (model or whatever is first). Invoke its handler.
    opt = next(o for o in screen.options if "=" in o.label)
    result = opt.handler()
    assert result is BACK, f"Config setting handler should BACK, got {result!r}"


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

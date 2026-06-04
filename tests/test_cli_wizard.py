"""Tests for sieve init --wizard interactive setup.

The wizard is built around a Prompter protocol so tests can feed canned
answers without touching stdin. Network probes are stubbed via a
ProviderProbe protocol. Filesystem effects land under tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sieve.cli_wizard import (
    WizardAnswers,
    WizardContext,
    ListPrompter,
    run_wizard,
)


# --- Prompter ---

class TestListPrompter:
    def test_ask_returns_next_answer(self):
        p = ListPrompter(["one", "two"])
        assert p.ask("Q1", default=None) == "one"
        assert p.ask("Q2", default=None) == "two"

    def test_ask_empty_string_returns_default(self):
        p = ListPrompter(["", ""])
        assert p.ask("Q1", default="fallback") == "fallback"

    def test_confirm_yes(self):
        p = ListPrompter(["y", "n"])
        assert p.confirm("Q?", default=False) is True
        assert p.confirm("Q?", default=False) is False

    def test_confirm_empty_uses_default(self):
        p = ListPrompter([""])
        assert p.confirm("Q?", default=True) is True

    def test_ask_out_of_answers_raises(self):
        p = ListPrompter([])
        with pytest.raises(RuntimeError, match="ran out of answers"):
            p.ask("Q1", default="x")


# --- run_wizard happy path ---

class _FakeProbe:
    """Stub provider probe; returns a canned (reachable, models) tuple."""

    def __init__(self, reachable: bool, models: list[str] | None = None):
        self.reachable = reachable
        self.models = models or []
        self.calls: list[str] = []

    def check(self, url: str) -> tuple[bool, list[str]]:
        self.calls.append(url)
        return self.reachable, self.models


@pytest.fixture
def sieve_dir(tmp_path, monkeypatch):
    """Redirect the wizard's SIEVE_DIR to a tmp location."""
    d = tmp_path / ".sieve"
    monkeypatch.setattr("sieve.cli_wizard.SIEVE_DIR", d)
    return d


def test_wizard_ollama_auto_detect_happy_path(sieve_dir):
    prompter = ListPrompter([
        "1",                         # Step 1: provider type = Ollama
        "",                          # Step 1: URL = default (Ollama auto)
        "1",                         # Step 2: first model (qwen3.5:9b)
        "",                          # Step 3: port = default 11435
        "y",                         # Step 4: auto-generate key
        "",                          # Step 5: store location = default
        "y",                         # Step 6: confirm
    ])
    probe = _FakeProbe(reachable=True, models=["qwen3.5:9b", "llama3:8b"])

    ctx = WizardContext(
        prompter=prompter, probe=probe,
        download_model=lambda: None,
        port_available=lambda _: True,
    )
    answers = run_wizard(ctx)

    assert isinstance(answers, WizardAnswers)
    assert answers.provider_type == "ollama"
    assert answers.provider_url == "http://127.0.0.1:11434"
    assert answers.model == "qwen3.5:9b"
    assert answers.port == 11435
    assert answers.generate_key is True
    assert answers.store_path == sieve_dir / "memory.db"
    assert answers.confirmed is True


def test_wizard_cloud_provider_prompts_for_api_key(sieve_dir):
    prompter = ListPrompter([
        "2",                         # provider type = OpenAI
        "https://api.openai.com/v1", # URL (no auto for cloud)
        "sk-test-key",               # api key
        "gpt-4o-mini",               # model (free-text for cloud)
        "",                          # port default
        "y",                         # auto key
        "",                          # store default
        "y",                         # confirm
    ])
    probe = _FakeProbe(reachable=True)

    ctx = WizardContext(
        prompter=prompter, probe=probe,
        download_model=lambda: None,
        port_available=lambda _: True,
    )
    answers = run_wizard(ctx)

    assert answers.provider_type == "openai"
    assert answers.api_key == "sk-test-key"
    assert answers.model == "gpt-4o-mini"


def test_wizard_connection_fails_then_retry_succeeds(sieve_dir):
    prompter = ListPrompter([
        "1",                         # ollama
        "http://wrong:11434",        # first URL (will fail)
        "r",                         # choose retry
        "http://127.0.0.1:11434",    # retry URL
        "1",                         # model
        "",                          # port
        "y",                         # key
        "",                          # store
        "y",                         # confirm
    ])
    # Probe: first call fails, second succeeds.
    probe = _FakeProbe(reachable=False)
    def check(url):
        probe.calls.append(url)
        return (True, ["m1"]) if "127.0.0.1" in url else (False, [])
    probe.check = check

    ctx = WizardContext(
        prompter=prompter, probe=probe,
        download_model=lambda: None,
        port_available=lambda _: True,
    )
    answers = run_wizard(ctx)
    assert answers.provider_url == "http://127.0.0.1:11434"
    assert answers.model == "m1"


def test_wizard_port_in_use_prompts_for_alternative(sieve_dir):
    prompter = ListPrompter([
        "1", "", "1",          # provider / model
        "11435",               # first port (will be "in use")
        "11436",               # alternative port
        "y", "",               # key + store defaults
        "y",                   # confirm
    ])
    probe = _FakeProbe(reachable=True, models=["m1"])
    # Port checker: 11435 in use, 11436 free.
    port_checker = lambda p: p == 11436
    ctx = WizardContext(
        prompter=prompter, probe=probe,
        download_model=lambda: None,
        port_available=port_checker,
    )
    answers = run_wizard(ctx)
    assert answers.port == 11436


def test_wizard_custom_passphrase(sieve_dir):
    prompter = ListPrompter([
        "1", "", "1", "",             # provider+model+port defaults
        "n",                          # DON'T auto-generate
        "my-secret-passphrase-123",   # custom passphrase
        "",                           # store default
        "y",                          # confirm
    ])
    probe = _FakeProbe(reachable=True, models=["m1"])
    ctx = WizardContext(
        prompter=prompter, probe=probe,
        download_model=lambda: None,
        port_available=lambda _: True,
    )
    answers = run_wizard(ctx)
    assert answers.generate_key is False
    assert answers.passphrase == "my-secret-passphrase-123"


def test_wizard_user_rejects_at_confirmation(sieve_dir):
    prompter = ListPrompter([
        "1", "", "1", "", "y", "",
        "n",            # reject the summary
    ])
    probe = _FakeProbe(reachable=True, models=["m1"])
    ctx = WizardContext(
        prompter=prompter, probe=probe,
        download_model=lambda: None,
        port_available=lambda _: True,
    )
    answers = run_wizard(ctx)
    assert answers.confirmed is False


# --- apply_wizard_answers writes config + creates key/store ---

def test_apply_wizard_writes_config_file(sieve_dir, tmp_path):
    from sieve.cli_wizard import apply_wizard_answers

    answers = WizardAnswers(
        provider_type="ollama",
        provider_url="http://127.0.0.1:11434",
        api_key=None,
        model="qwen3.5:9b",
        port=11500,
        generate_key=True,
        passphrase=None,
        store_path=sieve_dir / "memory.db",
        confirmed=True,
    )

    apply_wizard_answers(answers)

    cfg_path = sieve_dir / "sieve.yaml"
    assert cfg_path.exists()
    data = yaml.safe_load(cfg_path.read_text())
    assert data["listen"]["port"] == 11500
    assert data["provider"]["base_url"] == "http://127.0.0.1:11434"
    assert data["provider"]["default_model"] == "qwen3.5:9b"


def test_apply_wizard_custom_passphrase_written_to_keyfile(sieve_dir):
    from sieve.cli_wizard import apply_wizard_answers

    answers = WizardAnswers(
        provider_type="ollama",
        provider_url="http://127.0.0.1:11434",
        api_key=None,
        model="qwen3.5:9b",
        port=11435,
        generate_key=False,
        passphrase="custom-secret",
        store_path=sieve_dir / "memory.db",
        confirmed=True,
    )
    apply_wizard_answers(answers)

    keyfile = sieve_dir / ".sieve_key"
    assert keyfile.exists()
    assert keyfile.read_text().strip() == "custom-secret"
    # 0600 perms (owner r/w only)
    assert oct(keyfile.stat().st_mode)[-3:] == "600"


def test_apply_wizard_rejected_writes_nothing(sieve_dir):
    from sieve.cli_wizard import apply_wizard_answers

    answers = WizardAnswers(
        provider_type="ollama", provider_url="", api_key=None,
        model="", port=11435, generate_key=True, passphrase=None,
        store_path=sieve_dir / "memory.db", confirmed=False,
    )
    apply_wizard_answers(answers)

    assert not (sieve_dir / "sieve.yaml").exists()


# --- CLI surface: `sieve init --wizard` ---

def test_sieve_init_wizard_invokes_wizard_path(monkeypatch, sieve_dir):
    """The --wizard flag must route init through run_wizard, not the
    original lazy-default init path."""
    from sieve import cli as cli_mod

    called: dict[str, bool] = {"wizard": False, "lazy": False}

    def fake_wizard_flow():
        called["wizard"] = True

    monkeypatch.setattr(cli_mod, "_run_wizard_flow", fake_wizard_flow)

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["init", "--wizard"])
    assert result.exit_code == 0, result.output
    assert called["wizard"] is True


def test_sieve_init_default_path_unchanged(monkeypatch, sieve_dir):
    """Plain `sieve init` must NOT invoke the wizard."""
    from sieve import cli as cli_mod

    called = {"wizard": False}
    monkeypatch.setattr(cli_mod, "_run_wizard_flow", lambda: called.__setitem__("wizard", True))
    # Force the lazy path to no-op (it already exists) by making SIEVE_DIR
    # preexist and answering "no" to reinitialise.
    sieve_dir.mkdir(parents=True, exist_ok=True)
    (sieve_dir / "sieve.yaml").write_text("listen:\n  port: 11435\n")

    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["init"], input="n\n")
    # Accept any exit code; what matters is the wizard was NOT triggered.
    assert called["wizard"] is False

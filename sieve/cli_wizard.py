"""Interactive `sieve init --wizard` flow.

Split out of cli.py so the wizard can be unit-tested without a TTY:
``run_wizard`` takes a ``WizardContext`` that injects the prompter,
the provider probe, and the port checker. The CLI wraps these with
real implementations in ``_run_wizard_flow``.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import yaml


SIEVE_DIR = Path("~/.sieve").expanduser()


# ── Prompter protocol ──────────────────────────────────────────────────────

class Prompter(Protocol):
    def ask(self, question: str, default: str | None = None) -> str: ...
    def confirm(self, question: str, default: bool = True) -> bool: ...


class ListPrompter:
    """A canned-answer prompter used by tests.

    Feeds entries from a list in order. Empty strings fall through to
    the caller-supplied default (matches Click's prompt semantics).
    """

    def __init__(self, answers: list[str]):
        self._answers = list(answers)

    def ask(self, question: str, default: str | None = None) -> str:
        if not self._answers:
            raise RuntimeError(
                f"ListPrompter ran out of answers at ask({question!r})"
            )
        raw = self._answers.pop(0)
        if raw == "" and default is not None:
            return default
        return raw

    def confirm(self, question: str, default: bool = True) -> bool:
        if not self._answers:
            raise RuntimeError(
                f"ListPrompter ran out of answers at confirm({question!r})"
            )
        raw = self._answers.pop(0).strip().lower()
        if raw == "":
            return default
        return raw in ("y", "yes", "true", "1")


# ── Context + answers ──────────────────────────────────────────────────────

class ProviderProbe(Protocol):
    def check(self, url: str) -> tuple[bool, list[str]]: ...


def _default_port_available(port: int) -> bool:
    """Try to bind a TCP socket on 127.0.0.1:port to test availability."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        s.close()
    return True


@dataclass
class WizardContext:
    prompter: Prompter
    probe: ProviderProbe
    download_model: Callable[[], None]
    port_available: Callable[[int], bool] = field(
        default=_default_port_available
    )


@dataclass
class WizardAnswers:
    provider_type: str           # ollama|openai|anthropic|custom
    provider_url: str
    api_key: str | None
    model: str
    port: int
    generate_key: bool
    passphrase: str | None       # None when generate_key is True
    store_path: Path
    confirmed: bool


# ── Wizard ─────────────────────────────────────────────────────────────────

_PROVIDER_CHOICES = {
    "1": ("ollama", "http://127.0.0.1:11434"),
    "2": ("openai", "https://api.openai.com/v1"),
    "3": ("anthropic", "https://api.anthropic.com"),
    "4": ("custom", ""),
}


def _step_provider(ctx: WizardContext) -> tuple[str, str, str | None, str]:
    """Returns (provider_type, url, api_key, chosen_model).

    Loops on failed connection tests until the user either succeeds or
    types 'a' to abort. The 'r' shortcut re-prompts for the URL.
    """
    # --- provider type ---
    choice = ctx.prompter.ask(
        "Provider [1=Ollama 2=OpenAI 3=Anthropic 4=Custom]",
        default="1",
    )
    provider_type, default_url = _PROVIDER_CHOICES.get(
        choice, _PROVIDER_CHOICES["1"]
    )

    # --- URL + connection test (with retry loop) ---
    url = ctx.prompter.ask(
        "Provider base URL", default=default_url
    ) or default_url
    api_key: str | None = None
    models: list[str] = []

    while True:
        reachable, models = ctx.probe.check(url)
        if reachable:
            break
        choice = ctx.prompter.ask(
            f"Could not reach {url}. [r]etry with new URL, [s]kip test, [a]bort",
            default="r",
        )
        if choice == "s":
            break
        if choice == "a":
            raise SystemExit(
                f"Wizard aborted — could not reach provider at {url}"
            )
        url = ctx.prompter.ask("Provider base URL", default=default_url) or default_url

    # --- cloud providers need an API key ---
    if provider_type in ("openai", "anthropic"):
        api_key = ctx.prompter.ask(
            f"{provider_type.title()} API key", default=""
        )

    # --- model selection ---
    if provider_type == "ollama" and models:
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")
        pick = ctx.prompter.ask(
            f"Model [1-{len(models)}]", default="1"
        )
        try:
            model = models[int(pick) - 1]
        except (ValueError, IndexError):
            model = models[0]
    else:
        suggested = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "ollama": "qwen3.5:9b",
            "custom": "qwen3.5:9b",
        }[provider_type]
        model = ctx.prompter.ask("Model", default=suggested) or suggested

    return provider_type, url, api_key, model


def _step_port(ctx: WizardContext) -> int:
    """Prompt for port, reject ones already bound."""
    while True:
        raw = ctx.prompter.ask(
            "Sieve listen port", default="11435"
        )
        try:
            port = int(raw)
        except ValueError:
            continue
        if ctx.port_available(port):
            return port
        # else: loop again — user should pick another


def _step_key(ctx: WizardContext) -> tuple[bool, str | None]:
    generate = ctx.prompter.confirm(
        "Generate encryption key automatically? [Y/n]", default=True
    )
    if generate:
        return True, None
    pp = ctx.prompter.ask("Custom passphrase", default="")
    if not pp:
        # Fall back to generated if the user leaves it blank
        return True, None
    return False, pp


def _step_store(ctx: WizardContext) -> Path:
    default = str(SIEVE_DIR / "memory.db")
    raw = ctx.prompter.ask("Store path", default=default)
    return Path(raw).expanduser()


def _step_confirm(ctx: WizardContext, answers: WizardAnswers) -> bool:
    # Render summary
    print("")
    print("Summary:")
    print(f"  Provider: {answers.provider_type}  ({answers.provider_url})")
    if answers.api_key:
        print(f"  API key: {'*' * 8}{answers.api_key[-4:]}")
    print(f"  Model: {answers.model}")
    print(f"  Port: {answers.port}")
    print(
        "  Key: auto-generated"
        if answers.generate_key
        else "  Key: custom passphrase"
    )
    print(f"  Store: {answers.store_path}")
    print("")
    return ctx.prompter.confirm("Proceed?", default=True)


def run_wizard(ctx: WizardContext) -> WizardAnswers:
    provider_type, url, api_key, model = _step_provider(ctx)
    port = _step_port(ctx)
    generate_key, passphrase = _step_key(ctx)
    store_path = _step_store(ctx)

    answers = WizardAnswers(
        provider_type=provider_type,
        provider_url=url,
        api_key=api_key or None,
        model=model,
        port=port,
        generate_key=generate_key,
        passphrase=passphrase,
        store_path=store_path,
        confirmed=False,
    )
    answers.confirmed = _step_confirm(ctx, answers)
    return answers


# ── Apply ──────────────────────────────────────────────────────────────────

def _build_yaml(answers: WizardAnswers) -> str:
    data = {
        "listen": {"host": "127.0.0.1", "port": answers.port},
        "provider": {
            "type": "auto",
            "base_url": answers.provider_url,
            "default_model": answers.model,
        },
        "embeddings": {"provider": "fastembed"},
        "store": {"path": str(answers.store_path)},
    }
    if answers.api_key:
        data["provider"]["api_key"] = answers.api_key
    return yaml.safe_dump(data, sort_keys=False)


def apply_wizard_answers(answers: WizardAnswers) -> None:
    """Write config, keyfile, and initialise the store.

    No-op when the user rejected the summary.
    """
    if not answers.confirmed:
        return

    sieve_dir = SIEVE_DIR
    sieve_dir.mkdir(parents=True, exist_ok=True)

    # Config
    (sieve_dir / "sieve.yaml").write_text(_build_yaml(answers))

    # Keyfile — only for custom passphrases; the generated-key path leaves
    # it to sieve.store.get_or_create_passphrase on first open.
    if not answers.generate_key and answers.passphrase:
        keyfile = sieve_dir / ".sieve_key"
        keyfile.parent.mkdir(parents=True, exist_ok=True)
        keyfile.write_text(answers.passphrase)
        keyfile.chmod(0o600)

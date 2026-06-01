"""`sieve-install` — the Rolls-Royce first-run experience.

Distinct from ``sieve wizard``:

- ``sieve-install`` is the one-command path from "pip installed" to
  "proxy running, point your agent here". Purpose-built for first
  run. Covers local, LAN, and cloud endpoints in one flow.
- ``sieve wizard`` is the ongoing management menu for users who
  already have Sieve set up.

Design principles (see the D9 task for rationale):

1. Detect, don't assume — probe every external thing.
2. Bounded timeouts — no network call can hang the installer.
3. Idempotent — running twice produces the same end state as once.
4. Surface errors, don't swallow them — users get actual causes.
5. Every prompt is skippable via ``--no-input`` + flags.
6. State is auditable — we don't write config until we're committing.
7. Failed steps roll back — never leave the user in a half-state.
8. Plain-text fallbacks when rich / colour / unicode aren't safe.

The whole thing is one file because it's one flow. If it grows
past ~800 lines we'll split into ``_installer/*.py`` by step.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click

from sieve._branding import render_splash
from sieve._menu import MenuOption, MenuScreen

logger = logging.getLogger("recall.installer")

SIEVE_DIR = Path("~/.sieve").expanduser()

# All probes have a bounded timeout so the installer never hangs on
# a wedged endpoint. 3s is long enough for normal loads on slow LANs
# and short enough that a wrong URL fails within a coffee-sip.
_PROBE_TIMEOUT_S = 3.0


# ── Decisions captured by the wizard, applied at the end ──────────────


@dataclass
class InstallPlan:
    """What the installer is about to do.

    Collected across the prompts so that (a) we can show a confirm
    summary before writing anything, and (b) the commit step is
    atomic — we either apply the whole plan or none of it.
    """
    provider_url: str
    provider_api_key: str | None
    model: str
    autostart: bool
    start_now: bool

    def redacted(self) -> "InstallPlan":
        """Copy safe to display — redacts any API key to '…'."""
        return InstallPlan(
            provider_url=self.provider_url,
            provider_api_key=(
                "…" if self.provider_api_key else None
            ),
            model=self.model,
            autostart=self.autostart,
            start_now=self.start_now,
        )


# ── Main entry point ─────────────────────────────────────────────────


@click.command()
@click.option(
    "--no-input",
    is_flag=True,
    default=False,
    help="Skip all prompts; use defaults. For CI / scripted installs.",
)
@click.option(
    "--provider",
    default=None,
    help="Skip the 'where is your LLM' step and use this URL.",
)
@click.option(
    "--model",
    default=None,
    help="Skip the model picker and use this model name.",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for cloud endpoints. Read from env if --provider is "
    "a cloud URL and this isn't set.",
)
def main(
    no_input: bool,
    provider: str | None,
    model: str | None,
    api_key: str | None,
) -> None:
    """Set up Sieve in one flow: splash → provider → model →
    autostart → start → ready panel.
    """
    # Install signal handlers so Ctrl-C leaves the machine in a
    # recoverable state. Any config written gets rolled back.
    rollback_handlers: list[Callable[[], None]] = []
    _install_cleanup_hooks(rollback_handlers)

    console = _make_console()

    try:
        _main_flow(
            console=console,
            no_input=no_input,
            provider_override=provider,
            model_override=model,
            api_key_override=api_key,
            rollback_handlers=rollback_handlers,
        )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Install cancelled.[/] Any changes have been "
            "rolled back — run `sieve-install` again to retry."
        )
        _run_rollback(rollback_handlers, console)
        sys.exit(130)
    except _InstallerExit as exc:
        # Clean "user said no / environment can't support this" exit.
        if exc.message:
            console.print(exc.message)
        _run_rollback(rollback_handlers, console)
        sys.exit(exc.code)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"\n[bold red]Install failed:[/] {exc}\n"
            "[dim]Run `sieve-install` again to retry. Existing config "
            "(if any) has not been modified.[/]"
        )
        _run_rollback(rollback_handlers, console)
        sys.exit(1)


class _InstallerExit(Exception):
    """Clean exit with a user-facing message. Code 0 = normal exit."""
    def __init__(self, message: str = "", code: int = 0):
        self.message = message
        self.code = code


def _make_console():
    """Build a rich Console that degrades gracefully on dumb terminals."""
    from rich.console import Console
    import shutil
    try:
        width = shutil.get_terminal_size((100, 24)).columns
    except Exception:
        width = 100
    return Console(width=max(80, min(120, width)))


def _install_cleanup_hooks(rollback: list[Callable[[], None]]) -> None:
    """SIGTERM → same rollback path as Ctrl-C. SIGINT is handled by
    Python's default KeyboardInterrupt mechanism."""
    def _on_term(_sig, _frame):
        # Running rollback before exit is best-effort; on SIGTERM
        # from a parent that's about to kill us anyway, we have
        # limited time.
        for fn in reversed(rollback):
            try:
                fn()
            except Exception:
                pass
        sys.exit(143)
    try:
        signal.signal(signal.SIGTERM, _on_term)
    except Exception:
        # Signal handlers can't always be installed (non-main thread,
        # some Windows conditions). Not a showstopper.
        pass


def _run_rollback(rollback: list[Callable[[], None]], console) -> None:
    """Run rollback hooks in reverse order. Swallows every error —
    cleanup must not raise."""
    for fn in reversed(rollback):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.debug("rollback hook failed: %s", exc)


# ── Main flow ────────────────────────────────────────────────────────


def _main_flow(
    *,
    console,
    no_input: bool,
    provider_override: str | None,
    model_override: str | None,
    api_key_override: str | None,
    rollback_handlers: list[Callable[[], None]],
) -> None:
    render_splash(console)

    # ── Already installed? ──────────────────────────────────────
    if _is_already_installed():
        _render_already_installed(console)
        raise _InstallerExit(code=0)

    # ── Welcome / context ───────────────────────────────────────
    _render_welcome(console)

    # ── Step 1: provider URL + optional API key ────────────────
    if provider_override:
        provider_url = provider_override
        provider_api_key = api_key_override or _api_key_from_env(provider_url)
        # Probe it; fail fast if provided and unreachable.
        if not _reachable(provider_url, api_key=provider_api_key):
            raise _InstallerExit(
                message=(
                    f"[red]--provider {provider_url} is not reachable.[/] "
                    "Verify it and retry."
                ),
                code=1,
            )
    else:
        if no_input:
            # Auto-detect a provider in priority order:
            # 1. ANTHROPIC_API_KEY env var → Anthropic
            # 2. OPENAI_API_KEY env var → OpenAI
            # 3. Local Ollama on 11434 → Ollama
            # Fail-fast with concrete next steps if none.
            provider_url, provider_api_key = _auto_detect_provider()
            if provider_url is None:
                raise _InstallerExit(
                    message=(
                        "[red]--no-input couldn't auto-detect a provider.[/]\n"
                        "[dim]Tried in order:[/]\n"
                        "  • ANTHROPIC_API_KEY env var (not set)\n"
                        "  • OPENAI_API_KEY env var (not set)\n"
                        "  • Ollama at http://127.0.0.1:11434 (not reachable)\n\n"
                        "[bold]Next steps:[/]\n"
                        "  • Set ANTHROPIC_API_KEY or OPENAI_API_KEY, OR\n"
                        "  • Start Ollama: [cyan]curl -fsSL https://ollama.com/install.sh | sh[/], OR\n"
                        "  • Pass [cyan]--provider URL --api-key KEY[/] explicitly."
                    ),
                    code=1,
                )
            console.print(
                f"[green]Auto-detected provider: {provider_url}[/]"
            )
        else:
            provider_url, provider_api_key = _pick_llm_location(console)

    # ── Step 2: model ──────────────────────────────────────────
    if model_override:
        chosen_model = model_override
    elif no_input:
        # Default to the first model the endpoint lists, or a
        # sensible per-provider fallback if listing isn't supported.
        chosen_model = _default_model_for(provider_url, provider_api_key)
    else:
        chosen_model = _pick_model_step(
            console, provider_url, provider_api_key,
        )

    # ── Step 3: autostart ──────────────────────────────────────
    autostart = _pick_autostart(console, no_input=no_input)

    # ── Step 4: start-now ──────────────────────────────────────
    if no_input:
        start_now = True  # No-input installs assume you want it running.
    else:
        start_now = _pick_start_now(console)

    plan = InstallPlan(
        provider_url=provider_url,
        provider_api_key=provider_api_key,
        model=chosen_model,
        autostart=autostart,
        start_now=start_now,
    )

    # ── Final confirm (skipped on --no-input) ──────────────────
    if not no_input:
        _render_plan_preview(console, plan)
        go = click.confirm("Apply this plan?", default=True)
        if not go:
            raise _InstallerExit(
                message="[yellow]Cancelled. No changes written.[/]",
                code=0,
            )

    # ── Apply: every step records a rollback hook ──────────────
    _apply_plan(console, plan, rollback_handlers)

    # Successful — clear rollback so we don't undo ourselves.
    rollback_handlers.clear()

    # ── Ready ──────────────────────────────────────────────────
    _render_ready_panel(console, plan)


# ── State detection ──────────────────────────────────────────────────


def _is_already_installed() -> bool:
    """Yaml present and readable? That's our install signal.

    NOTE: a broken / half-written yaml is NOT counted as installed;
    the user should be able to re-run the installer and have it
    complete the job. We catch any load exception and treat as
    "not installed".
    """
    yaml_path = SIEVE_DIR / "sieve.yaml"
    if not yaml_path.exists():
        return False
    try:
        from sieve.config import RecallConfig
        RecallConfig.load()
        return True
    except Exception:
        return False


def _render_already_installed(console) -> None:
    from rich.panel import Panel
    from sieve.config import RecallConfig
    cfg = RecallConfig.load()
    try:
        from sieve.cli import _read_pid
        running = _read_pid() is not None
    except Exception:
        running = False
    running_str = (
        "[green]running[/]" if running else "[yellow]stopped[/]"
    )
    lines = [
        f"[bold]Provider:[/] [cyan]{cfg.provider.base_url}[/]",
        f"[bold]Model:[/]    [cyan]{cfg.provider.default_model}[/]",
        f"[bold]Proxy:[/]    127.0.0.1:{cfg.listen.port}  ({running_str})",
        "",
        "To manage settings / service / autostart:",
        "  [cyan]sieve[/]          — interactive menu",
        "  [cyan]sieve --help[/]   — all commands",
    ]
    console.print()
    console.print(
        Panel("\n".join(lines), title="Sieve is already installed",
              border_style="cyan")
    )


def _render_welcome(console) -> None:
    console.print()
    console.print(
        "[bold]Welcome.[/] This will set up Sieve in about 60 seconds.\n"
    )
    console.print(
        "[dim]Sieve sits between your agent and your LLM. Agents send it "
        "big payloads — system prompts, tool schemas, history. Sieve "
        "strips those, retrieves the relevant context, and forwards a "
        "lean payload. For stateless models it adds durable memory.[/]\n"
    )


# ── Step 1: where is your LLM? ───────────────────────────────────────


def _pick_llm_location(console) -> tuple[str, str | None]:
    """Return (provider_url, api_key). api_key is None when not needed
    (local Ollama).

    Five-branch picker covering every supported provider type:
      1. Anthropic (Claude) — cloud, requires API key
      2. OpenAI — cloud, requires API key
      3. OpenAI-compatible — anything speaking /v1/chat/completions
         (OpenRouter, Groq, Together, vLLM, llama.cpp, LM Studio, etc.)
      4. Ollama (local) — local daemon, no key
      5. Custom URL — bring-your-own endpoint and key

    Before showing the picker, we auto-detect the most likely match
    (env var keys, then a reachable local Ollama) and offer to use it
    as a one-keystroke happy path. The picker only appears when
    auto-detect fails or the user declines the suggestion.
    """
    from sieve._wizard_helpers import NumberedChoice, pick_numbered

    # ── Auto-detect happy path ─────────────────────────────────────
    # If we can confidently identify a provider, offer it inline.
    detected_url, detected_key = _auto_detect_provider()
    if detected_url:
        label = _describe_provider(detected_url, detected_key)
        if click.confirm(
            f"[?] Found {label}. Use it?",
            default=True,
        ):
            return detected_url, detected_key
        # User declined; fall through to picker.

    # ── Branch picker ──────────────────────────────────────────────
    choices = [
        NumberedChoice(
            label="Anthropic (Claude)",
            value="anthropic",
            help="Cloud API. Requires ANTHROPIC_API_KEY. Best quality for most users.",
        ),
        NumberedChoice(
            label="OpenAI",
            value="openai",
            help="Cloud API. Requires OPENAI_API_KEY. GPT-4, GPT-4o, o1, o3.",
        ),
        NumberedChoice(
            label="OpenAI-compatible endpoint",
            value="openai_compat",
            help="OpenRouter, Groq, Together, Mistral cloud, vLLM, llama.cpp, LM Studio…",
        ),
        NumberedChoice(
            label="Ollama (local)",
            value="ollama",
            help="Local Ollama daemon. No key required.",
        ),
        NumberedChoice(
            label="Custom URL",
            value="custom",
            help="Bring your own — we'll prompt for endpoint and key separately.",
        ),
    ]
    pick = pick_numbered(
        "Which kind of LLM are you using?",
        choices,
        default="ollama",
        console=console,
    )

    if pick == "anthropic":
        return _setup_anthropic(console)
    if pick == "openai":
        return _setup_openai(console)
    if pick == "openai_compat":
        return _setup_openai_compat(console)
    if pick == "ollama":
        return _setup_ollama(console)
    # custom
    return _setup_custom(console)


def _describe_provider(url: str, api_key: str | None) -> str:
    """Human-readable provider label for confirmation prompts."""
    low = url.lower()
    if "anthropic" in low:
        return "Anthropic (Claude) via ANTHROPIC_API_KEY"
    if "openai" in low:
        return "OpenAI via OPENAI_API_KEY"
    if "127.0.0.1" in low or "localhost" in low:
        return f"local Ollama at {url}"
    return url


def _setup_anthropic(console) -> tuple[str, str]:
    """Anthropic Claude — fixed URL, prompt for key if not in env."""
    url = "https://api.anthropic.com/v1"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        console.print(
            f"[green]Found ANTHROPIC_API_KEY in your environment[/] "
            "[dim](not displayed)[/]"
        )
    else:
        api_key = click.prompt(
            "Anthropic API key (sk-ant-…)",
            type=str,
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
        if not api_key:
            raise _InstallerExit(
                message="[yellow]Anthropic needs an API key. Cancelled.[/]",
                code=0,
            )
    return url, api_key


def _setup_openai(console) -> tuple[str, str]:
    """OpenAI — fixed URL, prompt for key if not in env."""
    url = "https://api.openai.com/v1"
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        console.print(
            f"[green]Found OPENAI_API_KEY in your environment[/] "
            "[dim](not displayed)[/]"
        )
    else:
        api_key = click.prompt(
            "OpenAI API key (sk-…)",
            type=str,
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
        if not api_key:
            raise _InstallerExit(
                message="[yellow]OpenAI needs an API key. Cancelled.[/]",
                code=0,
            )
    return url, api_key


def _setup_openai_compat(console) -> tuple[str, str | None]:
    """OpenAI-compatible — prompt for URL and optional key."""
    url = click.prompt(
        "Endpoint base URL (e.g. https://openrouter.ai/api/v1)",
        type=str,
    ).strip()
    if not url:
        raise _InstallerExit(
            message="[yellow]No URL given. Cancelled.[/]",
            code=0,
        )
    api_key: str | None = click.prompt(
        "API key (leave blank if none required)",
        type=str,
        hide_input=True,
        default="",
        show_default=False,
    ).strip() or None
    # Probe.
    console.print(f"[dim]Probing {url}…[/]", end="")
    if _reachable(url, api_key=api_key):
        console.print(" [green]reachable ✓[/]")
    else:
        console.print(" [yellow]not reachable / rejected auth[/]")
        if not click.confirm(
            "Save this endpoint anyway (fix later via Config)?",
            default=False,
        ):
            raise _InstallerExit(
                message="[yellow]Cancelled. No changes written.[/]",
                code=0,
            )
    return url, api_key


def _setup_ollama(console) -> tuple[str, None]:
    """Local or LAN Ollama. Probe the default first, then offer the LAN
    path if not reachable."""
    default_url = "http://127.0.0.1:11434"
    console.print(f"[dim]Probing {default_url}…[/]", end="")
    if _reachable(default_url):
        console.print(" [green]reachable ✓[/]")
        return default_url, None
    console.print(" [yellow]not reachable[/]")
    console.print(
        "[dim]Tip: install with [cyan]curl -fsSL https://ollama.com/install.sh | sh[/]"
        " then [cyan]ollama serve[/] in another terminal.[/]"
    )
    if click.confirm(
        "Use a LAN Ollama (different host) instead?",
        default=False,
    ):
        return _pick_lan_ollama(console), None
    # Local-but-not-yet-running path: let the user proceed.
    if click.confirm(
        f"Save {default_url} anyway (start Ollama before `sieve start`)?",
        default=True,
    ):
        return default_url, None
    raise _InstallerExit(
        message="[yellow]Cancelled.[/]",
        code=0,
    )


def _setup_custom(console) -> tuple[str, str | None]:
    """Custom URL — minimal probing, user is on their own."""
    url = click.prompt(
        "Custom endpoint base URL",
        type=str,
    ).strip()
    if not url:
        raise _InstallerExit(
            message="[yellow]No URL given. Cancelled.[/]",
            code=0,
        )
    api_key: str | None = click.prompt(
        "API key (leave blank if none required)",
        type=str,
        hide_input=True,
        default="",
        show_default=False,
    ).strip() or None
    return url, api_key


def _pick_lan_ollama(console) -> str:
    """Loop until we get a reachable LAN URL or the user confirms
    the unreachable one on purpose."""
    while True:
        host = click.prompt(
            "Host or IP (e.g. 192.168.1.100 or ollama.lan)",
            type=str,
        ).strip()
        if not host:
            console.print("[yellow]Empty host.[/]")
            continue
        # Be permissive about what they paste; normalise.
        url = _normalise_url(host, default_port=11434)
        console.print(f"[dim]Probing {url}…[/]", end="")
        if _reachable(url):
            console.print(" [green]reachable ✓[/]")
            return url
        console.print(" [yellow]not reachable[/]")
        if click.confirm(
            "Use it anyway (you can fix it later via Config)?",
            default=False,
        ):
            return url
        # Otherwise loop.


# ── Step 2: model ────────────────────────────────────────────────────


def _pick_model_step(
    console,
    url: str,
    api_key: str | None,
) -> str:
    from sieve._wizard_helpers import pick_model
    console.print()
    console.print("[bold]Pick a model[/]")
    console.print(f"[dim]Fetching available models from {url}…[/]")
    chosen = pick_model(
        "Which model should Sieve use by default?",
        base_url=url,
        default=None,
        console=console,
        api_key=api_key,
    )
    chosen = (chosen or "").strip()
    if not chosen:
        # User left it blank — use a provider-appropriate default.
        chosen = _default_model_for(url, api_key)
        console.print(f"[dim]Using default: {chosen}[/]")
    return chosen


def _default_model_for(url: str, api_key: str | None) -> str:
    """Pick a best-effort default model when the user doesn't name one.

    Provider-specific defaults take precedence (Anthropic and OpenAI
    don't have a generic /v1/models that's friendly to anonymous probes,
    so we hardcode known-good defaults). Other providers fall back to
    enumerating their model list.

    - Anthropic: claude-sonnet-4-5-20250929 (latest as of release)
    - OpenAI: gpt-4o-mini (lowest-cost competent default)
    - Local / LAN Ollama: try /api/tags, take the first non-embedding.
    - OpenAI-compat: try /v1/models, same.
    - Fail-fast if no models available.
    """
    low = url.lower()
    if "anthropic.com" in low:
        return "claude-sonnet-4-5-20250929"
    if "api.openai.com" in low:
        return "gpt-4o-mini"

    try:
        from sieve._wizard_helpers import list_models
        models = list_models(url, api_key=api_key)
    except TypeError:
        # Older list_models without api_key kwarg — gracefully degrade.
        from sieve._wizard_helpers import list_models as _lm
        models = _lm(url)
    except Exception:
        models = []
    # Filter out obvious embedding models (Sieve needs a chat model).
    def _is_chat(name: str) -> bool:
        low = name.lower()
        return not any(
            bad in low for bad in ("embed", "rerank", "nomic-embed")
        )
    for m in models:
        if _is_chat(m):
            return m
    # Fail-fast if no models available (audit C#9).
    raise RuntimeError(
        f"No models available at {url}. "
        f"Please pull a model first (e.g. `ollama pull qwen3:14b`) and retry `sieve init`."
    )


# ── Step 3: autostart ────────────────────────────────────────────────


def _pick_autostart(console, *, no_input: bool) -> bool:
    """Return whether to enable autostart after install.

    Default: Y on desktop sessions (USER != root), N on root/server
    sessions. If autostart isn't supported on this host, skip silently
    with a one-line note.
    """
    from sieve._autostart import autostart_supported
    if not autostart_supported():
        console.print(
            "\n[dim]Autostart-on-boot isn't supported on this system. "
            "You can start the proxy manually with `sieve` → Service.[/]"
        )
        return False
    if no_input:
        return False  # Conservative default for scripted / CI installs.

    running_as_root = (os.environ.get("USER") == "root")
    default_yes = not running_as_root
    console.print()
    console.print("[bold]Autostart on boot[/]")
    if running_as_root:
        console.print(
            "[dim]You're running as root. Autostart uses systemd user "
            "services, which require a logged-in session. Most server "
            "setups keep it disabled and start Sieve on demand.[/]"
        )
    else:
        console.print(
            "[dim]Sieve starts automatically each time you log in. "
            "Disable later via `sieve` → Service → Autostart.[/]"
        )
    return click.confirm(
        "Enable autostart on boot?",
        default=default_yes,
    )


# ── Step 4: start-now ────────────────────────────────────────────────


def _pick_start_now(console) -> bool:
    console.print()
    return click.confirm(
        "Start the Sieve proxy now?",
        default=True,
    )


# ── Plan preview + apply ─────────────────────────────────────────────


def _render_plan_preview(console, plan: InstallPlan) -> None:
    from rich.panel import Panel
    r = plan.redacted()
    lines = [
        f"[bold]Provider:[/]   [cyan]{r.provider_url}[/]",
    ]
    if r.provider_api_key:
        lines.append(f"[bold]API key:[/]    [cyan]{r.provider_api_key}[/] "
                     "[dim](saved to ~/.sieve/sieve.yaml, file mode 600)[/]")
    lines.extend([
        f"[bold]Model:[/]      [cyan]{r.model}[/]",
        f"[bold]Autostart:[/]  [cyan]{'enabled' if r.autostart else 'disabled'}[/]",
        f"[bold]Start now:[/]  [cyan]{'yes' if r.start_now else 'no'}[/]",
    ])
    console.print()
    console.print(
        Panel("\n".join(lines), title="About to apply",
              border_style="cyan")
    )


def _apply_plan(
    console,
    plan: InstallPlan,
    rollback: list[Callable[[], None]],
) -> None:
    """Execute the plan, registering rollback hooks at each step."""
    # 1. Write config. If anything later fails, remove it.
    console.print("\n[dim]Writing configuration…[/]", end="")
    _write_yaml(plan)
    _chmod_yaml()
    rollback.append(_remove_yaml)
    console.print(" [green]✓[/]")

    # 2. Download embedding model (idempotent — fastembed caches).
    #    Can take a minute on first install.
    console.print("[dim]Preparing embedding model "
                  "(BAAI/bge-small-en-v1.5, ~50MB on first install)…[/]")
    try:
        from fastembed import TextEmbedding
        _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        console.print("[green]Embedding model ready.[/]")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"FastEmbed setup failed: {exc}. "
            "Check your internet connection or Python environment."
        ) from exc

    # 3. Initialise the encrypted store. Idempotent — no-op if it exists.
    console.print("[dim]Initialising encrypted memory store…[/]", end="")
    _init_store()
    console.print(" [green]✓[/]")

    # 4. Autostart (if requested). Has its own rollback.
    if plan.autostart:
        console.print("[dim]Enabling autostart on boot…[/]", end="")
        from sieve._autostart import enable_autostart, disable_autostart
        ok, msg = enable_autostart()
        if ok:
            console.print(" [green]✓[/]")
            rollback.append(lambda: disable_autostart())
        else:
            console.print(f" [yellow]skipped — {msg}[/]")

    # 5. Start now (if requested).
    if plan.start_now:
        console.print("[dim]Starting the proxy…[/]")
        from sieve.cli import start as start_cmd
        try:
            start_cmd.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass  # click calls sys.exit(0) on success; not a failure
        except Exception as exc:  # noqa: BLE001
            # Don't roll back the install — the user can retry start
            # from the menu. But do surface what happened.
            console.print(
                f"[yellow]Start failed:[/] {exc}\n"
                "[dim]Config is saved. Run `sieve` → Service → Start "
                "to retry.[/]"
            )


def _write_yaml(plan: InstallPlan) -> None:
    """Write ~/.sieve/sieve.yaml with the planned values."""
    SIEVE_DIR.mkdir(parents=True, exist_ok=True)
    # We mirror the shape emitted by `sieve init` for consistency.
    provider_block = [
        "provider:",
        "  type: auto",
        f"  base_url: {plan.provider_url}",
        f"  default_model: {plan.model}",
    ]
    if plan.provider_api_key:
        provider_block.append(f"  api_key: {plan.provider_api_key}")
    yaml_text = "\n".join([
        "# Sieve configuration — written by sieve-install.",
        "# See https://llmsieve.dev for the full schema.",
        "",
        "listen:",
        "  host: 127.0.0.1",
        "  port: 11435",
        "",
        *provider_block,
        "",
        "embeddings:",
        "  provider: fastembed",
        "",
        "store:",
        "  path: ~/.sieve/memory.db",
        "",
    ])
    (SIEVE_DIR / "sieve.yaml").write_text(yaml_text)


def _chmod_yaml() -> None:
    """If we wrote an API key into sieve.yaml, tighten permissions."""
    yaml_path = SIEVE_DIR / "sieve.yaml"
    try:
        yaml_path.chmod(0o600)
    except Exception:
        pass  # non-POSIX filesystems; best-effort


def _remove_yaml() -> None:
    """Rollback hook for the config write step."""
    yaml_path = SIEVE_DIR / "sieve.yaml"
    try:
        yaml_path.unlink()
    except Exception:
        pass


def _init_store() -> None:
    """Initialise the encrypted memory store. Idempotent."""
    from sieve.config import RecallConfig
    from sieve.store import MemoryStore
    cfg = RecallConfig.load()
    ms = MemoryStore(cfg.store)
    if not ms.db_path.exists():
        ms.open()
        ms.init_schema()
        ms.close()
    else:
        # Store exists (reinstall / partial previous install) — open
        # it just to verify it's valid + schema-current.
        ms.open()
        if not ms.is_initialized():
            ms.init_schema()
        ms.close()


# ── Ready panel ──────────────────────────────────────────────────────


def _render_ready_panel(console, plan: InstallPlan) -> None:
    from rich.panel import Panel
    from sieve.config import RecallConfig
    cfg = RecallConfig.load()
    proxy_url = f"http://127.0.0.1:{cfg.listen.port}"
    try:
        from sieve.cli import _read_pid
        running = _read_pid() is not None
    except Exception:
        running = False
    running_str = "[green]yes[/]" if running else "[yellow]no[/]"
    from sieve._autostart import autostart_status
    as_state = autostart_status()

    r = plan.redacted()
    lines = [
        f"[bold]Provider[/]   [cyan]{r.provider_url}[/]",
    ]
    if r.provider_api_key:
        lines.append(f"[bold]API key[/]    [dim]saved to "
                     "~/.sieve/sieve.yaml (mode 600)[/]")
    lines.extend([
        f"[bold]Model[/]      [cyan]{r.model}[/]",
        f"[bold]Running[/]    {running_str}  "
        f"[dim](point your agent at {proxy_url})[/]",
        f"[bold]Autostart[/]  [cyan]{as_state}[/]",
        "",
        "[bold]Try it:[/]",
        f"  • [cyan]sieve demo[/]       — 6-turn scripted conversation",
        f"  • [cyan]sieve benchmark[/]  — measure Sieve's value with your model",
        f"  • [cyan]sieve[/]            — interactive management menu",
    ])
    console.print()
    console.print(
        Panel("\n".join(lines), title="Sieve is ready",
              border_style="green")
    )
    console.print()


# ── Probes ──────────────────────────────────────────────────────────


def _reachable(
    url: str,
    api_key: str | None = None,
    timeout: float = _PROBE_TIMEOUT_S,
) -> bool:
    """Is an LLM endpoint at ``url`` reachable and responsive?

    Probes, in order:
      1. ``/api/tags`` — Ollama native.
      2. ``/v1/models`` with optional bearer auth — OpenAI-compatible.

    Returns True on the first 2xx response. Swallows all exceptions
    (we want a bool, not a raise on weird network issues).
    """
    import httpx
    base = url.rstrip("/")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for path in ("/api/tags", "/v1/models"):
        try:
            r = httpx.get(f"{base}{path}", headers=headers, timeout=timeout)
            if 200 <= r.status_code < 300:
                return True
        except Exception:
            continue
    return False


def _normalise_url(raw: str, *, default_port: int) -> str:
    """Accept '192.168.1.100', 'ollama.lan:11434', 'https://x/', etc.

    Only adds ``:default_port`` for bare http (no scheme given, or
    scheme explicitly http). Leaves https URLs alone so we don't
    append ``:443`` to a clean cloud URL.
    """
    raw = raw.strip().rstrip("/")
    if "://" not in raw:
        raw = f"http://{raw}"
    # Only add a default port for http; https carries its own implicit 443.
    from urllib.parse import urlparse
    parsed = urlparse(raw)
    if parsed.scheme == "https":
        return raw
    if parsed.port is None:
        host = parsed.hostname or "127.0.0.1"
        path = parsed.path or ""
        return f"{parsed.scheme}://{host}:{default_port}{path}"
    return raw


def _api_key_from_env(url: str) -> str | None:
    """Pick an env var for a given URL."""
    low = url.lower()
    if "anthropic" in low:
        return os.environ.get("ANTHROPIC_API_KEY")
    if "openai" in low:
        return os.environ.get("OPENAI_API_KEY")
    return None


def _auto_detect_provider() -> tuple[str | None, str | None]:
    """Try to detect a usable provider without prompting the user.

    Returns (url, api_key) on success, (None, None) on failure.

    Priority order — first match wins:
      1. ANTHROPIC_API_KEY env var set → Anthropic API
      2. OPENAI_API_KEY env var set → OpenAI API
      3. Local Ollama on 127.0.0.1:11434 reachable → local Ollama

    The priority deliberately favours cloud keys over local Ollama: if
    the user has set an API key in their env, they almost certainly
    want to use that provider over an accidentally-running Ollama.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        url = "https://api.anthropic.com/v1"
        return url, os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        url = "https://api.openai.com/v1"
        return url, os.environ["OPENAI_API_KEY"]
    default_ollama = "http://127.0.0.1:11434"
    if _reachable(default_ollama):
        return default_ollama, None
    return None, None

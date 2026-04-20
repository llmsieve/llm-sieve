"""Top-level wizard — the user-facing menu shell.

Entered via ``sieve`` (no args + TTY) or ``sieve wizard``. Provides
a single, discoverable place for the everyday things a Sieve user
needs: install, service control, store inspection, config tweaks,
benchmark, demo, uninstall.

The menu framework in ``sieve/_menu.py`` handles rendering and
navigation. This module builds the screens and wires each option to
a handler — handlers are short closures that call into the existing
subcommands (reusing ``sieve start``, ``sieve stop``, etc.) so the
CLI behaviour never diverges between the menu and the flags.

Architecture note: the wizard ALWAYS returns the user to the
top-level screen after an action completes (unless they explicitly
quit). Nothing in here ``sys.exit()``s mid-flow; actions render
their results, pause for acknowledgement if there's real output,
then ``return BACK`` so the user sees the menu again. No-one ever
ends up at a blank prompt wondering what happened.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from sieve._menu import BACK, QUIT, MenuApp, MenuOption, MenuScreen, ResetTo
from sieve._branding import render_splash

logger = logging.getLogger("recall.wizard")

SIEVE_DIR = Path("~/.sieve").expanduser()


# ── Small helpers shared by multiple screens ───────────────────────────


def _pause_for_enter(console, label: str = "Press enter to return…") -> None:
    """Wait for the user to acknowledge output before clobbering it
    with the next menu render. Keeps the experience paged."""
    console.print(f"\n[dim]{label}[/]")
    try:
        click.prompt("", default="", show_default=False)
    except click.Abort:
        pass  # Ctrl-C here is fine — MenuApp's loop will catch it next.


def _is_installed() -> bool:
    """``~/.sieve/sieve.yaml`` exists? That's our simplest install
    signal; init writes it, uninstall removes it."""
    return (SIEVE_DIR / "sieve.yaml").exists()


def _has_store() -> bool:
    return (SIEVE_DIR / "memory.db").exists()


# ── Top screen ─────────────────────────────────────────────────────────


def build_top_screen(console) -> MenuScreen:
    """Root of the wizard. Options adapt to install state — the
    'Install' entry becomes 'Reinstall' after setup, and action
    entries that require an install become disabled until there
    is one."""
    installed = _is_installed()

    def _install_handler():
        return build_install_screen(console)

    def _service_handler():
        return build_service_screen(console)

    def _store_handler():
        return build_store_screen(console)

    def _config_handler():
        return build_config_screen(console)

    def _benchmark_handler():
        console.print(
            "\n[dim]Launching the benchmark in interactive mode. "
            "Ctrl-C at any time returns you here.[/]\n"
        )
        # Defer the import so the wizard module stays cheap to load.
        from sieve.cli import benchmark
        try:
            # Invoking the Click command without arguments triggers
            # the benchmark's own wizard (numbered prompts, live
            # model list). When that command exits normally it
            # sys.exit(0)s — catch SystemExit so we come back to
            # the menu instead of the whole process dying.
            benchmark.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Benchmark error:[/] {exc}")
        _pause_for_enter(console)
        return BACK

    def _demo_handler():
        console.print(
            "\n[dim]Running the 6-turn demo in a sandbox (main store "
            "untouched).[/]\n"
        )
        from sieve.cli import demo
        try:
            demo.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Demo error:[/] {exc}")
        _pause_for_enter(console)
        return BACK

    def _uninstall_handler():
        return build_uninstall_screen(console)

    def _quit_handler():
        return QUIT

    install_label = "Reinstall" if installed else "Install"
    install_help = (
        "Change provider / re-init everything. Current config is preserved as a backup."
        if installed
        else "Set up Sieve for the first time: provider, model, embedding, store."
    )

    needs_install_help = (
        "" if installed else "[dim](install Sieve first)[/]"
    )

    options = [
        MenuOption(
            label=install_label,
            handler=_install_handler,
            help=install_help,
        ),
        MenuOption(
            label="Service — start, stop, restart, autostart",
            handler=_service_handler,
            help=f"Run the Sieve proxy. {needs_install_help}".strip(),
            enabled=installed,
        ),
        MenuOption(
            label="Store — stats and inspection",
            handler=_store_handler,
            help=f"What facts / entities / episodes Sieve has learned. {needs_install_help}".strip(),
            enabled=installed and _has_store(),
        ),
        MenuOption(
            label="Config — adjust settings",
            handler=_config_handler,
            help=f"Change provider, model, or other settings. {needs_install_help}".strip(),
            enabled=installed,
        ),
        MenuOption(
            label="Benchmark — measure Sieve's value",
            handler=_benchmark_handler,
            help=f"Runs a 15-turn baseline-vs-Sieve comparison. {needs_install_help}".strip(),
            enabled=installed,
        ),
        MenuOption(
            label="Demo — 6-turn scripted conversation",
            handler=_demo_handler,
            help=f"Sandboxed; your main store is not touched. {needs_install_help}".strip(),
            enabled=installed,
        ),
        MenuOption(
            label="Uninstall",
            handler=_uninstall_handler,
            help=(
                "Stops the service, disables autostart, removes ~/.sieve."
                if installed
                else "Nothing installed to uninstall."
            ),
            enabled=installed,
        ),
        MenuOption(
            label="Quit",
            handler=_quit_handler,
            key="q",
        ),
    ]
    subtitle = (
        "[dim]Pick a number, or press [cyan]q[/] to quit. "
        "Use [cyan]b[/] to go back from any submenu.[/]"
    )
    return MenuScreen(
        title="Sieve",
        subtitle=subtitle,
        options=options,
        allow_back=False,  # can't back out of root
        allow_quit=True,
    )


# ── Install screen ─────────────────────────────────────────────────────


def build_install_screen(console) -> MenuScreen:
    """Two install paths, plus cancel/back."""

    def _quick():
        return _run_quick_install(console)

    def _guided():
        return _run_guided_install(console)

    subtitle = (
        "[dim]Quick = zero prompts, best for a local Ollama. "
        "Guided = step-by-step, explains each choice.[/]"
    )
    return MenuScreen(
        title="Install Sieve",
        subtitle=subtitle,
        options=[
            MenuOption(
                label="Quick — accept all defaults (recommended for local Ollama)",
                handler=_quick,
                help="Provider: http://127.0.0.1:11434  ·  Embedder: FastEmbed  ·  Store: ~/.sieve/memory.db",
            ),
            MenuOption(
                label="Guided — walk through each setting with explanations",
                handler=_guided,
                help="Pick provider URL, model, store path, autostart, pricing-tier default.",
            ),
        ],
    )


_DEFAULT_PROVIDER_URL = "http://127.0.0.1:11434"


def _probe_provider(url: str, timeout: float = 2.0) -> bool:
    """True iff ``{url}/api/tags`` responds 200 within ``timeout``.

    Short timeout keeps the Quick-install happy-path fast; we only
    wait longer when the default is actually unreachable.
    """
    import httpx
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _run_quick_install(console):
    """Lazy install: zero prompts in the happy path; one targeted
    prompt when the default Ollama endpoint isn't reachable, so LAN
    users don't silently end up with a broken config.
    """
    console.print(
        "\n[bold]Quick install[/]  —  provider: "
        f"{_DEFAULT_PROVIDER_URL}, FastEmbed embedder, encrypted "
        "store at ~/.sieve/memory.db.\n"
    )
    if _is_installed():
        reinstall = click.confirm(
            "Sieve is already installed. Reinstall (preserves store, "
            "refreshes config)?",
            default=False,
        )
        if not reinstall:
            console.print("[yellow]Cancelled.[/]")
            _pause_for_enter(console)
            return BACK

    # Probe the default. If unreachable, give the user ONE chance to
    # point at their real Ollama — otherwise we'd write a URL they
    # can't reach and every benchmark/demo turn would 502.
    provider_url = _DEFAULT_PROVIDER_URL
    console.print(
        f"[dim]Probing {_DEFAULT_PROVIDER_URL}…[/]",
        end="",
    )
    reachable = _probe_provider(_DEFAULT_PROVIDER_URL)
    if reachable:
        console.print(" [green]reachable ✓[/]")
    else:
        console.print(" [yellow]not reachable[/]")
        console.print(
            "\n[bold]Ollama isn't running at the default address.[/]\n"
            "[dim]Common cases:\n"
            "  · Ollama is on another host on your network → enter its URL\n"
            "  · Ollama isn't installed or isn't started yet → start it\n"
            "    then press enter to use the default anyway[/]"
        )
        user_url = click.prompt(
            "Provider URL",
            default=_DEFAULT_PROVIDER_URL,
            show_default=True,
        ).strip()
        if user_url and user_url != _DEFAULT_PROVIDER_URL:
            console.print(f"[dim]Probing {user_url}…[/]", end="")
            if _probe_provider(user_url):
                console.print(" [green]reachable ✓[/]")
                provider_url = user_url
            else:
                console.print(" [yellow]not reachable either[/]")
                go = click.confirm(
                    "Write this URL anyway (you can fix it later via the "
                    "Config menu)?",
                    default=False,
                )
                if go:
                    provider_url = user_url
                else:
                    console.print("[yellow]Install cancelled.[/]")
                    _pause_for_enter(console)
                    return BACK

    from sieve.cli import init as init_cmd
    try:
        args = ["--provider", provider_url]
        if _is_installed():
            args.append("--force")
        init_cmd.main(standalone_mode=False, args=args)
    except (SystemExit, click.exceptions.Exit):
        pass
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Install failed:[/] {exc}")
        _pause_for_enter(console)
        return BACK
    _render_post_install_status(console)
    _offer_to_start_service(console)
    _pause_for_enter(console)
    # Rebuild the top screen from a fresh install state — otherwise
    # the user navigates back to a stale "Install everything's
    # unavailable" menu despite having just installed.
    return ResetTo(build_top_screen(console))


def _run_guided_install(console):
    """Guided install: prompt for each setting with explanations.

    Wraps `sieve init --wizard`'s underlying logic but with our
    numbered-picker UX and clearer inline copy. The existing wizard
    module does the heavy lifting; we just drive it.
    """
    console.print(
        "\n[bold]Guided install[/]\n"
        "[dim]I'll ask a few questions. Press enter at any prompt to "
        "accept the shown default.[/]\n"
    )
    from sieve.cli_wizard import run_wizard, WizardContext, apply_wizard_answers
    from sieve._wizard_helpers import pick_model, NumberedChoice, pick_numbered

    # Reuse the existing wizard's prompter/probe protocol but
    # substitute our numbered helpers for the model + provider steps.
    # NOTE: the existing wizard runs its own click.prompts — calling
    # it here preserves backwards compatibility for every setting we
    # haven't re-themed yet. The plan is to migrate it incrementally;
    # for this phase we just re-use it verbatim and add a "what's
    # next" panel.
    import httpx

    class _ClickPrompter:
        def prompt(self, q, default=None):
            return click.prompt(
                q, default=default if default is not None else "",
                show_default=bool(default),
            )

        def confirm(self, q, default=True):
            return click.confirm(q, default=default)

    class _HttpxProbe:
        def check(self, url):
            try:
                r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
                if r.status_code == 200:
                    data = r.json() or {}
                    models = [
                        m.get("name", "") for m in data.get("models", [])
                        if m.get("name")
                    ]
                    return True, models
            except Exception:
                pass
            return False, []

    def _download():
        console.print("Downloading embedding model (BAAI/bge-small-en-v1.5, ~50MB)…")
        from fastembed import TextEmbedding
        _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        console.print("[green]Embedding model ready.[/]")

    ctx = WizardContext(
        prompter=_ClickPrompter(),
        probe=_HttpxProbe(),
        download_model=_download,
    )
    try:
        answers = run_wizard(ctx)
    except KeyboardInterrupt:
        console.print("\n[yellow]Install cancelled.[/]")
        _pause_for_enter(console)
        return BACK
    if not answers.confirmed:
        console.print("[yellow]Cancelled; no changes made.[/]")
        _pause_for_enter(console)
        return BACK

    _download()
    apply_wizard_answers(answers)

    # Initialise the encrypted store using the FIX from earlier in
    # this session (load resolved config, not bare StoreConfig).
    from sieve.config import RecallConfig
    from sieve.store import MemoryStore
    cfg = RecallConfig.load()
    ms = MemoryStore(cfg.store)
    if not ms.db_path.exists():
        ms.open()
        ms.init_schema()
        ms.close()
        console.print(
            f"[green]Initialised encrypted store at[/] [cyan]{ms.db_path}[/]"
        )

    _render_post_install_status(console)
    _offer_to_start_service(console)
    _pause_for_enter(console)
    # Reset to a fresh top screen so the newly-installed state is
    # reflected. See the comment on the quick-install path.
    return ResetTo(build_top_screen(console))


def _offer_to_start_service(console) -> None:
    """After a fresh install, ask if the user wants the proxy running now.

    Default Yes — the user just installed Sieve, so starting the
    proxy is almost always what they want. Skipping still leaves
    them with a clearly-reachable Service → Start option in the
    top menu.

    If the service is ALREADY running (edge case: reinstall without
    a prior stop), we print a note and skip the prompt.
    """
    from sieve.cli import _read_pid
    if _read_pid() is not None:
        console.print(
            "\n[dim]Sieve is already running; no need to start.[/]"
        )
        return
    console.print()
    start_now = click.confirm(
        "Start the Sieve proxy now?",
        default=True,
    )
    if not start_now:
        console.print(
            "[dim]You can start it later from "
            "Service → Start, or with `sieve start`.[/]"
        )
        return
    from sieve.cli import start as start_cmd
    try:
        start_cmd.main(standalone_mode=False, args=[])
    except (SystemExit, click.exceptions.Exit):
        pass
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]Start failed:[/] {exc}\n"
            "[dim]You can retry from Service → Start.[/]"
        )


def _render_post_install_status(console) -> None:
    """Show the user what they just set up and what to do next."""
    from rich.panel import Panel
    from sieve.config import RecallConfig
    try:
        cfg = RecallConfig.load()
    except Exception as exc:
        console.print(f"[red]Config load failed:[/] {exc}")
        return
    lines = [
        f"[bold]Provider:[/]  [cyan]{cfg.provider.base_url}[/]",
        f"[bold]Model:[/]     [cyan]{cfg.provider.default_model}[/]",
        f"[bold]Embedder:[/]  [cyan]{cfg.embeddings.provider}[/]",
        f"[bold]Store:[/]     [cyan]{cfg.store.path}[/]",
        "",
        "[bold]What's next:[/]",
        "  • Start the proxy:     Service → Start",
        "  • Try the demo:        Demo",
        "  • Measure the value:   Benchmark",
        "",
        "[dim]At any time, rerun 'sieve' or 'sieve wizard' to return here.[/]",
    ]
    console.print(Panel("\n".join(lines), title="Sieve installed", border_style="green"))


# ── Service screen ─────────────────────────────────────────────────────


def build_service_screen(console) -> MenuScreen:
    """Live service control. Status is rendered on each visit so
    stop/start state is always current.

    NOTE: Phase 3a adds autostart-on-boot; this screen stubs the
    option with 'coming soon' until that's wired in.
    """
    from sieve.cli import _read_pid

    pid = _read_pid()
    running = pid is not None

    def _start():
        from sieve.cli import start as start_cmd
        try:
            start_cmd.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Start failed:[/] {exc}")
        _pause_for_enter(console)
        return build_service_screen(console)  # refresh

    def _stop():
        from sieve.cli import stop as stop_cmd
        try:
            stop_cmd.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Stop failed:[/] {exc}")
        _pause_for_enter(console)
        return build_service_screen(console)

    def _restart():
        from sieve.cli import stop as stop_cmd, start as start_cmd
        try:
            stop_cmd.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        try:
            start_cmd.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        _pause_for_enter(console)
        return build_service_screen(console)

    def _status():
        from sieve.cli import status as status_cmd
        try:
            status_cmd.main(standalone_mode=False, args=[])
        except (SystemExit, click.exceptions.Exit):
            pass
        _pause_for_enter(console)
        return BACK

    def _autostart():
        # Phase 3a will replace this handler with the real thing.
        return _autostart_screen(console)

    status_line = (
        f"[green]● Running[/] (PID {pid})" if running
        else "[yellow]○ Stopped[/]"
    )
    subtitle = f"Status: {status_line}"

    options = [
        MenuOption(
            label="Start",
            handler=_start,
            help="Spawn the proxy in the background.",
            enabled=not running,
        ),
        MenuOption(
            label="Stop",
            handler=_stop,
            help="Stop the running proxy.",
            enabled=running,
        ),
        MenuOption(
            label="Restart",
            handler=_restart,
            help="Stop, then start — useful after config changes.",
            enabled=running,
        ),
        MenuOption(
            label="Status — full details",
            handler=_status,
            help="Version, port, store stats, active phase.",
        ),
        MenuOption(
            label="Autostart on boot",
            handler=_autostart,
            help="Control whether Sieve starts automatically at login.",
        ),
    ]
    return MenuScreen(
        title="Service",
        subtitle=subtitle,
        options=options,
    )


def _autostart_screen(console) -> MenuScreen:
    """Phase 3a stub. Replaced by real systemd wiring in that task."""
    def _enable():
        from sieve._autostart import enable_autostart, autostart_supported, autostart_status
        if not autostart_supported():
            console.print(
                "[yellow]Autostart not supported on this system.[/] "
                "Sieve's autostart currently requires systemd user "
                "services (Linux)."
            )
            _pause_for_enter(console)
            return BACK
        ok, msg = enable_autostart()
        if ok:
            console.print(f"[green]{msg}[/]")
        else:
            console.print(f"[red]{msg}[/]")
        _pause_for_enter(console)
        return BACK

    def _disable():
        from sieve._autostart import disable_autostart, autostart_supported
        if not autostart_supported():
            console.print("[yellow]Autostart not supported on this system.[/]")
            _pause_for_enter(console)
            return BACK
        ok, msg = disable_autostart()
        console.print(f"[{'green' if ok else 'red'}]{msg}[/]")
        _pause_for_enter(console)
        return BACK

    from sieve._autostart import autostart_status, autostart_supported
    status_str = autostart_status()
    supported = autostart_supported()
    subtitle = f"Current: [cyan]{status_str}[/]"
    if not supported:
        subtitle += "  [dim](systemd not available on this system)[/]"

    return MenuScreen(
        title="Autostart",
        subtitle=subtitle,
        options=[
            MenuOption(
                label="Enable — start Sieve at login",
                handler=_enable,
                enabled=supported,
            ),
            MenuOption(
                label="Disable",
                handler=_disable,
                enabled=supported,
            ),
        ],
    )


# ── Store screen (shallow) ─────────────────────────────────────────────


def build_store_screen(console) -> MenuScreen:
    """Shallow store inspector. Shows counts on render; sub-options
    list the last N of each entity type. Deep navigation (search,
    paginate) ships later."""
    from sieve.config import RecallConfig
    from sieve.store import MemoryStore

    try:
        cfg = RecallConfig.load()
        ms = MemoryStore(cfg.store)
        if ms.db_path.exists():
            ms.open()
            stats = ms.stats()
            ms.close()
        else:
            stats = {}
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Store unavailable:[/] {exc}")
        _pause_for_enter(console)
        return BACK

    subtitle = (
        f"[cyan]{stats.get('facts_count', 0)}[/] facts · "
        f"[cyan]{stats.get('entities_count', 0)}[/] entities · "
        f"[cyan]{stats.get('episodes_count', 0)}[/] episodes · "
        f"[cyan]{stats.get('db_size_kb', 0):.0f}KB[/] on disk"
    )

    def _call_store_cmd(name: str, args: list[str]):
        """Dispatch to a `sieve store <name>` subcommand.

        The commands are defined in sieve/cli.py as `store_*_cmd`,
        registered on the `store` click group. We resolve them
        lazily to avoid a circular import at module load time.
        """
        from sieve import cli as cli_mod
        cmd = getattr(cli_mod, name, None)
        if cmd is None:
            console.print(f"[red]Command '{name}' not found[/]")
            return
        try:
            cmd.main(standalone_mode=False, args=args)
        except (SystemExit, click.exceptions.Exit):
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{name} failed:[/] {exc}")

    def _stats_detail():
        _call_store_cmd("store_stats_cmd", [])
        _pause_for_enter(console)
        return BACK

    def _facts():
        _call_store_cmd("store_facts_cmd", ["--limit", "10"])
        _pause_for_enter(console)
        return BACK

    def _entities():
        _call_store_cmd("store_entities_cmd", ["--limit", "10"])
        _pause_for_enter(console)
        return BACK

    def _episodes():
        _call_store_cmd("store_episodes_cmd", ["--limit", "10"])
        _pause_for_enter(console)
        return BACK

    return MenuScreen(
        title="Store",
        subtitle=subtitle,
        options=[
            MenuOption("Full stats table", _stats_detail),
            MenuOption("Last 10 facts", _facts),
            MenuOption("Last 10 entities", _entities),
            MenuOption("Last 10 episodes", _episodes),
        ],
    )


# ── Config screen ──────────────────────────────────────────────────────


# Settings that are safe to edit freely from the menu. Anything NOT
# in this allowlist requires either a direct sieve.yaml edit or a
# double-confirm flow (see _DANGEROUS_SETTINGS).
_SAFE_SETTINGS = {
    # path              -> (help, default-hint)
    "provider.default_model": (
        "Model sent to the LLM endpoint by default.",
        "From provider.default_model in sieve.yaml.",
    ),
    "pipeline.conversation_turns": (
        "How many prior turns to keep in context during ACCUMULATE.",
        "Integer; default 3.",
    ),
    "progression.phase_1_threshold": (
        "Fact count at which Sieve leaves OBSERVE and enters ACCUMULATE.",
        "Integer; default 20.",
    ),
    "progression.phase_2_threshold": (
        "Fact count at which Sieve enters ACTIVATE.",
        "Integer; default 50.",
    ),
    "tools.compression": (
        "How aggressively to compress agent tool schemas. "
        "moderate / aggressive / off.",
        "Default: moderate.",
    ),
}

# Settings where misconfiguration causes data-loss-or-worse. The
# menu marks them read-only; users can edit them by hand in
# ~/.sieve/sieve.yaml, where the consequences are at least visible.
_DANGEROUS_SETTINGS = {
    "provider.base_url": (
        "Changing this after the store has data could silently "
        "fail every retrieval — different providers imply different "
        "embeddings. Edit ~/.sieve/sieve.yaml directly if you know what you're doing."
    ),
    "store.embedding_dimensions": (
        "The vec tables are created at init with this dim and it "
        "can't be changed live — would require a store sterilisation."
    ),
    "security.auth_token": (
        "Auto-managed. Changing by hand can lock out the CLI."
    ),
    "embeddings.provider": (
        "Switching embedders after the store has facts invalidates "
        "every vector. Requires a fresh init."
    ),
}


def _store_fact_count() -> int:
    """Return the store's current fact count, or 0 if the store isn't
    accessible yet. Used to decide whether provider.base_url is safe
    to edit from the menu — zero facts means no cached embeddings
    that would be invalidated by a provider switch."""
    from sieve.config import RecallConfig
    from sieve.store import MemoryStore
    try:
        cfg = RecallConfig.load()
        ms = MemoryStore(cfg.store)
        if not ms.db_path.exists():
            return 0
        ms.open()
        try:
            return int(ms.stats().get("facts_count", 0))
        finally:
            ms.close()
    except Exception:
        return 0


def build_config_screen(console) -> MenuScreen:
    from sieve.config import RecallConfig
    try:
        cfg = RecallConfig.load()
    except Exception as exc:
        console.print(f"[red]Config load failed:[/] {exc}")
        _pause_for_enter(console)
        return BACK

    facts = _store_fact_count()
    # When the store is empty we can safely allow provider.base_url
    # edits from the menu — there are no cached embeddings to
    # invalidate. Once the store grows, switching providers changes
    # the embedding space and silently breaks retrieval.
    base_url_editable = (facts == 0)

    def _getter(path: str):
        obj = cfg
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                return None
        return obj

    def _probe_url(url: str, timeout: float = 2.0) -> bool:
        import httpx
        try:
            r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def _make_setting_handler(path: str, help_text: str, default_hint: str):
        def _handler():
            current = _getter(path)
            console.print(
                f"\n[bold]{path}[/]\n"
                f"[dim]{help_text}[/]\n"
                f"[dim]{default_hint}[/]\n\n"
                f"Current value: [cyan]{current}[/]\n"
            )
            # provider.default_model gets the live-model picker, same
            # UX as the benchmark wizard. A text prompt for a value
            # that should come from a list is a footgun — typos silently
            # write an unusable model name.
            if path == "provider.default_model":
                from sieve._wizard_helpers import pick_model
                endpoint = _getter("provider.base_url") or ""
                new = pick_model(
                    "Pick a model",
                    base_url=endpoint,
                    default=str(current) if current else None,
                    console=console,
                )
                new = (new or "").strip()
            else:
                new = click.prompt(
                    "New value (enter to keep current)",
                    default=str(current) if current is not None else "",
                    show_default=False,
                ).strip()
            if not new or new == str(current):
                console.print("[dim]Unchanged.[/]")
                _pause_for_enter(console)
                return BACK
            # provider.base_url gets a reachability probe — still
            # write if it fails, but surface it so the user knows
            # why their next request might 502.
            if path == "provider.base_url":
                console.print(f"[dim]Probing {new}…[/]", end="")
                if _probe_url(new):
                    console.print(" [green]reachable ✓[/]")
                else:
                    console.print(" [yellow]not reachable[/]")
                    if not click.confirm(
                        "Write this URL anyway?", default=False
                    ):
                        console.print("[yellow]Cancelled.[/]")
                        _pause_for_enter(console)
                        return BACK
            from sieve import cli_config as cc
            try:
                data = cc.load_raw()
                cc.set_path(data, path, new)
                cc.validate_raw(data)
                cc.write_raw(data)
                console.print(
                    f"[green]Updated {path} = {new}[/]\n"
                    f"[dim]Restart the service for changes to take effect.[/]"
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Update failed:[/] {exc}")
            _pause_for_enter(console)
            return BACK
        return _handler

    def _show_dangerous():
        console.print(
            "\n[bold]Settings that require direct ~/.sieve/sieve.yaml edits:[/]\n"
        )
        dangerous_view = dict(_DANGEROUS_SETTINGS)
        if not base_url_editable:
            # Make the reason specific — it's dangerous BECAUSE the
            # store already has data.
            dangerous_view["provider.base_url"] = (
                f"Currently hidden because the store has {facts} facts. "
                "Switching providers invalidates every embedding. "
                "Edit ~/.sieve/sieve.yaml if you've sterilised the store."
            )
        for path, reason in dangerous_view.items():
            console.print(f"  [yellow]{path}[/]")
            console.print(f"    [dim]{reason}[/]")
        _pause_for_enter(console)
        return BACK

    options = []
    # Build the editable list. If the store is empty, prepend
    # provider.base_url so it's the first thing a fresh user sees
    # (they're most likely to need it).
    editable = dict(_SAFE_SETTINGS)
    if base_url_editable:
        editable = {
            "provider.base_url": (
                "URL of the LLM endpoint Sieve forwards requests to.",
                "Safe to edit now because the store is empty. "
                "Restart the service after any change.",
            ),
            **_SAFE_SETTINGS,
        }

    for path, (help_text, default_hint) in editable.items():
        current = _getter(path)
        options.append(MenuOption(
            label=f"{path}  =  {current}",
            handler=_make_setting_handler(path, help_text, default_hint),
            help=help_text,
        ))
    options.append(MenuOption(
        label="View settings the menu doesn't edit (and why)",
        handler=_show_dangerous,
        help="Dangerous or init-only settings.",
    ))

    subtitle = (
        "[dim]Safe-to-edit settings are listed below. "
        "Restart the service after any change.[/]"
    )
    if base_url_editable:
        subtitle += (
            "\n[dim]Store is empty — provider.base_url is editable "
            "until you learn your first facts.[/]"
        )
    return MenuScreen(
        title="Config",
        subtitle=subtitle,
        options=options,
    )


# ── Uninstall screen ───────────────────────────────────────────────────


def build_uninstall_screen(console) -> MenuScreen:
    """Uninstall is high-consequence; present a preview + double-
    confirm before anything is removed."""
    from sieve._autostart import autostart_supported, autostart_status
    preview = [
        "[yellow]Uninstalling Sieve will:[/]",
        "  • Stop the running proxy (if any)",
        "  • Disable autostart (if enabled)",
        "  • Remove ~/.sieve/ (config, store, keys, logs)",
        "",
        "[dim]The llm-sieve Python package stays installed — "
        "`pip uninstall llm-sieve` removes that separately.[/]",
    ]

    def _confirm_and_run():
        console.print()
        for line in preview:
            console.print(line)
        console.print()
        first = click.confirm("Proceed with uninstall?", default=False)
        if not first:
            console.print("[green]Cancelled.[/]")
            _pause_for_enter(console)
            return BACK
        # Rich markup can't render inside click.prompt, so print the
        # instruction with rich first, then prompt with plain text.
        console.print("Type [red]yes[/] to confirm (anything else cancels).")
        really = click.prompt(
            "Confirm",
            default="",
            show_default=False,
        )
        if really.strip().lower() != "yes":
            console.print("[green]Cancelled.[/]")
            _pause_for_enter(console)
            return BACK

        # Stop the service + disable autostart + remove ~/.sieve.
        from sieve.cli import _read_pid
        import os
        import signal
        import shutil

        pid = _read_pid()
        if pid is not None:
            console.print("Stopping the proxy…")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Couldn't stop PID {pid}: {exc}[/]")

        if autostart_supported() and autostart_status() == "enabled":
            console.print("Disabling autostart…")
            from sieve._autostart import disable_autostart
            ok, msg = disable_autostart()
            console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

        console.print(f"Removing [cyan]{SIEVE_DIR}[/]…")
        try:
            shutil.rmtree(SIEVE_DIR, ignore_errors=True)
            console.print("[green]Sieve uninstalled.[/]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Removal failed:[/] {exc}")
        _pause_for_enter(console, label="Press enter to exit the wizard…")
        return QUIT

    return MenuScreen(
        title="Uninstall Sieve",
        subtitle="[yellow]High-consequence — requires double confirmation.[/]",
        options=[
            MenuOption(
                label="Preview + confirm + uninstall",
                handler=_confirm_and_run,
                help="Stop service, disable autostart, remove ~/.sieve/",
            ),
        ],
    )


# ── Public entry point ────────────────────────────────────────────────


def run_wizard(console=None, clear_between_screens: bool = False) -> None:
    """Render the splash, build the top screen, enter the loop.

    The CLI calls this from:
    - ``sieve`` with no args + TTY
    - ``sieve wizard`` (explicit)

    We deliberately do NOT clear the screen between menu transitions.
    Clearing wipes the splash after the first redraw and also fights
    terminal affordances (scrollback, copy-paste). Most mature CLI
    wizards (`gh`, `npm init`, `pip`) scroll naturally; we match that.
    """
    if console is None:
        from rich.console import Console
        console = Console()
    render_splash(console)
    app = MenuApp(
        initial=build_top_screen(console),
        console=console,
        clear_between_screens=clear_between_screens,
    )
    app.run()

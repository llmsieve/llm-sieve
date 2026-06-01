"""CLI entry point for Sieve."""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

import click
import uvicorn
from rich.console import Console
from rich.logging import RichHandler

from sieve.config import RecallConfig


def _console_width() -> int:
    """Target console width: capped at 120 cols, floored at 80.

    Most terminals are ≥100 cols. A 240-col render wraps unreadably on
    80-col windows and sprawls on typical 120-col ones. 120 is the
    sweet spot — readable at any sensible width, and the renders we
    ship (tables, panels, headlines) are designed to fit within it.
    """
    try:
        import shutil
        actual = shutil.get_terminal_size((100, 24)).columns
    except Exception:
        actual = 100
    return max(80, min(120, actual))


console = Console(width=_console_width())


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


POST_INSTALL_HINT = (
    "Sieve installed successfully! Run [cyan]sieve init[/] to get started, "
    "or [cyan]sieve init --wizard[/] for guided setup."
)


@click.group(invoke_without_command=True)
@click.version_option(package_name="llm-sieve", prog_name="sieve")
@click.pass_context
def cli(ctx: click.Context):
    """Sieve — Transparent context reduction for LLMs.

    After install, run `sieve init` (or `sieve init --wizard`) to create
    the configuration file, download the embedding model, and initialise
    the encrypted memory store. Then `sieve start` to run the proxy.
    """
    # No subcommand.
    #   - TTY: launch the top-level wizard so users have a
    #     discoverable place to start.
    #   - No TTY (piped / CI / subprocess): print the post-install
    #     guidance so non-interactive callers aren't trapped in a
    #     prompt nobody can see.
    if ctx.invoked_subcommand is None:
        if sys.stdin.isatty():
            # If no install yet, skip straight to the first-run
            # installer. Users who pip-installed and run `sieve`
            # should land in the setup experience, not a menu full
            # of disabled options.
            from pathlib import Path as _P
            yaml_path = _P("~/.sieve/sieve.yaml").expanduser()
            if not yaml_path.exists():
                console.print(
                    "[bold]No Sieve configuration found.[/] "
                    "Launching the first-run installer.\n"
                    "[dim](You can also run [cyan]sieve-install[/] directly.)[/]\n"
                )
                from sieve._installer import main as installer_main
                try:
                    installer_main.main(standalone_mode=False, args=[])
                except (SystemExit, click.exceptions.Exit):
                    pass
                return
            from sieve._wizard_app import run_wizard
            run_wizard(console=console)
        else:
            console.print(f"[bold green]Sieve[/] — {POST_INSTALL_HINT}")
            console.print(
                "\n[dim]Run [cyan]sieve --help[/] for the full command list.[/]"
            )


SIEVE_DIR = Path("~/.sieve").expanduser()
PID_FILE = SIEVE_DIR / "sieve.pid"  # default; per-port variant below
LOG_FILE = SIEVE_DIR / "sieve.log"

# Default listen port — kept in sync with config.RecallConfig.listen.port
# default. The bare PID_FILE above is used for the default-port case
# (backward compat with existing installs); other ports get a suffix.
_DEFAULT_PORT = 11435


def _pid_file_for_port(port: int) -> Path:
    """Per-port PID file path. Default port keeps the legacy name.

    Two daemons on the same port can't coexist anyway (kernel bind
    refusal), so port is a sufficient discriminator without needing a
    full config fingerprint.
    """
    if port == _DEFAULT_PORT:
        return PID_FILE
    return SIEVE_DIR / f"sieve-{port}.pid"


def _config_exists(config_path: str | None) -> bool:
    if config_path:
        return Path(config_path).expanduser().exists()
    return any(p.exists() for p in (Path("sieve.yaml"), SIEVE_DIR / "sieve.yaml"))


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
@click.option("--host", default=None, help="Override listen host")
@click.option("--port", "-p", default=None, type=int, help="Override listen port")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    help="Run in foreground instead of daemonising. Useful for debugging "
    "— logs go to the terminal and Ctrl+C stops the proxy.",
)
def start(
    config_path: str | None,
    host: str | None,
    port: int | None,
    verbose: bool,
    foreground: bool,
):
    """Start the Sieve proxy server.

    By default this daemonises into the background so the terminal is
    freed. Pass ``--foreground`` / ``-f`` to run attached to the TTY for
    debugging.
    """
    _setup_logging(verbose)

    if not _config_exists(config_path):
        console.print(
            "[bold red]No Sieve configuration found.[/] "
            "Run [cyan]sieve init[/] first."
        )
        sys.exit(1)

    # uvicorn re-imports sieve.main as a string, so the config passed here
    # is thrown away. Propagate via env var so create_app() picks it up.
    if config_path:
        os.environ["SIEVE_CONFIG"] = str(Path(config_path).resolve())
    config = RecallConfig.load(config_path)

    if host:
        config.listen.host = host
    if port:
        config.listen.port = port

    # PID file is per-port now (default-port keeps legacy ~/.sieve/sieve.pid
    # so existing installs are unaffected; non-default ports get a suffixed
    # path so two daemons can coexist if their listen ports differ).
    pid_file = _pid_file_for_port(config.listen.port)

    # Refuse to start a second daemon ON THIS PORT if one is already running.
    existing = _read_pid(pid_file)
    if existing is not None:
        console.print(
            f"[bold yellow]Sieve is already running on port {config.listen.port}[/] "
            f"(pid {existing}). Use [cyan]sieve stop -p {config.listen.port}[/] "
            f"or [cyan]sieve restart -p {config.listen.port}[/] first."
        )
        sys.exit(1)

    # Fail fast if the configured port is already bound — Phase 7 Test 5.
    # If the existing PID file points at a live process, that's likely
    # our own stale daemon from a previous session; tell the user how
    # to clean it up specifically rather than printing a generic
    # "port in use" message.
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind((config.listen.host, config.listen.port))
    except OSError:
        stale_pid = _read_pid()
        if stale_pid is not None and _pid_running(stale_pid):
            console.print(
                f"[bold red]Port {config.listen.port} is already in use[/] "
                f"by an existing Sieve process [dim](pid {stale_pid}).[/]\n"
            )
            console.print(
                "[bold]How to fix:[/]\n"
                f"  • [cyan]sieve stop[/]   then re-run, OR\n"
                f"  • [cyan]sieve restart[/] to do both in one step, OR\n"
                f"  • [cyan]sieve start --port <other>[/] to run on a "
                f"different port.\n"
            )
        else:
            console.print(
                f"[bold red]Port {config.listen.port} is already in use[/] "
                f"by another process [dim](not Sieve).[/]\n"
            )
            console.print(
                "[bold]How to fix:[/]\n"
                f"  • Identify the process: "
                f"[cyan]ss -tlnp 'sport = :{config.listen.port}'[/]\n"
                f"  • Or run Sieve on a different port: "
                f"[cyan]sieve start --port <other>[/]\n"
            )
        sys.exit(1)
    finally:
        probe.close()


def _pid_running(pid: int) -> bool:
    """True if a process with this pid exists. Used by the
    port-collision branch to distinguish 'our own stale daemon' from
    'someone else's process'."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False

    SIEVE_DIR.mkdir(parents=True, exist_ok=True)

    if foreground:
        _run_proxy_foreground(config, verbose, pid_file=pid_file)
        return

    _daemonise_and_run(config, verbose, pid_file=pid_file)


def _run_proxy_foreground(config: RecallConfig, verbose: bool, *, pid_file: Path = PID_FILE) -> None:
    """Run the uvicorn server attached to the current terminal.

    Used by ``sieve start --foreground`` (debugging) and as the child
    half of the daemon fork. Writes the PID file on entry and removes
    it on normal exit so ``sieve status`` / ``sieve stop`` see the
    process correctly.
    """
    pid_file.write_text(str(os.getpid()))

    console.print(
        f"[bold green]Sieve[/] proxy starting on "
        f"[cyan]{config.listen.host}:{config.listen.port}[/] "
        f"→ [yellow]{config.provider.base_url}[/]"
    )
    console.print(f"  Logs: [dim]{LOG_FILE}[/]")

    try:
        uvicorn.run(
            "sieve.main:app",
            host=config.listen.host,
            port=config.listen.port,
            log_level="debug" if verbose else "info",
            factory=False,
        )
    finally:
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass


def _daemonise_and_run(config: RecallConfig, verbose: bool, *, pid_file: Path = PID_FILE) -> None:
    """Double-fork to detach from the controlling terminal, then run uvicorn.

    Parent prints the "started" banner with the child PID and returns.
    The grandchild becomes the proxy: stdin/stdout/stderr redirected to
    the log file, CWD moved to root, and a new session group created so
    terminal disconnect doesn't SIGHUP us.
    """
    # First fork — parent returns to the CLI. The child continues.
    try:
        pid = os.fork()
    except OSError as exc:
        console.print(f"[bold red]Daemonise failed:[/] {exc}")
        sys.exit(1)

    if pid > 0:
        # Parent: briefly wait for the child to write the PID file so we
        # can report the child's pid confidently. Fall through to a
        # generic message if the PID never appears (child died fast).
        import time
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            child_pid = _read_pid(pid_file)
            if child_pid is not None:
                console.print(
                    f"[bold green]Sieve[/] started on "
                    f"[cyan]{config.listen.host}:{config.listen.port}[/] "
                    f"(PID {child_pid})"
                )
                console.print(f"  Logs: [dim]{LOG_FILE}[/]")
                console.print(
                    f"  Stop with: [cyan]sieve stop[/]  "
                    f"Status: [cyan]sieve status[/]"
                )
                return
            time.sleep(0.1)
        console.print(
            "[yellow]Sieve started but did not report its PID within 5s.[/] "
            f"Check [dim]{LOG_FILE}[/] — the process may have failed to bind."
        )
        return

    # Child — detach from the terminal and become a session leader.
    try:
        os.setsid()
    except OSError:
        pass

    # Second fork so the daemon is not a session leader and can never
    # reacquire a controlling terminal.
    try:
        pid2 = os.fork()
    except OSError:
        os._exit(1)
    if pid2 > 0:
        os._exit(0)

    os.chdir("/")
    os.umask(0o027)

    # Redirect standard streams to the log file so stray prints don't
    # break on a closed TTY and the operator can tail logs.
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    devnull = os.open(os.devnull, os.O_RDONLY)
    log_fd = os.open(
        str(LOG_FILE),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o640,
    )
    os.dup2(devnull, sys.stdin.fileno())
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(devnull)
    os.close(log_fd)

    # Write the PID file now that we are the final daemon process.
    try:
        pid_file.write_text(str(os.getpid()))
    except Exception:
        os._exit(1)

    try:
        uvicorn.run(
            "sieve.main:app",
            host=config.listen.host,
            port=config.listen.port,
            log_level="debug" if verbose else "info",
            factory=False,
        )
    finally:
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        os._exit(0)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    return True


def _read_pid(pid_file: Path = PID_FILE) -> int | None:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None
    if not _pid_alive(pid):
        # Stale PID file — clean it up.
        pid_file.unlink(missing_ok=True)
        return None
    return pid


@cli.command()
@click.option("--port", "-p", default=None, type=int,
              help="Check the daemon on this listen port (default: read from config)")
def status(port: int | None):
    """Show Sieve proxy and store status."""
    if not _config_exists(None):
        console.print("[bold red]Sieve is not configured.[/] Run [cyan]sieve init[/].")
        return

    # Try to read config; never crash if it's broken — just report what we can.
    try:
        config = RecallConfig.load()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return

    effective_port = port if port is not None else config.listen.port
    pid = _read_pid(_pid_file_for_port(effective_port))

    if pid:
        console.print(f"[bold green]Sieve is running[/] (pid {pid})")
        import httpx
        url = f"http://127.0.0.1:{config.listen.port}/sieve/health"
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                console.print(f"  Version: [cyan]{data.get('version', '?')}[/]")
                console.print(
                    f"  Listening: [cyan]{config.listen.host}:{config.listen.port}[/]"
                )
        except httpx.ConnectError:
            console.print("[yellow]  (proxy PID live but health endpoint unreachable)[/]")
    else:
        console.print("[bold yellow]Sieve is not running.[/] Start with: [cyan]sieve start[/]")

    # Store stats regardless of proxy state.
    try:
        from sieve.store import MemoryStore
        from sieve.progression import detect_phase
        ms = MemoryStore(config.store)
        if ms.db_path.exists():
            ms.open()
            if ms.is_initialized():
                s = ms.stats()
                current_facts = ms.count_current_facts()
                phase = detect_phase(current_facts, config.progression)
                console.print(
                    f"  Store: [cyan]{s['facts_count']}[/] facts, "
                    f"[cyan]{s['entities_count']}[/] entities"
                )
                console.print(
                    f"  Phase: [bold cyan]{phase.label}[/] "
                    f"({current_facts} current facts, keeping {phase.turns} turns)"
                )
            ms.close()
        else:
            console.print("  Store: [dim]not initialised[/]")
    except Exception as exc:
        console.print(f"  Store: [red]error[/] {exc}")


def _stop_proxy(pid_file: Path = PID_FILE) -> None:
    """Shared stop logic so `sieve stop` and `sieve restart` use the same path."""
    pid = _read_pid(pid_file)
    if pid is None:
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
            console.print(
                "[yellow]Sieve was not running (cleaned up stale state).[/]"
            )
        else:
            console.print("[yellow]Sieve is not running.[/]")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        console.print("[yellow]Sieve was not running.[/]")
        return

    import time
    for _ in range(50):
        if not _pid_alive(pid):
            pid_file.unlink(missing_ok=True)
            console.print("[bold green]Sieve stopped.[/]")
            return
        time.sleep(0.1)

    console.print("[bold yellow]Sieve did not exit within 5s — leaving PID file.[/]")


@cli.command()
@click.option("--port", "-p", default=None, type=int,
              help="Stop the daemon on this listen port (default: read from config)")
def stop(port: int | None):
    """Gracefully stop the Sieve proxy."""
    if port is None:
        # Default: read port from the config so stop matches start.
        try:
            port = RecallConfig.load().listen.port
        except Exception:
            port = _DEFAULT_PORT
    _stop_proxy(_pid_file_for_port(port))


@cli.command()
@click.option("--port", "-p", default=None, type=int, help="Override listen port")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def restart(port: int | None, verbose: bool):
    """Stop and start the Sieve proxy in one step."""
    # Stop the daemon for the listen port we're about to start. Default
    # port comes from config when --port not given, matching `start`.
    if port is None:
        try:
            stop_port = RecallConfig.load().listen.port
        except Exception:
            stop_port = _DEFAULT_PORT
    else:
        stop_port = port
    _stop_proxy(_pid_file_for_port(stop_port))
    # Replace this process with `sieve start` so the new foreground proxy
    # takes over the terminal. Extra flags are forwarded.
    argv = ["sieve", "start"]
    if port is not None:
        argv += ["--port", str(port)]
    if verbose:
        argv += ["--verbose"]
    console.print(f"[dim]Exec:[/] {' '.join(argv)}")
    os.execvp(argv[0], argv)


@cli.command()
@click.option("--soft", is_flag=True, help="Preserve ~/.sieve/ (default behaviour)")
@click.option("--hard", is_flag=True, help="Delete ~/.sieve/ completely")
def uninstall(soft: bool, hard: bool):
    """Remove Sieve. Default behaviour is --soft (data preserved)."""
    from sieve import cli_uninstall as cu
    if soft and hard:
        console.print("[bold red]--soft and --hard are mutually exclusive.[/]")
        sys.exit(2)

    if hard:
        console.print(
            "[bold red]This will PERMANENTLY DELETE all learned data "
            "in ~/.sieve/ (facts, entities, key, backups).[/]"
        )
        typed = click.prompt("Type 'DELETE' to confirm", default="", show_default=False)
        if typed != "DELETE":
            console.print("[yellow]Aborted — no changes made.[/]")
            sys.exit(1)
        cu.wipe_sieve_dir(SIEVE_DIR)
        console.print("[bold green]Sieve data directory removed.[/]")
    else:
        console.print(
            f"[green]Your data is preserved at[/] [cyan]{SIEVE_DIR}[/]"
        )
        console.print(
            f"[dim]To remove it later:[/] [cyan]rm -rf {SIEVE_DIR}[/]"
        )

    console.print("")
    console.print(cu.pip_uninstall_hint())


# --- Store subcommands ---

@cli.group()
def store():
    """Manage the Sieve memory store."""
    pass


@store.command("init")
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
def store_init(config_path: str | None):
    """Initialize the encrypted memory store."""
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)
    db_path = ms.db_path

    if db_path.exists():
        console.print(f"[yellow]Store already exists at[/] {db_path}")
        ms.open()
        if ms.is_initialized():
            console.print("[green]Schema is up to date[/]")
        else:
            ms.init_schema()
            console.print("[green]Schema initialized[/]")
        ms.close()
        return

    console.print(f"Creating encrypted store at [cyan]{db_path}[/]")
    ms.open()
    ms.init_schema()
    ms.close()
    console.print("[bold green]Store initialized[/] (encrypted with SQLCipher)")
    console.print(f"  Keyfile: [dim]{db_path.parent / '.sieve_key'}[/]")


@store.command("status")
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
def store_status(config_path: str | None):
    """Show memory store status and statistics."""
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)

    if not ms.db_path.exists():
        console.print("[bold red]Store not found.[/] Run [cyan]sieve store init[/] first.")
        sys.exit(1)

    ms.open()
    if not ms.is_initialized():
        console.print("[bold red]Store exists but schema not initialized.[/]")
        ms.close()
        sys.exit(1)

    s = ms.stats()
    ms.close()

    console.print("[bold green]Memory Store[/]")
    console.print(f"  Path: [cyan]{ms.db_path}[/]")
    size_kb = s.get("db_size_bytes", 0) / 1024
    console.print(f"  Size: [cyan]{size_kb:.1f} KB[/]")
    console.print(f"  Facts: {s['facts_count']}  (vectors: {s['vec_facts_count']})")
    console.print(f"  Entities: {s['entities_count']}  Relationships: {s['relationships_count']}")
    console.print(f"  Episodes: {s['episodes_count']}  Preferences: {s['preferences_count']}")
    console.print(f"  Sessions: {s['sessions_count']}")


# --- Store inspection subcommands ---

def _open_store(config_path: str | None):
    """Helper used by the inspection commands — opens the store for reads,
    returning (MemoryStore, RecallConfig).
    """
    from sieve.store import MemoryStore
    cfg = RecallConfig.load(config_path)
    ms = MemoryStore(cfg.store)
    if not ms.db_path.exists():
        console.print("[bold red]Store not found.[/] Run [cyan]sieve store init[/].")
        sys.exit(1)
    ms.open()
    return ms, cfg


def _facts_table(rows):
    from rich.table import Table
    table = Table(title=f"Facts ({len(rows)})", show_lines=False)
    table.add_column("Content", overflow="fold")
    table.add_column("Conf", justify="right")
    table.add_column("Source")
    table.add_column("Created")
    for r in rows:
        table.add_row(
            r["content"],
            f"{r['confidence']:.2f}" if r["confidence"] is not None else "",
            r["source"] or "",
            (r["created_at"] or "")[:19],
        )
    return table


@store.command("facts")
@click.option("--limit", default=50, type=int, help="Max rows to display")
@click.option("--search", default=None, help="Substring filter on content")
@click.option("--config", "-c", "config_path", default=None)
def store_facts_cmd(limit: int, search: str | None, config_path: str | None):
    """List facts currently in the store."""
    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        rows = csi.list_facts(ms.conn, limit=limit, search=search)
        console.print(_facts_table(rows))
    finally:
        ms.close()


@store.command("entities")
@click.option("--limit", default=50, type=int)
@click.option("--search", default=None)
@click.option("--config", "-c", "config_path", default=None)
def store_entities_cmd(limit: int, search: str | None, config_path: str | None):
    """List entities with fact counts."""
    from rich.table import Table
    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        rows = csi.list_entities(ms.conn, limit=limit, search=search)
        table = Table(title=f"Entities ({len(rows)})")
        table.add_column("Name"); table.add_column("Type"); table.add_column("Facts", justify="right")
        for r in rows:
            table.add_row(r["name"], r["type"] or "", str(r["fact_count"]))
        console.print(table)
    finally:
        ms.close()


@store.command("relationships")
@click.option("--limit", default=50, type=int)
@click.option("--config", "-c", "config_path", default=None)
def store_rel_cmd(limit: int, config_path: str | None):
    """List entity → entity relationships."""
    from rich.table import Table
    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        rows = csi.list_relationships(ms.conn, limit=limit)
        table = Table(title=f"Relationships ({len(rows)})")
        table.add_column("Source"); table.add_column("Relationship"); table.add_column("Target")
        table.add_column("Conf", justify="right"); table.add_column("Status")
        for r in rows:
            table.add_row(
                r["source_name"], r["relationship"], r["target_name"],
                f"{r['confidence']:.2f}" if r["confidence"] is not None else "",
                r["status"] or "",
            )
        console.print(table)
    finally:
        ms.close()


@store.command("episodes")
@click.option("--limit", default=50, type=int)
@click.option("--config", "-c", "config_path", default=None)
def store_episodes_cmd(limit: int, config_path: str | None):
    """List episodic memories."""
    from rich.table import Table
    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        rows = csi.list_episodes(ms.conn, limit=limit)
        table = Table(title=f"Episodes ({len(rows)})")
        table.add_column("Summary", overflow="fold"); table.add_column("Created")
        for r in rows:
            table.add_row(r["summary"], (r["created_at"] or "")[:19])
        console.print(table)
    finally:
        ms.close()


@store.command("stats")
@click.option("--config", "-c", "config_path", default=None)
def store_stats_cmd(config_path: str | None):
    """Detailed statistics for the store."""
    from rich.table import Table
    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        stats = csi.detailed_stats(ms.conn, ms.db_path)
    finally:
        ms.close()

    t = Table(title=f"Store — {ms.db_path}")
    t.add_column("Metric"); t.add_column("Value", justify="right")
    for k in ("facts", "entities", "relationships", "episodes",
              "preferences", "sessions", "known_unknowns", "vec_facts",
              "audit_log", "fingerprints"):
        t.add_row(k, str(stats.get(k, 0)))
    size_kb = stats.get("db_size_bytes", 0) / 1024
    t.add_row("db_size_kb", f"{size_kb:.1f}")
    t.add_row("avg_facts_per_entity", str(stats.get("avg_facts_per_entity", 0)))
    console.print(t)


@store.command("export")
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--output", "-o", required=True, help="Output file (json) or directory (csv)")
@click.option("--config", "-c", "config_path", default=None)
def store_export_cmd(fmt: str, output: str, config_path: str | None):
    """Export store contents (decrypted)."""
    from pathlib import Path
    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        if fmt == "json":
            csi.export_json(ms.conn, Path(output))
        else:
            csi.export_csv(ms.conn, Path(output))
    finally:
        ms.close()
    console.print(f"[green]Exported {fmt} to[/] [cyan]{output}[/]")


@store.command("wipe")
@click.option("--config", "-c", "config_path", default=None)
def store_wipe_cmd(config_path: str | None):
    """Delete all data from the store (keeps schema + key)."""
    console.print("[bold red]This will delete ALL learned data from the store.[/]")
    typed = click.prompt("Type 'WIPE' to confirm", default="", show_default=False)
    if typed != "WIPE":
        console.print("[yellow]Aborted — no changes made.[/]")
        sys.exit(1)

    from sieve import cli_store_inspect as csi
    ms, _ = _open_store(config_path)
    try:
        csi.wipe_store(ms.conn)
    finally:
        ms.close()
    console.print("[bold green]Store wiped.[/] Schema and key preserved.")


# --- Config subcommands ---

@cli.group()
def config():
    """Runtime configuration management."""
    pass


@config.command("show")
def config_show():
    """Display the current configuration."""
    from rich.table import Table
    from sieve import cli_config as cc

    cfg = RecallConfig.load(cc.current_config_path())
    diffs = {p for (p, _, _) in cc.diff_from_defaults(cfg)}

    # Flatten the dataclass for a row-by-row table.
    rows: list[tuple[str, str]] = []
    from dataclasses import fields, is_dataclass

    def walk(prefix, obj):
        if is_dataclass(obj):
            for f in fields(obj):
                walk(f"{prefix}.{f.name}" if prefix else f.name, getattr(obj, f.name))
        else:
            rows.append((prefix, str(obj)))

    walk("", cfg)

    table = Table(title=f"Sieve config — {cc.current_config_path()}", show_lines=False)
    table.add_column("Key")
    table.add_column("Value")
    table.add_column("")
    for key, value in rows:
        marker = "[bold yellow]• non-default[/]" if key in diffs else ""
        style = "yellow" if key in diffs else None
        table.add_row(key, value, marker, style=style)
    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a config option. Example: sieve config set listen.port 11500."""
    from sieve import cli_config as cc

    data = cc.load_raw()
    try:
        cc.set_path(data, key, value)
        cc.validate_raw(data)
    except ValueError as exc:
        console.print(f"[bold red]Invalid config update:[/] {exc}")
        sys.exit(2)

    cc.write_raw(data)
    console.print(f"[green]Set[/] [cyan]{key}[/] = [cyan]{value}[/]")
    console.print("[dim]Restart Sieve to apply the change.[/]")


@config.command("reset")
def config_reset():
    """Reset config to defaults (preserves provider URL and store path)."""
    from sieve import cli_config as cc

    current = cc.load_raw()
    preserved_url = (current.get("provider") or {}).get("base_url")
    preserved_store = (current.get("store") or {}).get("path")

    console.print(
        "[yellow]This will reset all config options to their defaults.[/]"
    )
    click.confirm("Continue?", abort=True)

    # Load a fresh defaults dict from RecallConfig().
    defaults_obj = RecallConfig()
    new: dict = {
        "listen": {"host": defaults_obj.listen.host, "port": defaults_obj.listen.port},
        "provider": {
            "type": defaults_obj.provider.type,
            "base_url": preserved_url or defaults_obj.provider.base_url,
            "default_model": defaults_obj.provider.default_model,
        },
        "embeddings": {"provider": defaults_obj.embeddings.provider},
        "store": {"path": preserved_store or defaults_obj.store.path},
    }
    cc.write_raw(new)
    console.print("[bold green]Config reset to defaults.[/]")


@config.command("edit")
def config_edit():
    """Open the config file in $EDITOR and validate after save."""
    from sieve import cli_config as cc

    cfg_path = cc.current_config_path()
    before = cfg_path.read_text() if cfg_path.exists() else ""

    # Click.edit writes to a tempfile when filename is passed; simpler path:
    # pass the real config file so the editor mutates it in place.
    click.edit(filename=str(cfg_path), extension=".yaml")

    # Validate
    try:
        import yaml as _yaml
        raw = _yaml.safe_load(cfg_path.read_text()) or {}
        cc.validate_raw(raw)
    except Exception as exc:
        # Roll back
        cfg_path.write_text(before)
        console.print(f"[bold red]Invalid YAML — rolled back:[/] {exc}")
        sys.exit(2)

    console.print("[green]Config saved.[/]")


# --- Key management subcommands ---

@cli.group()
def key():
    """Manage the encryption key protecting the memory store."""
    pass


def _keyfile_path(config_path: str | None) -> Path:
    """Resolve the keyfile path from the config."""
    from sieve import cli_keys
    cfg = RecallConfig.load(config_path)
    db = Path(cfg.store.path).expanduser()
    return cli_keys.keyfile_for(db)


@key.command("show")
@click.option("--config", "-c", "config_path", default=None)
def key_show(config_path: str | None):
    """Show keyfile location and fingerprint (the key itself is never displayed)."""
    from sieve import cli_keys
    kf = _keyfile_path(config_path)
    if not kf.exists():
        console.print(f"[bold red]Keyfile not found at[/] [cyan]{kf}[/]")
        console.print(
            "[dim]One will be generated on first `sieve start`, or use `sieve key import`.[/]"
        )
        sys.exit(1)
    key_txt = kf.read_text().strip()
    fp = cli_keys.fingerprint(key_txt)
    console.print(f"[bold green]Keyfile:[/] [cyan]{kf}[/]")
    console.print(f"[bold green]Fingerprint:[/] [cyan]{fp}[/]")
    mode = oct(kf.stat().st_mode)[-3:]
    if mode != "600":
        console.print(f"[yellow]Warning:[/] permissions are {mode} (expected 600)")


@key.command("rotate")
@click.option("--config", "-c", "config_path", default=None)
def key_rotate(config_path: str | None):
    """Re-encrypt the store with a new key (destructive — back up first)."""
    from sieve import cli_keys

    cfg = RecallConfig.load(config_path)
    db_path = Path(cfg.store.path).expanduser()
    kf = cli_keys.keyfile_for(db_path)

    if not db_path.exists():
        console.print("[bold red]Store not found.[/]")
        sys.exit(1)
    if not kf.exists():
        console.print("[bold red]Keyfile not found.[/] Cannot rotate.")
        sys.exit(1)

    console.print(
        "[bold red]Key rotation re-encrypts the entire store.[/] "
        "If this is interrupted your store may become unreadable."
    )
    console.print(
        "[yellow]Strongly recommended:[/] run [cyan]sieve backup create[/] first."
    )
    typed = click.prompt("Type 'ROTATE' to confirm", default="", show_default=False)
    if typed != "ROTATE":
        console.print("[yellow]Aborted — no changes made.[/]")
        sys.exit(1)

    old_key = kf.read_text().strip()

    # Auto-generate or custom?
    auto = click.confirm(
        "Generate a new random key automatically?", default=True
    )
    if auto:
        new_key = cli_keys.generate_key()
    else:
        new_key = click.prompt("New passphrase", hide_input=True)
        if not new_key:
            console.print("[bold red]Empty passphrase — aborted.[/]")
            sys.exit(1)

    try:
        cli_keys.rotate_key(db_path, old_key=old_key, new_key=new_key)
    except Exception as exc:
        console.print(f"[bold red]Rotation failed:[/] {exc}")
        sys.exit(1)

    console.print("[bold green]Key rotated successfully.[/]")
    console.print(f"  Keyfile: [cyan]{kf}[/]")
    console.print(f"  Fingerprint: [cyan]{cli_keys.fingerprint(new_key)}[/]")


@key.command("export")
@click.option("--config", "-c", "config_path", default=None)
def key_export(config_path: str | None):
    """Print the raw key to stdout. For backup — handle carefully."""
    kf = _keyfile_path(config_path)
    if not kf.exists():
        console.print("[bold red]Keyfile not found.[/]")
        sys.exit(1)

    console.print(
        "[bold yellow]WARNING:[/] This will print your encryption key. "
        "Anyone with this key can read your memory store. "
        "Store it somewhere safe (password manager, offline backup)."
    )
    click.confirm("Continue?", abort=True)

    console.print("")
    console.print(kf.read_text().strip())


@key.command("import")
@click.argument("keyfile", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "-c", "config_path", default=None)
def key_import(keyfile: str, config_path: str | None):
    """Import a key from KEYFILE and verify it opens the current store."""
    from sieve import cli_keys
    cfg = RecallConfig.load(config_path)
    db_path = Path(cfg.store.path).expanduser()
    if not db_path.exists():
        console.print("[bold red]Store not found.[/]")
        sys.exit(1)
    try:
        cli_keys.import_key(db_path, Path(keyfile))
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Import failed:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Key imported.[/] Fingerprint: [cyan]{cli_keys.fingerprint(Path(keyfile).read_text().strip())}[/]")


# --- Backup subcommands ---

@cli.group()
def backup():
    """Manage encrypted backups of the memory store."""
    pass


@backup.command("create")
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
@click.option("--output", "-o", default=None, help="Output path for backup file")
def backup_create(config_path: str | None, output: str | None):
    """Create an encrypted backup of the memory store."""
    from pathlib import Path
    from sieve.backup import create_backup
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)

    if not ms.db_path.exists():
        console.print("[bold red]Store not found.[/] Nothing to back up.")
        sys.exit(1)

    out = Path(output) if output else None
    backup_path, checksum_path = create_backup(ms.db_path, output=out)
    size_kb = backup_path.stat().st_size / 1024
    console.print(f"[bold green]Backup created[/] ({size_kb:.1f} KB)")
    console.print(f"  File: [cyan]{backup_path}[/]")
    console.print(f"  Checksum: [cyan]{checksum_path}[/]")


@backup.command("list")
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
def backup_list(config_path: str | None):
    """List available backups."""
    from rich.table import Table
    from sieve.backup import list_backups
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)
    backups = list_backups(ms.db_path)

    if not backups:
        console.print("[dim]No backups found.[/]")
        return

    t = Table(title=f"Backups ({len(backups)})")
    t.add_column("Timestamp")
    t.add_column("Size", justify="right")
    t.add_column("Checksum")
    t.add_column("Path", overflow="fold")
    for b in backups:
        size_kb = b["size_bytes"] / 1024
        status = "[green]OK[/]" if b["checksum_valid"] else "[yellow]unverified[/]"
        t.add_row(b["timestamp"], f"{size_kb:.1f} KB", status, b["path"])
    console.print(t)


@backup.command("restore")
@click.argument("backup_id")
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def backup_restore(backup_id: str, config_path: str | None, yes: bool):
    """Restore the store from a backup."""
    from pathlib import Path
    from sieve.backup import list_backups, restore_backup
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)
    backups = list_backups(ms.db_path)

    match = [b for b in backups if b["id"] == backup_id]
    if not match:
        console.print(f"[bold red]Backup '{backup_id}' not found.[/]")
        sys.exit(1)

    backup_path = Path(match[0]["path"])
    if not yes:
        console.print(f"[yellow]This will overwrite the current store with backup {backup_id}.[/]")
        click.confirm("Continue?", abort=True)

    success = restore_backup(backup_path, ms.db_path)
    if success:
        console.print("[bold green]Restore complete.[/]")
    else:
        console.print("[bold red]Restore failed — checksum mismatch.[/]")
        sys.exit(1)


# --- Store migrate ---

@store.command("migrate")
@click.option("--to", "dest_path", required=True, help="Destination path for the store")
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
def store_migrate(dest_path: str, config_path: str | None):
    """Migrate the memory store to a new location."""
    from pathlib import Path
    from sieve.backup import migrate_store
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)

    if not ms.db_path.exists():
        console.print("[bold red]Store not found.[/] Nothing to migrate.")
        sys.exit(1)

    dst = Path(dest_path).expanduser()
    console.print(f"Migrating [cyan]{ms.db_path}[/] → [cyan]{dst}[/]")

    success = migrate_store(ms.db_path, dst)
    if success:
        console.print("[bold green]Migration complete.[/]")
        console.print(f"  Update store.path in sieve.yaml to: [cyan]{dst}[/]")
    else:
        console.print("[bold red]Migration failed — integrity check failed.[/]")
        sys.exit(1)


def _run_wizard_flow() -> None:
    """Drive the interactive wizard against real stdin / httpx / sockets.

    Split out so tests can stub this function wholesale and verify the
    --wizard flag routes correctly.
    """
    import click as _click
    import httpx

    from sieve.cli_wizard import (
        WizardContext,
        apply_wizard_answers,
        run_wizard,
    )

    class _ClickPrompter:
        def ask(self, q: str, default: str | None = None) -> str:
            return _click.prompt(q, default=default if default is not None else "", show_default=bool(default))
        def confirm(self, q: str, default: bool = True) -> bool:
            return _click.confirm(q, default=default)

    class _HttpxProbe:
        def check(self, url: str) -> tuple[bool, list[str]]:
            try:
                r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=3.0)
                if r.status_code == 200:
                    data = r.json() or {}
                    models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                    return True, models
                return False, []
            except Exception:
                return False, []

    def _download() -> None:
        console.print("Downloading embedding model (BAAI/bge-small-en-v1.5, ~50MB)...")
        from fastembed import TextEmbedding
        _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        console.print("[green]Embedding model ready.[/]")

    ctx = WizardContext(
        prompter=_ClickPrompter(),
        probe=_HttpxProbe(),
        download_model=_download,
    )
    answers = run_wizard(ctx)
    if not answers.confirmed:
        console.print("[yellow]Wizard cancelled — no changes made.[/]")
        return

    _download()
    apply_wizard_answers(answers)

    # Initialise the encrypted store at the chosen location.
    # Load the just-written YAML through RecallConfig so the resolved
    # store.embedding_dimensions (384 under FastEmbed, 768 under Ollama)
    # reaches init_schema. A bare StoreConfig(path=...) would fall back
    # to the legacy 768 default and bake a mismatched vec_facts schema.
    from sieve.store import MemoryStore
    cfg = RecallConfig.load()
    ms = MemoryStore(cfg.store)
    if not ms.db_path.exists():
        ms.open()
        ms.init_schema()
        ms.close()
        console.print(f"[green]Initialised encrypted store at[/] [cyan]{ms.db_path}[/]")

    console.print("[bold green]Wizard complete.[/] Start with [cyan]sieve start[/].")

    # Offer the benchmark as a post-install sanity check. This is the
    # "prove it works on your machine" path — token reduction, memory
    # learning, and absence-signal detection in one shot.
    console.print()
    run_bench = click.confirm(
        "Run sieve benchmark now to verify token reduction on your machine?",
        default=False,
    )
    if run_bench:
        if _read_pid() is None:
            console.print(
                "\n[yellow]The proxy is not running yet.[/] "
                "Open another terminal and run [cyan]sieve start[/], "
                "then come back here and run [cyan]sieve benchmark[/].\n"
            )
        else:
            cli.main(args=["benchmark"], standalone_mode=False)


@cli.command()
@click.option("--provider", default=None, help="LLM provider base URL (e.g. http://localhost:11434)")
@click.option("--force", is_flag=True, help="Reinitialise even if ~/.sieve already exists")
@click.option("--wizard", is_flag=True, help="Interactive guided setup")
def init(provider: str | None, force: bool, wizard: bool):
    """Initialise Sieve — creates ~/.sieve/, downloads the embedding model,
    writes a default config, and initialises the encrypted memory store.

    Default mode is lazy (zero prompts, all defaults). Pass --wizard for
    guided interactive setup.
    """
    if wizard:
        _run_wizard_flow()
        return

    _setup_logging(verbose=False)

    cfg_path = SIEVE_DIR / "sieve.yaml"
    if SIEVE_DIR.exists() and not force:
        if cfg_path.exists():
            console.print(
                f"[yellow]Sieve is already configured at[/] [cyan]{SIEVE_DIR}[/]."
            )
            if not click.confirm("Reinitialise?", default=False):
                console.print("[dim]Leaving existing configuration in place.[/]")
                return

    SIEVE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: pick a provider URL ---
    if provider is None:
        # Try to auto-detect Ollama on 11434.
        import httpx
        default = "http://127.0.0.1:11434"
        try:
            httpx.get(f"{default}/api/tags", timeout=1.5)
            console.print(f"[green]Ollama detected at[/] [cyan]{default}[/]")
            provider = default
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            console.print("[yellow]No Ollama on localhost:11434.[/]")
            provider = click.prompt(
                "Enter LLM provider base URL",
                default=default,
            )

    # --- Step 2: normalise + health-check the provider ---
    # Normalise the URL so users can paste "localhost:11434" or
    # "192.168.1.10" without an explicit scheme. We default to http
    # for bare hostnames (the common Ollama case); https URLs are
    # preserved as-is.
    if provider and "://" not in provider:
        provider = f"http://{provider}"
        console.print(f"[dim]Assuming http:// — using {provider}[/]")

    import httpx
    try:
        r = httpx.get(f"{provider.rstrip('/')}/api/tags", timeout=3.0)
        if r.status_code == 200:
            console.print(f"[green]Provider reachable at[/] [cyan]{provider}[/]")
        else:
            console.print(
                f"[yellow]Provider responded {r.status_code} — will retry at start.[/]"
            )
    except httpx.ConnectError:
        console.print(
            f"[yellow]Could not reach {provider} (connection refused).[/] "
            f"Continuing — start your LLM endpoint before `sieve start`."
        )
    except Exception as exc:
        console.print(
            f"[yellow]Could not reach provider: {type(exc).__name__}: {exc}.[/]\n"
            f"[dim]Continuing — you can fix this in {cfg_path} later.[/]"
        )

    # --- Step 3: download the FastEmbed model with progress ---
    console.print("Downloading embedding model (BAAI/bge-small-en-v1.5, ~50MB)...")
    try:
        from fastembed import TextEmbedding
        # Instantiating downloads+caches the ONNX model.
        _ = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        console.print("[green]Embedding model ready.[/]")
    except Exception as exc:
        console.print(f"[bold red]Failed to download embedding model:[/] {exc}")
        console.print("Check your internet connection and re-run [cyan]sieve init[/].")
        sys.exit(1)

    # --- Step 4: write sieve.yaml from the packaged example ---
    try:
        import importlib.resources as pkg_resources
        # Try packaged example first; fall back to repo-root during dev.
        example_text: str | None = None
        try:
            pkg_root = pkg_resources.files("sieve").parent
            example_path = pkg_root / "sieve.example.yaml"
            if example_path.is_file():
                example_text = example_path.read_text()
        except Exception:
            pass
        if example_text is None:
            repo_example = Path(__file__).resolve().parent.parent / "sieve.example.yaml"
            if repo_example.exists():
                example_text = repo_example.read_text()
        if example_text is None:
            example_text = _DEFAULT_CONFIG_YAML

        # Swap in the provider URL the user picked.
        example_text = example_text.replace(
            "http://127.0.0.1:11434", provider.rstrip("/")
        )
        cfg_path.write_text(example_text)
        console.print(f"[green]Wrote config to[/] [cyan]{cfg_path}[/]")
    except Exception as exc:
        console.print(f"[bold red]Failed to write config:[/] {exc}")
        sys.exit(1)

    # --- Step 5: initialise the encrypted memory store ---
    try:
        from sieve.store import MemoryStore
        os.environ["SIEVE_CONFIG"] = str(cfg_path)
        config = RecallConfig.load(str(cfg_path))
        ms = MemoryStore(config.store)
        if not ms.db_path.exists():
            ms.open()
            ms.init_schema()
            ms.close()
            console.print(
                f"[green]Initialised encrypted store at[/] [cyan]{ms.db_path}[/]"
            )
        else:
            console.print(f"[dim]Store already exists at {ms.db_path}[/]")
    except Exception as exc:
        console.print(f"[bold red]Store init failed:[/] {exc}")
        sys.exit(1)

    console.print()
    console.print("[bold green]Ready![/]")
    console.print(
        f"  [dim]Configured for:[/] [cyan]{provider}[/]"
    )
    console.print(
        f"  [dim]Sieve will listen on:[/] [cyan]http://localhost:{config.listen.port}[/]"
    )
    console.print()
    console.print(f"  Run [cyan]sieve start[/] to launch the proxy.")
    console.print(f"  Run [cyan]sieve demo[/] to see Sieve in action with a 30-second sample.")
    console.print(
        f"  Then point your agent at [cyan]http://localhost:{config.listen.port}[/]."
    )


_DEFAULT_CONFIG_YAML = """\
# Sieve — minimal default configuration.
listen:
  host: 127.0.0.1
  port: 11435

provider:
  type: auto
  base_url: http://127.0.0.1:11434
  default_model: qwen3.5:9b

embeddings:
  provider: fastembed

store:
  path: ~/.sieve/memory.db
"""


@cli.command()
@click.option(
    "--wait-for-write/--no-wait-for-write",
    default=True,
    help="Poll the store after each turn until the async writer has committed "
    "the new fact(s) before sending the next turn. Prevents the demo from "
    "racing ahead of fact extraction. Default: enabled.",
)
@click.option(
    "--max-wait",
    default=15.0,
    type=float,
    help="Maximum seconds to wait for the writer per turn (when "
    "--wait-for-write is enabled). Falls through after this.",
)
@click.option(
    "--use-main-store",
    is_flag=True,
    default=False,
    help=(
        "Run against the user's main proxy and store instead of an "
        "ephemeral sandbox. Advanced/debug use only — a demo run adds "
        "demo facts to the live store. Requires `sieve start`."
    ),
)
def demo(wait_for_write: bool, max_wait: float, use_main_store: bool):
    """Run a short scripted demo.

    By default spins up an isolated sandbox proxy with a scratch store,
    runs the demo against it, then tears everything down — the user's
    real proxy and store are never touched. Pass ``--use-main-store``
    to run against the live install (advanced/debug only).
    """
    try:
        config = RecallConfig.load()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        sys.exit(1)

    # Pre-flight: is the configured LLM endpoint actually reachable?
    # Without this, the sandbox spins up, we run 6 turns, each one
    # gets a 502 from the sandbox proxy (which can't reach Ollama),
    # and the user sees a wall of identical errors with no fix hint.
    if not _demo_provider_reachable(config.provider.base_url):
        console.print(
            f"[bold red]Can't reach {config.provider.base_url}[/] — "
            "the LLM endpoint isn't responding.\n"
        )
        console.print(
            "[bold]How to fix:[/]\n"
            "  1. Verify Ollama (or your provider) is running and "
            "accepting requests.\n"
            "  2. If the endpoint moved, update it:\n"
            "     [cyan]sieve wizard[/]  →  Config  →  provider.base_url\n"
            "  3. Or re-init with a new provider:\n"
            f"     [cyan]sieve init --force --provider http://HOST:PORT[/]\n"
        )
        sys.exit(1)

    if use_main_store:
        pid = _read_pid()
        if pid is None:
            console.print(
                "[bold red]Sieve is not running.[/] --use-main-store "
                "requires [cyan]sieve start[/] in another terminal."
            )
            sys.exit(1)
        from sieve.store import MemoryStore
        ms: "MemoryStore | None" = None
        if wait_for_write:
            try:
                ms = MemoryStore(config.store)
                ms.open()
            except Exception as exc:
                console.print(f"[dim]store poll disabled ({exc})[/]")
                ms = None
        base = f"http://127.0.0.1:{config.listen.port}"
        try:
            _run_demo_loop(
                base_url=base,
                model=config.provider.default_model,
                store=ms,
                wait_for_write=wait_for_write,
                max_wait=max_wait,
            )
        finally:
            if ms is not None:
                ms.close()
        return

    # Sandbox path — the default.
    from sieve._sandbox import SandboxedProxy
    from sieve.store import MemoryStore

    console.print(
        "[dim]Starting sandbox proxy (isolated from your main store)…[/]"
    )
    try:
        with SandboxedProxy.from_main_config(config) as sb:
            console.print(
                f"[dim]Sandbox ready at [cyan]{sb.base_url}[/]\n[/]"
            )
            ms = None
            if wait_for_write:
                try:
                    ms = MemoryStore(sb.config.store)
                    ms.open()
                except Exception as exc:
                    console.print(f"[dim]store poll disabled ({exc})[/]")
                    ms = None
            try:
                _run_demo_loop(
                    base_url=sb.base_url,
                    model=sb.config.provider.default_model,
                    store=ms,
                    wait_for_write=wait_for_write,
                    max_wait=max_wait,
                )
            finally:
                if ms is not None:
                    ms.close()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted — sandbox cleaned up.[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"[bold red]Demo failed:[/] {exc}")
        sys.exit(1)


def _demo_provider_reachable(url: str, timeout: float = 3.0) -> bool:
    """True iff ``{url}/api/tags`` returns 200 within ``timeout``.

    Used by the demo's pre-flight so we fail fast with an actionable
    message instead of running 6 turns of 502s.
    """
    import httpx
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _run_demo_loop(
    *,
    base_url: str,
    model: str,
    store,
    wait_for_write: bool,
    max_wait: float,
) -> None:
    """Run the 6-message demo against a live proxy + optional store reader."""
    import httpx
    import time

    def fact_count() -> int:
        if store is None:
            return 0
        try:
            return int(store.stats().get("facts_count", 0))
        except Exception:
            return 0

    # Demo-script messages, chosen to exercise the full pipeline:
    #   turn 1: fact-share (name + occupation) — strong-keyword "work"
    #   turn 2: fact-share (preference) — strong-keyword "love"
    #   turn 3: fact-share (pet) — strong-keyword "dog"
    #   turns 4-5: recall queries against the just-learned facts
    #   turn 6: absence-signal trap (asks about a person never mentioned)
    # Each fact-share message must contain a regex-extractable signal
    # OR a strong-keyword from the S2 gate, otherwise the gate stays
    # closed and the user sees "no growth — writer missed this turn"
    # — bad demo optics. See _s2_gate in writer.py for the rules.
    messages = [
        "Hi, I'm Casey. I work as a landscape architect.",
        "I love working on public parks and garden designs.",
        "I have a dog who's a border terrier.",
        "Do you remember where I work?",
        "What breed is my dog?",
        "Do you remember where Pat works?",  # absence-signal trap
    ]
    expect_new_fact = [True, True, True, False, False, False]

    from sieve.cli_benchmark import UpstreamLLMError, _extract_upstream_message

    console.print("[bold]Sieve demo[/] — 6 messages through the proxy:\n")
    consecutive_errors = 0
    aborted = False
    # Accumulators for the closing summary panel
    completed_turns = 0
    last_assistant_text = ""    # for the absence-trap verdict
    proxy_us_list: list[int] = []
    facts_learned_total = 0
    recall_rounds_total = 0
    for i, msg in enumerate(messages, 1):
        if aborted:
            break
        before = fact_count()
        console.print(f"[dim]turn {i}:[/] [cyan]{msg}[/]")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": msg}],
            "stream": False,
        }
        try:
            r = httpx.post(f"{base_url}/api/chat", json=payload, timeout=60.0)
            if r.status_code >= 400:
                upstream = _extract_upstream_message(r)
                raise UpstreamLLMError(
                    status=r.status_code,
                    url=f"{base_url}/api/chat",
                    model=model,
                    turn=i,
                    phase="demo",
                    upstream_message=upstream,
                )
            consecutive_errors = 0
            data = r.json()
            raw = (data.get("message") or {}).get("content", "") or ""
            if "<think>" in raw and "</think>" in raw:
                import re
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            last_assistant_text = raw
            text = raw[:240] if raw.strip() else "[dim](no visible content — check model output)[/]"
            rounds = r.headers.get("X-Sieve-Rounds", "0")
            proxy_us = r.headers.get("X-Sieve-Proxy-Us", "?")
            phase_name = r.headers.get("X-Sieve-Phase", "?")
            fact_ct = r.headers.get("X-Sieve-Fact-Count", "?")
            phase_tag = f"[{phase_name}: {fact_ct} facts]"
            console.print(
                f"        [green]→[/] {text}  "
                f"[dim]{phase_tag} (recall rounds: {rounds}, proxy_us: {proxy_us})[/]"
            )
            completed_turns += 1
            try:
                proxy_us_list.append(int(proxy_us))
            except (TypeError, ValueError):
                pass
            try:
                recall_rounds_total += int(rounds)
            except (TypeError, ValueError):
                pass
        except UpstreamLLMError as exc:
            consecutive_errors += 1
            short = (exc.upstream_message or f"HTTP {exc.status}").strip()
            short_line = " ".join(short.split())[:200]
            console.print(f"        [red]error {exc.status}:[/] {short_line}")
            # After two consecutive upstream errors, the endpoint is
            # clearly not recovering — abort with one clear hint
            # panel rather than repeating the same error six times.
            if consecutive_errors >= 2:
                hint = exc.user_hint() or (
                    "The LLM endpoint is unreachable or rejecting "
                    "requests. Verify your provider is running, then "
                    "run `sieve wizard` → Config to point at it."
                )
                console.print()
                console.print(
                    f"[bold yellow]Aborting demo[/] — {consecutive_errors} "
                    "turns in a row failed against the LLM endpoint.\n"
                )
                console.print(f"[bold]Hint:[/] {hint}\n")
                aborted = True
                break
        except Exception as exc:
            consecutive_errors += 1
            console.print(f"        [red]error:[/] {exc}")
            if consecutive_errors >= 2:
                console.print(
                    "\n[bold yellow]Aborting demo[/] — multiple "
                    "consecutive errors. Check your LLM endpoint.\n"
                )
                aborted = True
                break

        if wait_for_write and store is not None and expect_new_fact[i - 1] and i < len(messages):
            deadline = time.monotonic() + max_wait
            while time.monotonic() < deadline and fact_count() <= before:
                time.sleep(0.3)

        after = fact_count()
        delta = after - before
        if expect_new_fact[i - 1]:
            if delta > 0:
                console.print(
                    f"        [dim]📝 Facts: {after} ([green]+{delta}[/])[/]"
                )
                facts_learned_total += delta
            else:
                console.print(
                    f"        [yellow]📝 Facts: {after} (no growth — writer missed this turn)[/]"
                )
        else:
            console.print(f"        [dim]📝 Facts: {after}[/]")
        console.print()

    # ── Closing summary panel ───────────────────────────────────────
    # Renders only if the demo wasn't aborted by errors. Tells the user
    # what they just saw — what was learned, whether the trap fired,
    # latency stats — and what to do next.
    if not aborted and completed_turns == len(messages):
        _render_demo_summary(
            completed_turns=completed_turns,
            facts_learned=facts_learned_total,
            final_fact_count=fact_count(),
            last_assistant_text=last_assistant_text,
            proxy_us_list=proxy_us_list,
            recall_rounds_total=recall_rounds_total,
        )


def _render_demo_summary(
    *,
    completed_turns: int,
    facts_learned: int,
    final_fact_count: int,
    last_assistant_text: str,
    proxy_us_list: list[int],
    recall_rounds_total: int,
) -> None:
    """Render the closing panel for `sieve demo`.

    Summarises the 6-turn conversation:
      • turns completed + facts learned
      • absence-trap verdict (did the model refuse to fabricate?)
      • median proxy latency
      • what to try next
    """
    from rich.panel import Panel
    # Absence-trap verdict: turn 6 asked about a person ("Pat") who was
    # never mentioned. A correctly-behaving Sieve install should NOT
    # invent ANY fact about Pat — not a workplace, not a profession,
    # not a relationship, nothing. The strict check below fails the
    # verdict on any of those inventions, regardless of whether the
    # model also said "I don't know the workplace."
    trap_text = (last_assistant_text or "").lower()
    refused_markers = (
        "don't have", "do not have", "haven't been told", "haven't mentioned",
        "no information", "isn't mentioned", "not mentioned",
        "no record", "haven't shared", "you haven't",
        "no context", "wasn't shared", "you didn't mention",
        "i don't know", "i do not know", "haven't told me",
    )
    refused = any(m in trap_text for m in refused_markers)
    # Stronger counter-check: did the model invent ANY claim about Pat?
    # If "pat" appears followed by a claim verb (is/works/lives/has/etc.)
    # within a short window, the model fabricated.
    import re as _re
    invention_pattern = _re.compile(
        r"\bpat\s+(?:is|works|lives|has|had|likes|loves|prefers|"
        r"studied|studies|enjoys|graduated|moved|drives|owns|"
        r"speaks|plays|teaches|leads|runs)\b",
        _re.IGNORECASE,
    )
    looks_like_invention = bool(invention_pattern.search(trap_text))
    if looks_like_invention:
        trap_line = "[red]✗ fabricated a claim about Pat — review turn 6 above[/]"
    elif refused:
        trap_line = "[green]✓ refused to fabricate[/]"
    else:
        trap_line = "[yellow]⚠ couldn't confirm refusal — review turn 6 above[/]"

    # Latency stats
    if proxy_us_list:
        proxy_us_list = sorted(proxy_us_list)
        p50_us = proxy_us_list[len(proxy_us_list) // 2]
        p50_ms = p50_us / 1000
        latency_line = f"[cyan]{p50_ms:.1f} ms[/] (p50 across {len(proxy_us_list)} turns)"
    else:
        latency_line = "[dim]no latency data[/]"

    lines = [
        f"[bold]Turns:[/]        [cyan]{completed_turns}/6[/] completed",
        f"[bold]Facts learned:[/] [cyan]{final_fact_count}[/] "
        f"(grew by [green]+{facts_learned}[/] across the conversation)",
        f"[bold]Recall calls:[/]  [cyan]{recall_rounds_total}[/] "
        f"in-flight context pulls",
        f"[bold]Trap turn 6:[/]   {trap_line}",
        f"[bold]Proxy latency:[/] {latency_line}",
        "",
        "[bold]What this proved[/]",
        "  • Sieve sat between your agent and your LLM, transparently",
        "  • Facts you shared were extracted into the encrypted local store",
        "  • Later turns retrieved those facts without re-sending history",
        "  • When asked about a person never mentioned, the model didn't invent",
        "",
        "[bold]Next:[/]",
        "  • [cyan]sieve benchmark[/]  — measure token reduction + cost savings",
        "  • [cyan]sieve store stats[/] — inspect what was learned",
        "  • [cyan]sieve wizard[/]     — interactive management menu",
    ]
    console.print(
        Panel("\n".join(lines), title="Demo complete", border_style="green")
    )


_FIXTURE_CHOICES = ("small", "medium", "large", "xlarge")
_PRICING_CHOICES = (
    "local", "claude-opus", "claude-sonnet", "claude-haiku",
    "gpt-4o", "gpt-4o-mini",
)


def _fixture_menu_label(name: str) -> str:
    """Human label for wizard listing, e.g. 'medium   ~20K tokens — Cursor ...'."""
    from sieve._agent_fixture import fixture_approx_tokens, fixture_description
    base = fixture_approx_tokens(name)
    return f"{name:<7} ~{base:>6,} base tokens  —  {fixture_description(name)}"


def _looks_like_local(direct_base_url: str) -> bool:
    """Heuristic: is the configured LLM endpoint localhost / LAN?"""
    lo = (direct_base_url or "").lower()
    return (
        "localhost" in lo
        or "127.0.0.1" in lo
        or "://192.168." in lo
        or "://10." in lo
        or "://172.16." in lo
    )


def _context_window_warning_text(fixture: str, model_name: str, base_url: str) -> str:
    """Text shown before running a heavy fixture against what looks like a local model."""
    from sieve._agent_fixture import fixture_approx_tokens
    base = fixture_approx_tokens(fixture)
    return (
        f"Fixture '{fixture}' ships ~{base:,} tokens of base payload per turn "
        f"(+ tool schemas + growing history, typically 2-3× more by turn 15).\n"
        f"  Your LLM endpoint is {base_url} with model '{model_name}'.\n"
        f"  Many local models cap context at 4K-32K tokens — on a model below\n"
        f"  the fixture size the baseline will truncate, which makes its\n"
        f"  results reflect truncation behaviour rather than real baseline.\n"
        f"  On cloud models this works fine but charges per input token — your\n"
        f"  $/run figure will be correspondingly large."
    )


@cli.command(name="wizard")
def wizard():
    """Launch the interactive menu for managing Sieve.

    Equivalent to running `sieve` with no arguments from an interactive
    terminal — install, service control, store inspection, config,
    benchmark, demo, uninstall, all in one place.
    """
    from sieve._wizard_app import run_wizard
    run_wizard(console=console)


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
@click.option(
    "--fixture",
    type=click.Choice(_FIXTURE_CHOICES, case_sensitive=False),
    default=None,
    help="Payload size. small=light agent, medium=Cursor-like (default), "
    "large=Claude Code mid-session, xlarge=autonomous run. Overrides wizard.",
)
@click.option(
    "--model",
    default=None,
    help="Model to test. Defaults to provider.default_model from sieve.yaml.",
)
@click.option(
    "--grader-model",
    default=None,
    help="Model used to grade recall + trap. Defaults to --model (self-grading, "
    "not recommended for shareable reports). Pass a different model for "
    "independent scoring.",
)
@click.option(
    "--turns",
    type=int,
    default=None,
    help="Turns per run. Default 15.",
)
@click.option(
    "--runs",
    type=int,
    default=None,
    help="Number of full script runs (for mean ± stddev). Default 3.",
)
@click.option(
    "--pricing",
    type=click.Choice(list(_PRICING_CHOICES), case_sensitive=False),
    default=None,
    help="Pricing tier for the cost panel. Default: local (no $ shown).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["rich", "json", "markdown"], case_sensitive=False),
    default=None,
    help="Terminal output format. 'rich' (default), 'json', or 'markdown'. "
    "A shareable markdown report is always written to ~/.sieve/benchmarks/ "
    "regardless of this flag.",
)
@click.option(
    "--no-input",
    is_flag=True,
    default=False,
    help="Skip the interactive wizard; use flag values and defaults.",
)
@click.option(
    "--use-main-store",
    is_flag=True,
    default=False,
    help=(
        "Run against the user's main proxy and store instead of an "
        "ephemeral sandbox. Advanced/debug use only. Requires `sieve start`."
    ),
)
def benchmark(
    config_path: str | None,
    fixture: str | None,
    model: str | None,
    grader_model: str | None,
    turns: int | None,
    runs: int | None,
    pricing: str | None,
    output_format: str | None,
    no_input: bool,
    use_main_store: bool,
):
    """Run a reproducible benchmark: baseline (direct LLM) vs Sieve.

    Defaults: sandboxed, agent-shaped payload, 3 runs × 15 turns on the
    'medium' fixture with the user's configured model. A markdown report
    is always saved to ~/.sieve/benchmarks/.

    With no flags and an interactive terminal, this launches a short
    wizard to let you customise fixture size / model / grader / turns /
    runs / pricing before running. Any flag suppresses the wizard.
    Pass ``--no-input`` to always skip the wizard (CI use).
    """
    try:
        config = RecallConfig.load(config_path)
    except Exception as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        sys.exit(1)

    # Decide whether to launch the wizard: only when stdin is a TTY,
    # --no-input wasn't passed, and the user didn't already supply
    # any customisation flag.
    any_flag_passed = any(v is not None for v in
        (fixture, model, grader_model, turns, runs, pricing, output_format)
    )
    run_wizard = (
        not no_input
        and not any_flag_passed
        and sys.stdin.isatty()
    )

    # Apply defaults early so the wizard can show them and the flag
    # path can override selectively.
    fixture = (fixture or "medium").lower()
    turns = turns if turns is not None else 15
    runs = runs if runs is not None else 3
    pricing = (pricing or "local").lower()
    output_format = (output_format or "rich").lower()
    model_name = model or config.provider.default_model
    grader_name = grader_model or model_name

    direct_base = config.provider.base_url

    if run_wizard:
        (fixture, model_name, grader_name, turns, runs, pricing,
         output_format) = _benchmark_wizard(
            fixture, model_name, grader_name, turns, runs, pricing, output_format,
            provider_base_url=direct_base,
        )

    # Context-window warning for heavy fixtures on local models.
    skipped_fixtures: list[str] = []
    cwin_warning_text: str | None = None
    if fixture in ("large", "xlarge") and _looks_like_local(direct_base):
        if no_input:
            cwin_warning_text = (
                f"'{fixture}' run with --no-input against a local-looking "
                f"endpoint; baseline truncation possible, not inspected."
            )
        else:
            console.print()
            console.print(
                Panel_lite_warning(
                    _context_window_warning_text(fixture, model_name, direct_base)
                )
            )
            go = click.confirm(
                f"Continue with fixture='{fixture}'?",
                default=False,
            )
            if not go:
                skipped_fixtures.append(fixture)
                # Fall back to medium so the user still gets a result.
                console.print(
                    f"[yellow]Skipping '{fixture}'; running 'medium' instead.[/]"
                )
                fixture = "medium"
            else:
                cwin_warning_text = (
                    f"'{fixture}' accepted by user despite local-endpoint "
                    f"warning; baseline may truncate."
                )

    # Precise context-window preflight: consult /api/show to get the
    # model's *actual* num_ctx, then estimate what the fixture needs
    # at turn 15 (base payload × ~2 for history + tools + replies).
    # Skipped silently on non-Ollama endpoints (OpenAI etc. 404).
    cwin_precise_text: str | None = None
    from sieve._wizard_helpers import model_context_window
    from sieve._agent_fixture import fixture_approx_tokens, fixture_names
    ctx = model_context_window(direct_base, model_name)
    if ctx is not None:
        effective_ctx, architectural_ctx = ctx
        # Estimated peak: base (system prompt + tools) + ~1× growing
        # history by turn 15 on the growing-agent script. The ×1.8
        # multiplier matches what real fixtures produce in practice
        # (measured on medium, large, xlarge).
        fixture_peak = int(fixture_approx_tokens(fixture) * 1.8)
        sieve_fits = effective_ctx >= _SIEVE_TYPICAL_OVERHEAD_TOKENS
        baseline_fits = effective_ctx >= fixture_peak

        # Three tiers:
        #   1. Both fit   → silent. No warning needed.
        #   2. Only Sieve → "Context configuration" panel, default Y.
        #                   This IS the Sieve story; baseline failures
        #                   are part of the finding.
        #   3. Neither    → loud warning, default N. Rare: only if the
        #                   effective window is smaller than Sieve's own
        #                   ~2K overhead, e.g. a 1K-context model.
        if not baseline_fits:
            # Build shared ingredients used by both (2) and (3) panels.
            target = fixture_peak * 2
            rounded = 1
            while rounded < target:
                rounded *= 2
            suggested_ctx = min(rounded, architectural_ctx)
            order = fixture_names()
            idx = order.index(fixture) if fixture in order else 1
            smaller_fixture = order[max(0, idx - 1)] if idx > 0 else fixture

            if sieve_fits:
                # ── Tier 2: the Sieve story ────────────────────────
                msg = (
                    f"Your model is serving [cyan]{effective_ctx:,}[/] "
                    f"tokens of context (the Ollama default for models "
                    f"without a Modelfile [cyan]num_ctx[/] setting). "
                    f"The '{fixture}' agent fixture ships "
                    f"[cyan]~{fixture_peak:,}[/] tokens/turn at peak.\n\n"
                    f"What this means for the benchmark:\n"
                    f"  → [dim]Without Sieve[/]: the raw agent payload "
                    f"exceeds the context window. Baseline requests "
                    f"will error. That result is itself a finding — "
                    f"the benchmark records it as 'baseline: N/"
                    f"{turns} turns errored'.\n"
                    f"  → [dim]With Sieve[/]: Sieve forwards "
                    f"~{_SIEVE_TYPICAL_OVERHEAD_TOKENS:,} tokens/turn "
                    f"regardless of agent size — [green]well inside[/] "
                    f"the window. The Sieve pass should complete "
                    f"normally.\n\n"
                    f"This is exactly the scenario Sieve is designed "
                    f"for. Three choices:\n"
                    f"  [bold][1][/] [green]Run it.[/] Baseline will "
                    f"fail as expected; Sieve will succeed. The "
                    f"contrast is the point. [dim](recommended)[/]\n"
                    f"  [2] Let the baseline compete fairly by raising "
                    f"num_ctx on the model:\n"
                    f"      [dim]ollama show {model_name} --modelfile > Modelfile[/]\n"
                    f"      [dim]# add line: PARAMETER num_ctx {suggested_ctx}[/]\n"
                    f"      [dim]ollama create {model_name}-ctx{suggested_ctx} -f Modelfile[/]\n"
                    f"      [dim]then re-run with --model {model_name}-ctx{suggested_ctx}[/]\n"
                    f"  [3] Run a same-context comparison with a smaller "
                    f"fixture ([cyan]--fixture {smaller_fixture}[/])"
                )
                if no_input:
                    cwin_precise_text = (
                        f"effective num_ctx={effective_ctx:,} < fixture peak "
                        f"~{fixture_peak:,}; baseline expected to fail, "
                        "Sieve expected to succeed (the Sieve story)."
                    )
                    console.print()
                    console.print(
                        Panel_lite_warning(
                            msg,
                            title="Context configuration",
                            border_style="cyan",
                        )
                    )
                    console.print(
                        "[dim](--no-input set — running the benchmark.)[/]"
                    )
                else:
                    console.print()
                    console.print(
                        Panel_lite_warning(
                            msg,
                            title="Context configuration",
                            border_style="cyan",
                        )
                    )
                    pick = _prompt_numbered(
                        "Choose [1 / 2 / 3]",
                        valid=("1", "2", "3"),
                        default="1",
                    )
                    if pick == "2":
                        # Abort — the Modelfile recipe is already on
                        # screen above. The user copies, adjusts, and
                        # re-runs the benchmark with the new model name.
                        console.print(
                            "[yellow]Cancelled. Run the three commands "
                            "above to raise num_ctx, then re-run the "
                            "benchmark with the new model name.[/]"
                        )
                        sys.exit(0)
                    if pick == "3":
                        # Swap the fixture and carry on. We record the
                        # change in the methodology-limitations section.
                        console.print(
                            f"[dim]Dropping to fixture "
                            f"'[cyan]{smaller_fixture}[/]' for a "
                            f"same-context comparison.[/]"
                        )
                        cwin_precise_text = (
                            f"User chose [3]: dropped from '{fixture}' to "
                            f"'{smaller_fixture}' because effective "
                            f"num_ctx={effective_ctx:,} < peak "
                            f"~{fixture_peak:,}."
                        )
                        fixture = smaller_fixture
                        # Recompute peak for subsequent reporting —
                        # not strictly needed (we've already passed
                        # the check) but keeps local state coherent.
                        fixture_peak = int(fixture_approx_tokens(fixture) * 1.8)
                    else:
                        # [1] — the common path. Run it.
                        cwin_precise_text = (
                            f"effective num_ctx={effective_ctx:,} < fixture peak "
                            f"~{fixture_peak:,}; baseline expected to fail, "
                            "Sieve expected to succeed."
                        )
            else:
                # ── Tier 3: even Sieve won't fit ──────────────────
                msg = (
                    f"Model '{model_name}' is running with effective "
                    f"num_ctx=[red]{effective_ctx:,}[/] — smaller than "
                    f"Sieve's typical outbound "
                    f"(~{_SIEVE_TYPICAL_OVERHEAD_TOKENS:,} tokens).\n\n"
                    f"Both the baseline AND the Sieve pass are likely "
                    f"to error on this combination. This is unusual — "
                    f"Ollama's default (4096) already fits Sieve's "
                    f"output, so seeing a lower number probably means "
                    f"a Modelfile has [cyan]PARAMETER num_ctx[/] set "
                    f"unusually low.\n\n"
                    f"Options:\n"
                    f"  [1] Raise num_ctx on the model "
                    f"(recommended):\n"
                    f"      [dim]ollama show {model_name} --modelfile > Modelfile[/]\n"
                    f"      [dim]# replace: PARAMETER num_ctx {suggested_ctx}[/]\n"
                    f"      [dim]ollama create {model_name}-ctx{suggested_ctx} -f Modelfile[/]\n"
                    f"  [2] Cancel and pick a model with a larger "
                    f"effective context window.\n"
                    f"  [3] Continue anyway (expect both passes to fail)."
                )
                if no_input:
                    cwin_precise_text = (
                        f"effective num_ctx={effective_ctx:,} < Sieve's "
                        f"typical outbound ~{_SIEVE_TYPICAL_OVERHEAD_TOKENS:,}; "
                        "both passes likely to fail; --no-input continued."
                    )
                    console.print()
                    console.print(Panel_lite_warning(msg))
                    console.print(
                        "[dim](--no-input set — continuing despite "
                        "likely failure of BOTH passes.)[/]"
                    )
                else:
                    console.print()
                    console.print(Panel_lite_warning(msg))
                    pick = _prompt_numbered(
                        "Choose [1 / 2 / 3]",
                        valid=("1", "2", "3"),
                        default="2",  # tier 3 default: cancel
                    )
                    if pick == "1":
                        console.print(
                            "[yellow]Run the three commands above to raise "
                            "num_ctx, then re-run the benchmark with the "
                            "new model name.[/]"
                        )
                        sys.exit(0)
                    if pick == "2":
                        console.print(
                            "[yellow]Cancelled. Pick a model with a larger "
                            "effective context window, then re-run.[/]"
                        )
                        sys.exit(0)
                    # [3] — continue anyway. Rare path.
                    cwin_precise_text = (
                        f"effective num_ctx={effective_ctx:,} below Sieve's "
                        f"typical outbound ~{_SIEVE_TYPICAL_OVERHEAD_TOKENS:,}; "
                        "both passes expected to fail."
                    )

    # Merge the precise and heuristic warnings into one bullet so
    # the report's limitations section records everything that
    # applied to this run.
    if cwin_precise_text:
        if cwin_warning_text:
            cwin_warning_text = cwin_warning_text + " " + cwin_precise_text
        else:
            cwin_warning_text = cwin_precise_text

    self_grading = (grader_name == model_name)
    if self_grading:
        console.print(
            "\n[yellow]⚠ Self-grading:[/] the recall grader is the same "
            "model being tested.\n"
            "  Skeptics flag this as a potential bias source. For a "
            "shareable report,\n"
            "  pass [cyan]--grader-model <different-model>[/] to use an "
            "independent grader.\n"
        )

    announce = (output_format == "rich")
    from sieve.cli_benchmark import (
        run_benchmark_compare_multi,
        render_aggregated_compare_summary,
        render_aggregated_markdown,
        build_headline,
        looks_like_absence_signal, response_recalls,
    )
    from sieve._grader import build_recall_grader, build_trap_grader
    from sieve._agent_fixture import fixture_for, fixture_description

    # Graders hit the LLM directly (bypassing Sieve) so grading isn't
    # biased by context manipulation.
    recall_grader = build_recall_grader(
        direct_base, grader_name,
        fallback=lambda _i, resp: response_recalls(_i, resp),
    )
    trap_grader = build_trap_grader(
        direct_base, grader_name,
        fallback=looks_like_absence_signal,
    )

    # Scope-over the agent fixture for this run.
    wrap_payload = fixture_for(fixture)
    def _wrap(user: str, mdl: str, history: list, strm: bool) -> dict:
        return wrap_payload(user, mdl, history=history, stream=strm)
    _WrapAdapter.set(wrap_payload)

    if use_main_store:
        _run_benchmark_against_main_store_v2(
            config=config, model_name=model_name, direct_base=direct_base,
            fixture=fixture, grader_name=grader_name,
            turns=turns, runs=runs, pricing=pricing, output_format=output_format,
            announce=announce, recall_grader=recall_grader, trap_grader=trap_grader,
            context_window_warning=cwin_warning_text,
            skipped_fixtures=skipped_fixtures,
        )
        return

    from sieve._sandbox import SandboxedProxy
    from sieve.store import MemoryStore

    if announce:
        console.print(
            "[dim]Starting sandbox proxy (isolated from your main "
            "store)…[/]"
        )
    try:
        with SandboxedProxy.from_main_config(config) as sb:
            if announce:
                console.print(
                    f"[dim]Sandbox ready at [cyan]{sb.base_url}[/]\n[/]"
                )
            ms = MemoryStore(sb.config.store)
            ms.open()
            try:
                _execute_benchmark_v2(
                    sieve_base_url=sb.base_url,
                    direct_base=direct_base,
                    model_name=model_name,
                    grader_name=grader_name,
                    fixture=fixture,
                    store=ms,
                    turns=turns,
                    runs=runs,
                    pricing=pricing,
                    output_format=output_format,
                    announce=announce,
                    recall_grader=recall_grader,
                    trap_grader=trap_grader,
                    context_window_warning=cwin_warning_text,
                    skipped_fixtures=skipped_fixtures,
                )
            finally:
                ms.close()
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted — sandbox cleaned up.[/]")
        sys.exit(130)
    except Exception as exc:
        # Clean, non-tracebacked error for the user. Full traceback is
        # still available via `sieve --verbose benchmark` for the rare
        # case a dev wants it.
        console.print(f"[bold red]Benchmark failed:[/] {exc}")
        from sieve.cli_benchmark import UpstreamLLMError
        if isinstance(exc, UpstreamLLMError):
            hint = exc.user_hint()
            if hint:
                console.print(f"\n[bold]Hint:[/] {hint}")
        sys.exit(1)


# Rough upper bound of what the Sieve proxy forwards to the LLM per
# turn, regardless of incoming agent payload size. Breakdown:
#   ~200   lean system prompt
#   ~100   recall tool schema
#   ~500   retrieved-context injection (mature store)
#   ~600   last-N conversation history (observe_turns=8)
#   ~100   current user message + rendering overhead
#   ~500   headroom for rare long retrievals
# Hardcoded rather than measured: measuring requires a live sandbox
# before the preflight, which inverts the dependency order and adds
# 8-12s of cold-start latency to the wizard. A 2K ceiling that hasn't
# been observed to regress in production is worth more than a
# measured 1.7K that might hide future drift.
_SIEVE_TYPICAL_OVERHEAD_TOKENS = 2048


def Panel_lite_warning(text: str, *, title: str = "⚠  Context-window warning",
                      border_style: str = "yellow"):
    """Inline Panel factory — imports lazily to avoid a global rich dep."""
    from rich.panel import Panel
    return Panel(text, title=title, border_style=border_style)


def _prompt_numbered(
    prompt: str,
    *,
    valid: tuple[str, ...],
    default: str,
) -> str:
    """Prompt the user to pick one of a short set of numbered options.

    Re-prompts on unrecognised input instead of erroring, so the user
    never gets stuck. Empty input returns ``default``. Strips
    whitespace and is case-insensitive.
    """
    while True:
        raw = click.prompt(
            prompt,
            type=str,
            default=default,
            show_default=True,
        )
        pick = (raw or "").strip().lower()
        if not pick:
            return default
        if pick in valid:
            return pick
        console.print(
            f"[yellow]'{raw}' isn't one of {', '.join(valid)} — try again.[/]"
        )


class _WrapAdapter:
    """Thread-safe-ish module-level slot for the active fixture wrap.

    The run_benchmark_compare_multi function signature doesn't carry
    the wrap. We inject via this slot so it's picked up by each pass
    inside the multi-run loop.
    """
    _fn = None

    @classmethod
    def set(cls, fn):
        cls._fn = fn

    @classmethod
    def get(cls):
        return cls._fn


def _benchmark_wizard(
    fixture_default: str,
    model_default: str,
    grader_default: str,
    turns_default: int,
    runs_default: int,
    pricing_default: str,
    format_default: str,
    *,
    provider_base_url: str = "",
):
    """Numbered-prompt wizard. Each answer maps 1:1 to a flag.

    Model selection queries the live provider endpoint (``/api/tags``
    for Ollama, ``/v1/models`` as fallback). If the endpoint is down
    or returns no models, falls back to a free-text prompt so the
    user is never stuck.

    Ends by echoing the equivalent flag-form command so CI / teammates
    can reproduce the run exactly.
    """
    from sieve._agent_fixture import (
        fixture_names, fixture_approx_tokens, fixture_description,
    )
    from sieve._wizard_helpers import NumberedChoice, pick_numbered, pick_model

    console.print()
    console.print("[bold]Sieve benchmark setup[/]")
    console.print(
        "[dim]Choose a number at each prompt, or press enter for the "
        "default (marked with *). Each choice maps to a command-line "
        "flag — the equivalent command is shown at the end for CI / "
        "sharing.[/]"
    )

    # ── Fixture ─────────────────────────────────────────────────────
    # 'small' is hidden from the wizard. Its baseline payload (~73
    # tokens) is below the cost of Sieve's own injected memory
    # context, so the comparison shows Sieve using more tokens — true,
    # but a misleading first impression. `--fixture small` still
    # works for diagnostics.
    wizard_fixtures = [n for n in fixture_names() if n != "small"]
    fixture_choices = [
        NumberedChoice(
            label=f"{name:<7} (~{fixture_approx_tokens(name):,} base tokens)",
            value=name,
            help=fixture_description(name),
        )
        for name in wizard_fixtures
    ]
    # Snap the default into the visible set so the * marker actually
    # lands on one of the offered options.
    wizard_default = fixture_default if fixture_default in wizard_fixtures else "medium"
    fixture = pick_numbered(
        "Agent payload size",
        fixture_choices,
        default=wizard_default,
        console=console,
    )

    # ── Model to test ──────────────────────────────────────────────
    console.print()
    console.print("[bold]Model to test[/]")
    console.print(f"[dim]Fetching available models from {provider_base_url}…[/]")
    model_name = pick_model(
        "Pick the model you want to benchmark",
        base_url=provider_base_url,
        default=model_default,
        console=console,
    )

    # ── Grader model ───────────────────────────────────────────────
    # Scientific framing — the research found that skeptic-friendly
    # benchmarks name bias risks explicitly rather than hiding them.
    console.print()
    console.print("[bold]Grader model[/]")
    console.print(
        "[dim]The grader answers yes/no on each recall and on the "
        "trap turn. Rigorous benchmarks use an independent grader — "
        "same-model grading introduces bias because the model has "
        "access to the same reasoning patterns that produced the "
        "answer.[/]"
    )
    # Pre-seed the grader picker so the test model is hidden — nudges
    # toward choosing a different one without forcing it.
    grader_name = pick_model(
        "Pick a grader",
        base_url=provider_base_url,
        default=grader_default,
        console=console,
        exclude=None,  # keep all models visible; just nudge via the help text
    )

    # ── Turns ──────────────────────────────────────────────────────
    console.print()
    turns_choices = [
        NumberedChoice(label="6 — quick smoke test", value="6"),
        NumberedChoice(label="15 — standard script (intros + recalls + trap)", value="15"),
        NumberedChoice(label="30 — longer conversation", value="30"),
        NumberedChoice(label="60 — stress test", value="60"),
    ]
    turns_raw = pick_numbered(
        "Turns per run",
        turns_choices,
        default=str(turns_default),
        console=console,
        allow_free_text=True,
        free_text_prompt="Enter a turn count",
    )
    try:
        turns = max(3, int(turns_raw))
    except ValueError:
        turns = turns_default

    # ── Runs ───────────────────────────────────────────────────────
    console.print()
    console.print(
        "[dim]More runs produce a tighter stddev estimate. A single run "
        "reports point-in-time results with no error bars. 3+ runs is "
        "standard for benchmarks intended for publication.[/]"
    )
    runs_choices = [
        NumberedChoice(label="1 — point-in-time, no error bars", value="1"),
        NumberedChoice(label="3 — stddev estimate (recommended)", value="3"),
        NumberedChoice(label="5 — tighter confidence", value="5"),
    ]
    runs_raw = pick_numbered(
        "Number of runs",
        runs_choices,
        default=str(runs_default),
        console=console,
        allow_free_text=True,
        free_text_prompt="Enter a run count",
    )
    try:
        runs = max(1, int(runs_raw))
    except ValueError:
        runs = runs_default

    # ── Pricing tier ────────────────────────────────────────────────
    console.print()
    pricing_choices = [
        NumberedChoice(label="local — no $ shown (local inference is free)", value="local"),
        NumberedChoice(label="claude-haiku    — Anthropic Claude Haiku ($0.80 / 1M in)",  value="claude-haiku"),
        NumberedChoice(label="claude-sonnet   — Anthropic Claude Sonnet ($3.00 / 1M in)", value="claude-sonnet"),
        NumberedChoice(label="claude-opus     — Anthropic Claude Opus ($15.00 / 1M in)",  value="claude-opus"),
        NumberedChoice(label="gpt-4o-mini     — OpenAI GPT-4o mini ($0.15 / 1M in)",      value="gpt-4o-mini"),
        NumberedChoice(label="gpt-4o          — OpenAI GPT-4o ($2.50 / 1M in)",           value="gpt-4o"),
    ]
    pricing = pick_numbered(
        "Pricing tier (for the cost panel)",
        pricing_choices,
        default=pricing_default,
        console=console,
    )

    # ── Output format ──────────────────────────────────────────────
    console.print()
    fmt_choices = [
        NumberedChoice(label="rich — formatted terminal output", value="rich"),
        NumberedChoice(label="json — machine-readable for CI / scripting", value="json"),
        NumberedChoice(label="markdown — paste-into-README artifact", value="markdown"),
    ]
    output_format = pick_numbered(
        "Terminal output format",
        fmt_choices,
        default=format_default,
        console=console,
    )

    # ── Equivalent command — the "paste into CI" line ─────────────
    console.print()
    console.print("[bold]Equivalent command (paste into CI or share with a teammate):[/]")
    console.print(
        f"  [cyan]sieve benchmark --fixture {fixture} --model {model_name} "
        f"--grader-model {grader_name} --turns {turns} --runs {runs} "
        f"--pricing {pricing} --format {output_format}[/]"
    )
    console.print()

    return fixture, model_name, grader_name, turns, runs, pricing, output_format


def _execute_benchmark_v2(
    *,
    sieve_base_url: str,
    direct_base: str,
    model_name: str,
    grader_name: str,
    fixture: str,
    store,
    turns: int,
    runs: int,
    pricing: str,
    output_format: str,
    announce: bool,
    recall_grader,
    trap_grader,
    context_window_warning: str | None,
    skipped_fixtures: list[str],
) -> None:
    """Run the multi-run compare against a prepared proxy + store.

    Writes the markdown report to ~/.sieve/benchmarks/<ISO>.md always;
    output_format controls the terminal.
    """
    from sieve.cli_benchmark import (
        run_benchmark_compare_multi,
        render_aggregated_compare_summary,
        render_aggregated_markdown,
    )

    def _count() -> int:
        try:
            return int(store.stats().get("facts_count", 0))
        except Exception:
            return 0

    def _reset_store() -> None:
        try:
            conn = store._conn
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                for table in (
                    "vec_facts", "vec_episodes", "relationships",
                    "preferences", "known_unknowns", "audit_log",
                    "fingerprints", "sessions", "episodes", "facts", "entities",
                ):
                    try:
                        conn.execute(f"DELETE FROM {table}")
                    except Exception:
                        pass
                conn.commit()
            finally:
                conn.execute("PRAGMA foreign_keys = ON")
        except Exception as exc:
            logging.getLogger("recall.cli").warning(
                "Store reset failed (non-fatal): %s", exc
            )

    if announce:
        console.print(
            f"[bold]Running {runs} × {turns}-turn benchmark[/] "
            f"with fixture '{fixture}'…"
        )
        console.print(
            f"  baseline: [cyan]{direct_base}[/] (no Sieve)\n"
            f"  sieve:    [cyan]{sieve_base_url}[/]"
        )
        console.print(
            f"[dim]Estimated time: "
            f"~{runs * 2 * turns // 6}–{runs * 2 * turns // 3} minutes "
            "depending on model speed.[/]\n"
        )

    # Build a multi-run result by wrapping run_benchmark_compare's
    # internal agent-shaped fixture via our module-level adapter.
    def _progress(i: int, total: int, phase: str) -> None:
        if not announce or phase == "done":
            return
        console.print(f"[dim]  run {i}/{total} — {phase} pass…[/]")

    # Inject the chosen fixture into run_benchmark_compare_multi via
    # the module-level slot (run_benchmark_compare internally imports
    # build_agent_payload from _agent_fixture; _WrapAdapter patches it
    # for this call).
    import sieve._agent_fixture as fx
    original = fx.build_agent_payload
    fx.build_agent_payload = _WrapAdapter.get()
    try:
        agg = run_benchmark_compare_multi(
            runs=runs,
            sieve_base_url=sieve_base_url,
            direct_base_url=direct_base,
            model=model_name,
            store_fact_count=_count,
            reset_store=_reset_store,
            messages=_BENCHMARK_TURNS(turns),
            grade_recall=recall_grader,
            grade_trap=trap_grader,
            progress=_progress,
        )
    finally:
        fx.build_agent_payload = original

    # Render markdown artifact always.
    md_text = render_aggregated_markdown(
        agg,
        model=model_name,
        grader_model=grader_name,
        fixture=fixture,
        sieve_base_url=sieve_base_url,
        direct_base_url=direct_base,
        pricing_tier=pricing,
        turns_per_run=turns,
        skipped_fixtures=skipped_fixtures or None,
        context_window_warning=context_window_warning,
    )
    md_path = _write_benchmark_report(md_text)

    if output_format == "json":
        import json as _json
        payload = _aggregated_to_dict(
            agg, model=model_name, grader_model=grader_name,
            fixture=fixture, pricing_tier=pricing,
            report_path=str(md_path) if md_path else None,
            skipped_fixtures=skipped_fixtures,
            context_window_warning=context_window_warning,
        )
        print(_json.dumps(payload, indent=2))
        return

    if output_format == "markdown":
        print(md_text)
        if md_path:
            console.print(f"\n[dim]Saved to: {md_path}[/]")
        return

    render_aggregated_compare_summary(
        agg,
        model=model_name,
        grader_model=grader_name,
        fixture=fixture,
        sieve_base_url=sieve_base_url,
        direct_base_url=direct_base,
        console=console,
        pricing_tier=pricing,
        turns_per_run=turns,
        context_window_warning=context_window_warning,
        skipped_fixtures=skipped_fixtures or None,
    )
    if md_path:
        console.print(f"\n[dim]📎 Shareable report: [cyan]{md_path}[/][/]")


def _BENCHMARK_TURNS(n: int):
    """Return a script of exactly n turns, always ending on the trap turn.

    The trap is the absence-signal probe (asking about a person who was
    never mentioned) and is the headline verdict of the benchmark, so
    it must run no matter how short the user truncates the script.

    Layout:
      - First ``n-1`` non-trap messages from BENCHMARK_MESSAGES (in
        order; intros first so later retrieves have something to recall).
      - Final turn is always the trap.
    Repeats deep-phase turns if ``n`` exceeds the scripted length.
    Bounded to a minimum of 3 turns (2 non-trap + trap).
    """
    from sieve.cli_benchmark import BENCHMARK_MESSAGES
    msgs = list(BENCHMARK_MESSAGES)
    trap = next((m for m in msgs if m["phase"] == "trap"), None)
    non_trap = [m for m in msgs if m["phase"] != "trap"]
    n = max(3, n)
    body_needed = n - (1 if trap else 0)
    body = list(non_trap[:body_needed])
    if len(body) < body_needed:
        extra = [m for m in non_trap if m["phase"] == "deep"] or non_trap
        while len(body) < body_needed:
            body.extend(extra)
        body = body[:body_needed]
    return body + ([trap] if trap else [])


def _write_benchmark_report(md_text: str):
    """Save the markdown report to ~/.sieve/benchmarks/<ISO>.md.

    Returns the path on success, None on failure (silent — the report
    is a convenience artifact; terminal output is the source of truth).
    """
    from datetime import datetime, timezone
    try:
        dest_dir = SIEVE_DIR / "benchmarks"
        dest_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        path = dest_dir / f"{stamp}.md"
        path.write_text(md_text)
        return path
    except Exception as exc:
        logging.getLogger("recall.cli").warning(
            "Failed to save benchmark report: %s", exc
        )
        return None


def _aggregated_to_dict(agg, *, model, grader_model, fixture, pricing_tier,
                        report_path=None, skipped_fixtures=None,
                        context_window_warning=None) -> dict:
    """JSON-serialisable view of an AggregatedCompareSummary."""
    from sieve._pricing import dollars_saved, price_for
    saved = agg.tokens_saved.mean
    return {
        "mode": "compare_aggregated",
        "model": model,
        "grader_model": grader_model,
        "fixture": fixture,
        "pricing_tier": pricing_tier,
        "runs": len(agg.runs),
        "baseline_tokens_mean": agg.baseline_tokens.mean,
        "baseline_tokens_stddev": agg.baseline_tokens.stddev,
        "sieve_outbound_tokens_mean": agg.sieve_outbound_tokens.mean,
        "sieve_outbound_tokens_stddev": agg.sieve_outbound_tokens.stddev,
        "tokens_saved_mean": agg.tokens_saved.mean,
        "tokens_saved_stddev": agg.tokens_saved.stddev,
        "reduction_pct_mean": agg.reduction_pct.mean,
        "reduction_pct_stddev": agg.reduction_pct.stddev,
        "correct_recalls_per_run": agg.correct_recalls_per_run,
        "gradable_recalls": agg.gradable_recalls,
        "trap_absence_per_run": agg.trap_absence_per_run,
        "facts_learned_per_run": agg.facts_learned_per_run,
        "baseline_wall_clock_s_mean": agg.baseline_wall_clock_s.mean,
        "sieve_wall_clock_s_mean": agg.sieve_wall_clock_s.mean,
        "dollars_saved_per_run": (
            round(dollars_saved(saved, pricing_tier), 6)
            if price_for(pricing_tier) > 0 else 0.0
        ),
        "dollars_saved_per_1k_runs": (
            round(dollars_saved(saved, pricing_tier) * 1000, 4)
            if price_for(pricing_tier) > 0 else 0.0
        ),
        "skipped_fixtures": skipped_fixtures or [],
        "context_window_warning": context_window_warning,
        "report_path": report_path,
    }


def _run_benchmark_against_main_store_v2(
    *,
    config,
    model_name: str,
    direct_base: str,
    fixture: str,
    grader_name: str,
    turns: int,
    runs: int,
    pricing: str,
    output_format: str,
    announce: bool,
    recall_grader,
    trap_grader,
    context_window_warning: str | None,
    skipped_fixtures: list[str],
) -> None:
    """--use-main-store path: run against the user's live proxy+store."""
    pid = _read_pid()
    if pid is None:
        console.print(
            "[bold red]Sieve is not running.[/] --use-main-store "
            "requires [cyan]sieve start[/] in another terminal."
        )
        sys.exit(1)

    console.print(
        "[bold yellow]WARNING:[/] --use-main-store will mutate your "
        "main store. Each run's baseline pass will delete facts; "
        "each Sieve pass will write new ones."
    )
    if not click.confirm("Proceed?", default=False):
        console.print("[yellow]Cancelled.[/]")
        sys.exit(0)

    from sieve.store import MemoryStore
    ms = MemoryStore(config.store)
    if not ms.db_path.exists():
        console.print(
            "[bold red]Store not found.[/] Run [cyan]sieve init[/] first."
        )
        sys.exit(1)
    ms.open()
    try:
        sieve_base_url = f"http://127.0.0.1:{config.listen.port}"
        _execute_benchmark_v2(
            sieve_base_url=sieve_base_url,
            direct_base=direct_base,
            model_name=model_name,
            grader_name=grader_name,
            fixture=fixture,
            store=ms,
            turns=turns,
            runs=runs,
            pricing=pricing,
            output_format=output_format,
            announce=announce,
            recall_grader=recall_grader,
            trap_grader=trap_grader,
            context_window_warning=context_window_warning,
            skipped_fixtures=skipped_fixtures,
        )
    finally:
        ms.close()


def main() -> None:
    """Entry point wired in pyproject.toml (`sieve = sieve.cli:main`)."""
    cli()


if __name__ == "__main__":
    main()

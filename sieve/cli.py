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

console = Console(width=240)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@click.group()
@click.version_option(package_name="llm-sieve", prog_name="sieve")
def cli():
    """Sieve — Transparent context reduction for LLMs."""
    pass


SIEVE_DIR = Path("~/.sieve").expanduser()
PID_FILE = SIEVE_DIR / "sieve.pid"
LOG_FILE = SIEVE_DIR / "sieve.log"


def _config_exists(config_path: str | None) -> bool:
    if config_path:
        return Path(config_path).expanduser().exists()
    return any(p.exists() for p in (Path("sieve.yaml"), SIEVE_DIR / "sieve.yaml"))


@cli.command()
@click.option("--config", "-c", "config_path", default=None, help="Path to sieve.yaml")
@click.option("--host", default=None, help="Override listen host")
@click.option("--port", "-p", default=None, type=int, help="Override listen port")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def start(config_path: str | None, host: str | None, port: int | None, verbose: bool):
    """Start the Sieve proxy server."""
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

    # Fail fast if the configured port is already bound — Phase 7 Test 5.
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind((config.listen.host, config.listen.port))
    except OSError:
        console.print(
            f"[bold red]Port {config.listen.port} is already in use.[/] "
            f"Use --port to specify another."
        )
        sys.exit(1)
    finally:
        probe.close()

    # Write PID file so `sieve stop` / `sieve status` can find us.
    SIEVE_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

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
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours
    return True


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    if not _pid_alive(pid):
        # Stale PID file — clean it up.
        PID_FILE.unlink(missing_ok=True)
        return None
    return pid


@cli.command()
def status():
    """Show Sieve proxy and store status."""
    pid = _read_pid()

    if not _config_exists(None):
        console.print("[bold red]Sieve is not configured.[/] Run [cyan]sieve init[/].")
        return

    # Try to read config; never crash if it's broken — just report what we can.
    try:
        config = RecallConfig.load()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        return

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
        ms = MemoryStore(config.store)
        if ms.db_path.exists():
            ms.open()
            if ms.is_initialized():
                s = ms.stats()
                console.print(
                    f"  Store: [cyan]{s['facts_count']}[/] facts, "
                    f"[cyan]{s['entities_count']}[/] entities"
                )
            ms.close()
        else:
            console.print("  Store: [dim]not initialised[/]")
    except Exception as exc:
        console.print(f"  Store: [red]error[/] {exc}")


@cli.command()
def stop():
    """Gracefully stop the Sieve proxy."""
    pid = _read_pid()
    if pid is None:
        if PID_FILE.exists():
            PID_FILE.unlink(missing_ok=True)
            console.print(
                "[yellow]Sieve was not running (cleaned up stale state).[/]"
            )
        else:
            console.print("[yellow]Sieve is not running.[/]")
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        console.print("[yellow]Sieve was not running.[/]")
        return

    # Wait up to 5s for graceful shutdown.
    import time
    for _ in range(50):
        if not _pid_alive(pid):
            PID_FILE.unlink(missing_ok=True)
            console.print("[bold green]Sieve stopped.[/]")
            return
        time.sleep(0.1)

    console.print("[bold yellow]Sieve did not exit within 5s — leaving PID file.[/]")


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
    from sieve.backup import list_backups
    from sieve.store import MemoryStore

    config = RecallConfig.load(config_path)
    ms = MemoryStore(config.store)
    backups = list_backups(ms.db_path)

    if not backups:
        console.print("[dim]No backups found.[/]")
        return

    console.print(f"[bold green]{len(backups)} backup(s) found[/]")
    for b in backups:
        size_kb = b["size_bytes"] / 1024
        status = "[green]OK[/green]" if b["checksum_valid"] else "[yellow]unverified[/yellow]"
        console.print(f"  {b['timestamp']}  {size_kb:.1f} KB  {status}  {b['id']}")


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


@cli.command()
@click.option("--provider", default=None, help="LLM provider base URL (e.g. http://localhost:11434)")
@click.option("--force", is_flag=True, help="Reinitialise even if ~/.sieve already exists")
def init(provider: str | None, force: bool):
    """Initialise Sieve — creates ~/.sieve/, downloads the embedding model,
    writes a default config, and initialises the encrypted memory store.
    """
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

    # --- Step 2: health-check the provider (warn but don't block) ---
    import httpx
    try:
        r = httpx.get(f"{provider.rstrip('/')}/api/tags", timeout=3.0)
        if r.status_code == 200:
            console.print(f"[green]Provider reachable at[/] [cyan]{provider}[/]")
        else:
            console.print(
                f"[yellow]Provider responded {r.status_code} — will retry at start.[/]"
            )
    except Exception as exc:
        console.print(
            f"[yellow]Could not reach provider ({exc}).[/] "
            f"Continuing — you can fix this in {cfg_path} later."
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
        f"Start Sieve with: [cyan]sieve start[/]  "
        f"(point your agent at [cyan]http://localhost:{config.listen.port}[/])"
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
def demo():
    """Run a short scripted demo against a running Sieve proxy."""
    pid = _read_pid()
    if pid is None:
        console.print(
            "[bold red]Sieve is not running.[/] Start it in another terminal with [cyan]sieve start[/]."
        )
        sys.exit(1)

    import httpx

    try:
        config = RecallConfig.load()
    except Exception as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        sys.exit(1)

    base = f"http://127.0.0.1:{config.listen.port}"
    messages = [
        "Hi, I'm Casey. I work as a landscape architect.",
        "My favourite project so far is the riverside park in Bristol.",
        "I have a dog called Mabel, she's a border terrier.",
        "Do you remember where I work?",
        "What breed is Mabel?",
        "Do you remember where Pat works?",  # absence-signal trap
    ]

    console.print("[bold]Sieve demo[/] — 6 messages through the proxy:\n")
    for i, msg in enumerate(messages, 1):
        console.print(f"[dim]turn {i}:[/] [cyan]{msg}[/]")
        payload = {
            "model": config.provider.default_model,
            "messages": [{"role": "user", "content": msg}],
            "stream": False,
        }
        try:
            r = httpx.post(f"{base}/api/chat", json=payload, timeout=60.0)
            r.raise_for_status()
            data = r.json()
            text = (data.get("message") or {}).get("content", "")[:240]
            rounds = r.headers.get("X-Sieve-Rounds", "0")
            proxy_us = r.headers.get("X-Sieve-Proxy-Us", "?")
            console.print(
                f"        [green]→[/] {text}  "
                f"[dim](recall rounds: {rounds}, proxy_us: {proxy_us})[/]"
            )
        except Exception as exc:
            console.print(f"        [red]error:[/] {exc}")
        console.print()

    console.print(
        "[dim]Check [cyan]sieve status[/] to see how many facts Sieve learned.[/]"
    )


def main() -> None:
    """Entry point wired in pyproject.toml (`sieve = sieve.cli:main`)."""
    cli()


if __name__ == "__main__":
    main()

"""Ephemeral sandboxed proxy for the benchmark and demo commands.

Both commands need a running Sieve proxy with an empty store. Doing
that against the user's real proxy + store would either (a) pollute
their profile with Sam/Alex/Luna fixture data or (b) outright wipe it
if the command resets the store between runs. That risk is
unacceptable for a tool that stores personal context.

The SandboxedProxy context manager solves this by spawning a fresh,
isolated proxy on a free port, backed by a disposable encrypted store
in a scratch directory. Nothing touches ~/.sieve/.

Layout of a live sandbox::

    ~/.cache/sieve/sandbox-<uuid>/
        sieve.yaml           # scratch config (inherits user's provider/model)
        .sieve_key           # fresh SQLCipher keyfile
        .sieve_auth_token    # fresh auth token
        memory.db            # empty encrypted store
        proxy.pid            # PID of the live uvicorn process
        proxy.log            # stdout+stderr of the sandbox proxy

Teardown happens on context-manager exit, SIGINT, SIGTERM, and normal
process exit (atexit). The sweep_orphan_sandboxes helper cleans up
directories left behind by a hard kill (kill -9 / OS crash) by
checking pidfiles against live processes.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from sieve.config import RecallConfig

logger = logging.getLogger("recall.sandbox")


SANDBOX_ROOT = Path("~/.cache/sieve").expanduser()
SANDBOX_PREFIX = "sandbox-"
# If a sandbox directory has no live pidfile AND hasn't been modified
# in this long, the sweeper treats it as orphaned and removes it. Long
# enough to tolerate a slow benchmark (2-5 min) plus a bit of headroom.
_ORPHAN_AGE_S = 900  # 15 minutes
# Max seconds to wait for the sandbox proxy to answer /sieve/health.
# First launch pays FastEmbed warmup + reranker load (~8-12s on a
# modest VM). Keep the budget generous but bounded.
_READY_TIMEOUT_S = 60.0
_READY_POLL_S = 0.3
# Graceful stop budget. The uvicorn process has to close the
# embedding/reranker threads and flush sqlite — give it some time
# before escalating to SIGKILL.
_STOP_GRACE_S = 10.0


def _pick_free_port() -> int:
    """Return a port the OS considers free right now.

    Race-safe enough for our use: bind-then-release. The process is
    about to spawn uvicorn which will bind the same port within a few
    seconds. Another process COULD steal it in that window, but the
    proxy's own `bind()` would then fail fast and we'd report the
    collision rather than corrupting state.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pid_alive(pid: int) -> bool:
    """True if the given PID is a live process we can see."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but belongs to another user. Treat as live;
        # we'd rather keep an unrelated process than kill it.
        return True
    except OSError:
        return False
    return True


def _is_sandbox_alive(sandbox_dir: Path) -> bool:
    """Does the sandbox dir have a pidfile pointing at a live process?"""
    pidfile = sandbox_dir / "proxy.pid"
    if not pidfile.exists():
        return False
    try:
        pid = int(pidfile.read_text().strip())
    except (ValueError, OSError):
        return False
    return _pid_alive(pid)


def sweep_orphan_sandboxes(root: Path = SANDBOX_ROOT) -> list[Path]:
    """Remove sandbox dirs left behind by a crashed/killed proxy.

    Returns the list of removed paths for logging. A sandbox is
    considered orphaned when ONE of:
      - pidfile exists but points at a dead PID
      - pidfile is absent AND the directory hasn't been touched in
        _ORPHAN_AGE_S (defensive fallback for a proxy that died
        before writing its pidfile)

    Live sandboxes are always spared.
    """
    removed: list[Path] = []
    if not root.exists():
        return removed
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(SANDBOX_PREFIX):
            continue
        if _is_sandbox_alive(child):
            continue
        pidfile = child / "proxy.pid"
        if not pidfile.exists():
            # Check age — don't clobber a directory we're about to use
            # but that hasn't written its pidfile yet.
            try:
                age = time.time() - child.stat().st_mtime
            except OSError:
                age = _ORPHAN_AGE_S + 1
            if age < _ORPHAN_AGE_S:
                continue
        try:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(child)
            logger.info("swept orphan sandbox %s", child)
        except Exception as exc:
            logger.warning("failed to sweep %s: %s", child, exc)
    return removed


@dataclass(frozen=True)
class SandboxHandle:
    """Read-only view of a live sandbox, returned by __enter__."""

    base_url: str                 # e.g. "http://127.0.0.1:11499"
    port: int
    directory: Path               # ~/.cache/sieve/sandbox-<uuid>/
    config_path: Path             # sandbox yaml
    store_path: Path              # sandbox memory.db
    config: RecallConfig          # fully-resolved RecallConfig of the sandbox
    provider_base_url: str        # the user's LLM endpoint (for bypass calls)


class SandboxedProxy:
    """Context manager: spawn an isolated proxy, tear down on exit.

    Typical use::

        with SandboxedProxy.from_main_config(main_config) as sandbox:
            run_benchmark(base_url=sandbox.base_url, ...)
        # sandbox dir + proxy gone here

    Teardown runs on:
      - normal context exit
      - SIGINT / SIGTERM (Ctrl-C or external kill)
      - interpreter shutdown (atexit) — belt-and-braces for code paths
        that forget the `with` block

    The teardown is idempotent: calling it twice is a no-op.
    """

    def __init__(self, *, main_config: RecallConfig, keep_on_fail: bool = False):
        self._main_config = main_config
        self._keep_on_fail = keep_on_fail
        self._dir: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._port: int | None = None
        self._prior_signal_handlers: dict[int, object] = {}
        self._torn_down = False

    # ── Construction helpers ────────────────────────────────────────

    @classmethod
    def from_main_config(
        cls, main_config: RecallConfig, *, keep_on_fail: bool = False
    ) -> "SandboxedProxy":
        """Create a sandbox that inherits the user's provider settings."""
        return cls(main_config=main_config, keep_on_fail=keep_on_fail)

    # ── Context-manager protocol ────────────────────────────────────

    def __enter__(self) -> SandboxHandle:
        sweep_orphan_sandboxes()

        SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
        directory = SANDBOX_ROOT / f"{SANDBOX_PREFIX}{uuid.uuid4().hex[:12]}"
        directory.mkdir(parents=True)
        self._dir = directory

        # Fresh keyfile + auth token so this sandbox can't collide
        # with the user's main install.
        (directory / ".sieve_key").write_bytes(os.urandom(32).hex().encode())
        (directory / ".sieve_auth_token").write_bytes(
            os.urandom(24).hex().encode()
        )

        # Pick a port, synthesise a config.
        self._port = _pick_free_port()
        store_path = directory / "memory.db"
        config_path = directory / "sieve.yaml"
        _write_sandbox_yaml(
            main_config=self._main_config,
            target_path=config_path,
            store_path=store_path,
            port=self._port,
            auth_token_file=directory / ".sieve_auth_token",
            key_file=directory / ".sieve_key",
        )

        # Initialise the store at the sandbox path so the proxy opens
        # cleanly instead of having to bootstrap on first request.
        _init_sandbox_store(config_path)

        # Install signal + atexit handlers BEFORE spawning the child
        # so a fast SIGINT still gets a clean teardown.
        self._install_signal_handlers()
        atexit.register(self._atexit_teardown)

        # Spawn the proxy.
        log_path = directory / "proxy.log"
        log_fh = open(log_path, "ab", buffering=0)
        env = os.environ.copy()
        env["SIEVE_CONFIG"] = str(config_path)
        # The store derives its keyfile path from db_path.parent, so the
        # sandbox's .sieve_key (written alongside memory.db) is already
        # on the expected path. No env-var plumbing required.
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "sieve.main:app",
                    "--host", "127.0.0.1",
                    "--port", str(self._port),
                    "--log-level", "warning",
                ],
                stdout=log_fh,
                stderr=log_fh,
                stdin=subprocess.DEVNULL,
                env=env,
                # New process group so SIGINT to the CLI doesn't
                # propagate to the child here — we explicitly manage
                # teardown via self._install_signal_handlers.
                start_new_session=True,
            )
        finally:
            log_fh.close()

        # Write the pidfile atomically for the orphan sweeper.
        (directory / "proxy.pid").write_text(str(self._proc.pid))

        # Wait for /sieve/health. If we time out, tear down and raise.
        base_url = f"http://127.0.0.1:{self._port}"
        try:
            _wait_for_health(base_url, proc=self._proc)
        except Exception:
            self._teardown()
            raise

        # Build the handle. The resolved config is re-read from disk
        # so callers see exactly what the sandbox proxy sees.
        resolved = RecallConfig.load(str(config_path))
        return SandboxHandle(
            base_url=base_url,
            port=self._port,
            directory=directory,
            config_path=config_path,
            store_path=store_path,
            config=resolved,
            provider_base_url=self._main_config.provider.base_url,
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self._keep_on_fail:
            logger.warning(
                "keeping sandbox %s for inspection (keep_on_fail)", self._dir
            )
            self._restore_signal_handlers()
            self._torn_down = True
            return
        self._teardown()

    # ── Internal teardown ───────────────────────────────────────────

    def _teardown(self) -> None:
        """Idempotent: stop child, remove directory, restore signals."""
        if self._torn_down:
            return
        self._torn_down = True

        self._restore_signal_handlers()

        proc = self._proc
        if proc is not None:
            self._proc = None
            if proc.poll() is None:
                # Still running — try graceful SIGTERM first.
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=_STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "sandbox proxy did not stop within %.1fs, SIGKILL",
                        _STOP_GRACE_S,
                    )
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=3.0)

        if self._dir is not None and self._dir.exists():
            d = self._dir
            self._dir = None
            shutil.rmtree(d, ignore_errors=True)

    def _atexit_teardown(self) -> None:
        """atexit hook — belt-and-braces for non-exception exits."""
        with contextlib.suppress(Exception):
            self._teardown()

    # ── Signal handling ─────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        """Install SIGINT / SIGTERM handlers that tear down before raising.

        We save the prior handlers so we can restore them on clean
        teardown — this avoids clobbering the CLI's own Ctrl-C
        behaviour after the sandbox is gone.
        """
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prior = signal.getsignal(sig)
                self._prior_signal_handlers[sig] = prior

                def _handler(signum, _frame, _prior=prior):
                    try:
                        self._teardown()
                    finally:
                        # Re-raise as the default behaviour: SIGINT
                        # becomes KeyboardInterrupt, SIGTERM exits.
                        if signum == signal.SIGINT:
                            raise KeyboardInterrupt()
                        sys.exit(128 + signum)

                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Not always installable (e.g. non-main thread);
                # atexit still catches clean exits.
                pass

    def _restore_signal_handlers(self) -> None:
        for sig, prior in list(self._prior_signal_handlers.items()):
            with contextlib.suppress(Exception):
                signal.signal(sig, prior)  # type: ignore[arg-type]
        self._prior_signal_handlers.clear()


# ── Config + store scaffolding ─────────────────────────────────────────


def _write_sandbox_yaml(
    *,
    main_config: RecallConfig,
    target_path: Path,
    store_path: Path,
    port: int,
    auth_token_file: Path,
    key_file: Path,
) -> None:
    """Render a sieve.yaml for the sandbox proxy.

    Inherits the user's provider / model / embeddings / writer /
    progression / ablation settings so the sandbox behaves exactly like
    their install, only differing in:
      - listen.port (unique per sandbox)
      - store.path (scratch DB)
      - security.auth_token (fresh)

    We serialise via yaml rather than reaching into RecallConfig's
    private state — field names match the YAML schema in
    RecallConfig.load, so the round-trip is safe.
    """
    m = main_config
    doc: dict = {
        "listen": {"host": "127.0.0.1", "port": port},
        "provider": {
            "type": m.provider.type,
            "base_url": m.provider.base_url,
            "default_model": m.provider.default_model,
            "options": dict(m.provider.options or {}),
        },
        "embeddings": {
            "provider": m.embeddings.provider,
        },
        "store": {"path": str(store_path)},
        "pipeline": {
            "conversation_turns": m.pipeline.conversation_turns,
            "max_rounds": m.pipeline.max_rounds,
            "core_facts_size": m.pipeline.core_facts_size,
            "max_outbound_tokens": m.pipeline.max_outbound_tokens,
            "think_enabled": m.pipeline.think_enabled,
            "context_format": m.pipeline.context_format,
        },
        "progression": {
            "phase_1_threshold": m.progression.phase_1_threshold,
            "phase_2_threshold": m.progression.phase_2_threshold,
            "observe_turns": m.progression.observe_turns,
            "accumulate_turns": m.progression.accumulate_turns,
            "activate_turns": m.progression.activate_turns,
        },
        "writer": {
            "model": m.writer.model,
            "fallback_model": m.writer.fallback_model,
            "num_ctx": m.writer.num_ctx,
            "ghost_validator_enabled": m.writer.ghost_validator_enabled,
        },
        "tools": {
            "enabled": m.tools.enabled,
            "compression": m.tools.compression,
            "fallback_include_all": m.tools.fallback_include_all,
            "max_tools_injected": m.tools.max_tools_injected,
        },
        "security": {
            # Disable auth for the sandbox — the benchmark and demo
            # send plain requests and the sandbox is localhost-only.
            "auth_token": None,
            "allowed_origins": ["127.0.0.1"],
        },
        "ablation": {
            # Mirror the user's ablation flags so the benchmark tests
            # their actual configured feature set.
            "fingerprinting":        m.ablation.fingerprinting,
            "classifier":            m.ablation.classifier,
            "pre_populate":          m.ablation.pre_populate,
            "graph_traversal":       m.ablation.graph_traversal,
            "temporal_versioning":   m.ablation.temporal_versioning,
            "learning_loop":         m.ablation.learning_loop,
            "coherence_integrity":   m.ablation.coherence_integrity,
            "stage2_writer":         m.ablation.stage2_writer,
            "recall_tool":           m.ablation.recall_tool,
            "absence_signal":        m.ablation.absence_signal,
            "closed_world":          m.ablation.closed_world,
            "response_verification": m.ablation.response_verification,
            "schema_v2":             m.ablation.schema_v2,
            "tier2_classifier":      m.ablation.tier2_classifier,
            "extreme_summary":       m.ablation.extreme_summary,
        },
    }
    # Carry API key through if present (cloud users). We deliberately
    # DON'T copy it into the dict above because provider.api_key is
    # usually None and we don't want to write "null" into the YAML.
    if getattr(m.provider, "api_key", None):
        doc["provider"]["api_key"] = m.provider.api_key

    with open(target_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def _init_sandbox_store(config_path: Path) -> None:
    """Open the sandbox store once so init_schema runs at the right dim.

    Loads the sandbox YAML (not the user's!), opens a MemoryStore
    against it, runs init_schema, closes. Honours the FastEmbed →
    384-dim override applied by RecallConfig.load — the same fix
    shipped in the 'wizard baked 768-dim vec_facts' commit.
    """
    from sieve.store import MemoryStore
    cfg = RecallConfig.load(str(config_path))
    ms = MemoryStore(cfg.store)
    ms.open()
    if not ms.is_initialized():
        ms.init_schema()
    ms.close()


# ── Health wait ────────────────────────────────────────────────────────


def _wait_for_health(base_url: str, *, proc: subprocess.Popen) -> None:
    """Block until the sandbox proxy answers /sieve/health, or time out."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    health_url = f"{base_url}/sieve/health"
    while time.monotonic() < deadline:
        # If the child died during startup, stop waiting immediately.
        if proc.poll() is not None:
            raise RuntimeError(
                f"sandbox proxy exited with code {proc.returncode} "
                f"before becoming healthy — check proxy.log"
            )
        try:
            r = httpx.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
            pass
        except Exception as exc:
            logger.debug("health probe error: %s", exc)
        time.sleep(_READY_POLL_S)
    raise TimeoutError(
        f"sandbox proxy did not become ready within {_READY_TIMEOUT_S:.0f}s"
    )

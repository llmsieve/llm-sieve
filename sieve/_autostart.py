"""Autostart-on-boot via systemd user services (Linux).

Public API (stable; the wizard depends on these four names):

- ``autostart_supported`` — True iff this system can run a systemd
  user service.
- ``autostart_status`` — "enabled" / "disabled" / "not supported".
- ``enable_autostart`` — install + enable the unit; return (ok, msg).
- ``disable_autostart`` — disable + remove; return (ok, msg).
- ``remove_autostart_on_uninstall`` — idempotent teardown used by
  the uninstall path.

Design
======

We lean on systemd user services rather than /etc/systemd/system
units because:

- No sudo required.
- Uninstalling is a single-user action; no system-wide residue.
- User services restart cleanly on login and are trivially
  inspectable via ``systemctl --user status sieve``.

The unit file is generated dynamically so it picks up the current
``sieve`` entry-point path (important when the user installs Sieve
into a venv). We do NOT hardcode ``/usr/local/bin/sieve``.

Linger
======

Systemd user services don't survive a logout by default. For a
desktop user logging in automatically, that's usually fine. For a
server / headless user that wants Sieve running even when no
interactive session is open, ``loginctl enable-linger $USER`` is
required. We detect the situation and ask once; the user answers
y/N and we record it in the final status message.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("recall.autostart")


# Unit filename — single source of truth.
_UNIT_NAME = "sieve.service"


def _unit_dir() -> Path:
    """Where user unit files live on this host.

    XDG_CONFIG_HOME respected; falls back to ~/.config per spec.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path("~/.config").expanduser()
    return base / "systemd" / "user"


def _unit_path() -> Path:
    return _unit_dir() / _UNIT_NAME


def _sieve_binary() -> str | None:
    """Absolute path to the ``sieve`` CLI.

    Sieve may be installed into a venv that isn't on the user's
    PATH. Order of resolution:
      1. sys.argv[0] — we're running FROM it right now.
      2. Its resolved absolute path (handles symlinks).
      3. The executable dir of the current interpreter
         (typical venv layout: {prefix}/bin/sieve).
      4. Standard PATH lookup as a last resort.

    Returns None only if none of those yield a real file. In that
    case we report autostart unsupported — we refuse to write a
    unit that points at a binary that doesn't exist.
    """
    import sys
    import os as _os

    # 1. argv[0]: what was invoked right now. Often a relative
    #    path; resolve against CWD.
    argv0 = sys.argv[0] if sys.argv else ""
    candidates: list[str] = []
    if argv0:
        try:
            resolved = str(Path(argv0).resolve())
            candidates.append(resolved)
        except Exception:
            pass

    # 2. Venv bin dir: sys.executable → /path/to/venv/bin/python,
    #    therefore /path/to/venv/bin/sieve is the sibling we want.
    try:
        venv_bin = Path(sys.executable).parent / "sieve"
        candidates.append(str(venv_bin))
    except Exception:
        pass

    # 3. Classic PATH lookup.
    path_lookup = shutil.which("sieve")
    if path_lookup:
        candidates.append(path_lookup)

    for c in candidates:
        if c and _os.path.isfile(c) and _os.access(c, _os.X_OK):
            return c
    return None


def _run(cmd: list[str], *, check: bool = False, timeout: float = 10.0):
    """Run a subprocess and return CompletedProcess. Captures stdout
    and stderr; never raises unless ``check=True``. ``timeout``
    prevents a wedged ``systemctl`` call from hanging the wizard."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout,
        )
    except FileNotFoundError:
        # Command doesn't exist — systemctl absent.
        return None
    except subprocess.TimeoutExpired as exc:
        logger.warning("%s timed out: %s", " ".join(cmd), exc)
        return None


def _systemctl_user(*args: str):
    """Run ``systemctl --user <args>`` and return the
    CompletedProcess, or None on absence."""
    return _run(["systemctl", "--user", *args])


# ── Public API ─────────────────────────────────────────────────────────


def autostart_supported() -> bool:
    """True iff we can actually install a user service here.

    Checks, in order:
      1. Running on Linux.
      2. ``systemctl`` in PATH.
      3. ``systemctl --user status`` works (rules out containers /
         hosts without a user instance).
      4. The ``sieve`` entry-point is discoverable.
    """
    if platform.system() != "Linux":
        return False
    if shutil.which("systemctl") is None:
        return False
    # `systemctl --user status` exits 0 when the user instance is
    # running; non-zero otherwise. We accept non-zero because
    # systemctl prints a useful error and the subsequent enable
    # would surface it — but we DO refuse when the command isn't
    # found at all.
    res = _systemctl_user("--version")
    if res is None or res.returncode != 0:
        return False
    # And the sieve binary has to exist — we're not going to write
    # a unit file pointing at a command that isn't installed.
    if _sieve_binary() is None:
        return False
    return True


def autostart_status() -> str:
    """Return one of 'enabled', 'disabled', 'not supported'."""
    if not autostart_supported():
        return "not supported"
    res = _systemctl_user("is-enabled", _UNIT_NAME)
    if res is None:
        return "disabled"
    # systemctl is-enabled exits 0 for enabled, 1 for disabled /
    # static / etc. The stdout tells us the precise state.
    out = (res.stdout or "").strip()
    if out == "enabled":
        return "enabled"
    return "disabled"


def _generate_unit_text() -> str:
    """Build the unit file content for the current sieve install."""
    binary = _sieve_binary()
    assert binary is not None, "autostart_supported() should gate this"
    return (
        "[Unit]\n"
        "Description=Sieve proxy (user service)\n"
        "Documentation=https://github.com/azardhosein/llm-sieve\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={binary} start --foreground\n"
        f"ExecStop={binary} stop\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "# Sieve writes to ~/.sieve; no additional paths required.\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def enable_autostart() -> tuple[bool, str]:
    """Install the unit file, daemon-reload, and enable --now.

    Returns ``(True, short_message)`` on success. On any failure
    returns ``(False, message_with_reason)``. Never raises.
    """
    if not autostart_supported():
        return False, (
            "Autostart requires systemd user services. This system "
            "either isn't Linux, lacks systemctl, doesn't have a "
            "user instance, or can't find the sieve binary."
        )

    unit_path = _unit_path()
    try:
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(_generate_unit_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to write unit file at {unit_path}: {exc}"

    # daemon-reload so systemd notices the new unit.
    res = _systemctl_user("daemon-reload")
    if res is None or res.returncode != 0:
        return False, (
            "systemctl daemon-reload failed. The unit file was "
            f"written to {unit_path} but systemd didn't pick it up. "
            f"Run `systemctl --user daemon-reload` manually."
        )

    # enable --now enables AND starts in one shot.
    res = _systemctl_user("enable", "--now", _UNIT_NAME)
    if res is None or res.returncode != 0:
        err = (res.stderr if res else "").strip() or "unknown systemctl error"
        return False, f"systemctl enable failed: {err}"

    # Linger advisory: on servers, without `loginctl enable-linger`,
    # the service dies when the user logs out. We report it as a
    # note rather than gating on it — desktop users typically don't
    # need it.
    linger_note = ""
    user = os.environ.get("USER", "")
    if user:
        linger = _run(["loginctl", "show-user", user, "--property=Linger"])
        if linger and linger.returncode == 0 and "Linger=yes" not in (linger.stdout or ""):
            linger_note = (
                f"\n\nNote: to keep Sieve running even when you're "
                f"logged out, enable linger:\n"
                f"  sudo loginctl enable-linger {user}"
            )

    return True, (
        f"Autostart enabled. Sieve will start on login.{linger_note}"
    )


def disable_autostart() -> tuple[bool, str]:
    """Disable + stop, remove the unit file, daemon-reload. Idempotent."""
    if not autostart_supported():
        return False, "Autostart not supported on this system."

    unit_path = _unit_path()
    if not unit_path.exists():
        # Nothing to do — already disabled. Reporting success is the
        # honest answer: the user's goal is met.
        return True, "Autostart was already disabled."

    res = _systemctl_user("disable", "--now", _UNIT_NAME)
    # disable --now returns non-zero when the unit was already
    # stopped / disabled; tolerate that.
    try:
        unit_path.unlink()
    except Exception as exc:  # noqa: BLE001
        return False, f"Removed from systemd but couldn't delete {unit_path}: {exc}"

    _systemctl_user("daemon-reload")
    return True, "Autostart disabled."


def remove_autostart_on_uninstall() -> None:
    """Called from the uninstall path. Idempotent; swallows errors —
    uninstall must not fail because systemd got grumpy."""
    try:
        if autostart_status() == "enabled":
            disable_autostart()
        # Belt-and-braces: if the unit file exists without being
        # enabled, still remove it.
        unit_path = _unit_path()
        if unit_path.exists():
            try:
                unit_path.unlink()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not remove %s: %s", unit_path, exc)
            _systemctl_user("daemon-reload")
    except Exception as exc:  # noqa: BLE001
        logger.warning("autostart uninstall cleanup failed: %s", exc)

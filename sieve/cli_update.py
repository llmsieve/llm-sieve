"""`sieve update` — on-request check for a newer Sieve release on PyPI.

Design constraints:
- Zero telemetry by default. This command ONLY runs when the user
  invokes it explicitly. There is no background check, no
  hook in `sieve start`, no daemon-side polling.
- Network failure is non-fatal. If PyPI is unreachable, print a
  friendly message and exit 0 (user's local Sieve is fine).
- Single round-trip. One HTTP GET to pypi.org/pypi/llm-sieve/json,
  5s timeout. Nothing else.
- Output is small and copy-pasteable. The user can read the new
  version, see the exact upgrade command for their install path,
  and visit the release notes.
"""
from __future__ import annotations

from typing import Optional

import httpx
from rich.console import Console

PYPI_URL = "https://pypi.org/pypi/llm-sieve/json"
PYPI_TIMEOUT_S = 5.0


def _parse_version(v: str):
    """Parse a version string, returning a comparable object.

    Falls back to a string-tuple comparison if `packaging` is not
    available. `packaging` is already a transitive dep of pip/pipx,
    so it's almost always present, but be defensive.
    """
    try:
        from packaging.version import Version
        return Version(v)
    except Exception:
        return tuple(v.split("."))


def _detect_install_path() -> str:
    """Best-effort guess of how Sieve was installed.

    Used only to put the most-likely upgrade command first in the
    output. We always show both pipx and pip variants so the user
    can pick if our guess is wrong.

    Returns "pipx", "pip", or "unknown".
    """
    import sys
    from pathlib import Path
    try:
        exe = Path(sys.executable).resolve()
        # pipx venvs live under ~/.local/share/pipx/venvs/<package>/
        # (Linux) or ~/Library/Application Support/pipx/venvs/<package>/
        # (macOS). Either way, "pipx" appears in the path.
        parts = exe.parts
        if "pipx" in parts:
            return "pipx"
        # Otherwise assume pip-in-venv (we can't reliably distinguish
        # plain pip from pip-in-venv without more probing).
        return "pip"
    except Exception:
        return "unknown"


def _format_release_date(iso_ts: Optional[str]) -> str:
    """Render a PyPI ISO timestamp as YYYY-MM-DD, or empty string."""
    if not iso_ts:
        return ""
    try:
        # PyPI returns e.g. "2026-09-14T10:33:21.123456Z" or similar.
        return iso_ts[:10]
    except Exception:
        return ""


def run_update_check(
    console: Console | None = None,
    pypi_url: str = PYPI_URL,
    timeout_s: float = PYPI_TIMEOUT_S,
) -> int:
    """Check PyPI for a newer release; print a result panel.

    Returns:
      0 — user is on latest, or PyPI unreachable (non-fatal)
      0 — newer version available (informational, not an error)
      1 — invalid response from PyPI (genuinely unexpected)
    """
    console = console or Console()

    from sieve import __version__ as installed_v
    installed = installed_v.strip() or "unknown"

    console.print(f"Sieve {installed} (installed)")
    console.print(f"[dim]Checking {pypi_url}…[/]")

    try:
        resp = httpx.get(pypi_url, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        console.print(
            "[yellow]PyPI request timed out.[/] "
            "Couldn't check for updates — your local Sieve is unaffected."
        )
        return 0
    except httpx.HTTPError as exc:
        console.print(
            f"[yellow]Couldn't reach PyPI:[/] {exc}\n"
            "Your local Sieve is unaffected."
        )
        return 0
    except ValueError:
        console.print("[red]Invalid response from PyPI.[/]")
        return 1

    info = data.get("info") or {}
    latest = info.get("version", "").strip()
    if not latest:
        console.print("[red]PyPI returned no version field.[/]")
        return 1

    # Use the upload time of the latest release for the date stamp.
    releases = data.get("releases") or {}
    release_files = releases.get(latest) or []
    upload_time = release_files[0].get("upload_time_iso_8601") if release_files else None
    date_str = _format_release_date(upload_time)

    console.print(
        f"Latest:    [bold]{latest}[/]"
        + (f" — released {date_str}" if date_str else "")
    )

    # Compare. If we can't parse one of the versions, fall back to
    # string equality (which is correct often enough for a
    # first-pass heuristic).
    try:
        if _parse_version(installed) >= _parse_version(latest):
            console.print(
                "\n[green]You're on the latest version.[/] No upgrade needed."
            )
            return 0
    except Exception:
        if installed == latest:
            console.print(
                "\n[green]You're on the latest version.[/] No upgrade needed."
            )
            return 0

    # A newer version is available.
    console.print(
        f"\n[bold cyan]A newer version of Sieve is available.[/]\n"
    )

    install_kind = _detect_install_path()
    primary = "  [cyan]pipx upgrade llm-sieve[/]"
    secondary = "  [cyan]pip install --upgrade llm-sieve[/]   (if pip into a venv)"
    if install_kind == "pip":
        primary, secondary = (
            "  [cyan]pip install --upgrade llm-sieve[/]",
            "  [cyan]pipx upgrade llm-sieve[/]                (if installed via pipx)",
        )

    console.print("To upgrade:")
    console.print(primary)
    console.print(secondary)
    console.print(
        "\n[dim]Tip: run [cyan]sieve backup create[/] before upgrading if "
        "you've accumulated significant data.[/]"
    )
    console.print(
        f"\nRelease notes: "
        f"https://github.com/llmsieve/llm-sieve/releases/tag/v{latest}"
    )
    return 0

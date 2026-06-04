"""Uninstall helpers for `sieve uninstall --soft/--hard`.

--soft leaves the data directory in place; the user has to run
`pip uninstall llm-sieve` themselves (we can't self-uninstall the
running interpreter reliably).

--hard deletes ``~/.sieve/`` recursively after an explicit
confirmation. The pip uninstall instruction is still printed afterwards.
"""

from __future__ import annotations

import shutil
from pathlib import Path


SIEVE_DIR = Path("~/.sieve").expanduser()


def wipe_sieve_dir(sieve_dir: Path | None = None) -> None:
    """Recursively remove the sieve data directory if it exists.

    Also runs ``remove_autostart_on_uninstall`` so any systemd user
    service we installed earlier gets torn down. Both steps are
    idempotent — calling wipe on a clean system is a no-op.
    """
    # Autostart teardown first: if ~/.sieve is gone but the systemd
    # unit still points at a now-deleted binary, users would see
    # obscure "exec failed" errors on next boot. Disable before we
    # remove data.
    try:
        from sieve._autostart import remove_autostart_on_uninstall
        remove_autostart_on_uninstall()
    except Exception:
        # Uninstall must never fail because systemd got grumpy.
        pass
    target = sieve_dir if sieve_dir is not None else SIEVE_DIR
    if target.exists():
        shutil.rmtree(target)


def pip_uninstall_hint() -> str:
    """Message explaining how to remove the installed package."""
    return (
        "To remove the Sieve package itself, run one of:\n"
        "  pipx uninstall llm-sieve     (if installed via pipx — recommended)\n"
        "  pip uninstall llm-sieve      (if installed via pip into a venv)\n"
    )

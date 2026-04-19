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
    """Recursively remove the sieve data directory if it exists."""
    target = sieve_dir if sieve_dir is not None else SIEVE_DIR
    if target.exists():
        shutil.rmtree(target)


def pip_uninstall_hint() -> str:
    """Message explaining how to remove the pip-installed package."""
    return (
        "To remove the Sieve package itself, run:\n"
        "  pip uninstall llm-sieve\n"
    )

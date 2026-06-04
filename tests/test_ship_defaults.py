"""Ship-hygiene audit C#8/C#9/C#10 — safe shipping defaults."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from sieve.config import ListenConfig, _build_config

SIEVE_SRC = Path(__file__).resolve().parents[1] / "sieve"


def test_c8_listen_host_dataclass_default_is_loopback():
    """Fresh-install safety: default to 127.0.0.1, not 0.0.0.0."""
    assert ListenConfig().host == "127.0.0.1"


def test_c8_listen_host_loader_default_is_loopback():
    """YAML without listen.host set should also yield 127.0.0.1."""
    c = _build_config({})
    assert c.listen.host == "127.0.0.1"


def test_c8_listen_host_yaml_override_to_0000_still_works():
    """Back-compat: user explicitly asking for 0.0.0.0 in YAML is honoured."""
    c = _build_config({"listen": {"host": "0.0.0.0"}})
    assert c.listen.host == "0.0.0.0"


def test_c10_cli_demo_fixture_no_bristol():
    """Demo fixture in sieve/cli.py must not ship 'Bristol' inside a quoted
    string (locale-specific to the benchmark persona)."""
    cli_text = (SIEVE_SRC / "cli.py").read_text()
    # Flag any quoted string (single or double) containing Bristol (case-insensitive).
    quoted_bristol = re.search(
        r"""['"][^'"]*\bBristol\b[^'"]*['"]""",
        cli_text,
        re.IGNORECASE,
    )
    assert not quoted_bristol, (
        f"cli.py still ships 'Bristol' in a quoted string: "
        f"{quoted_bristol.group(0)!r}"
    )


def test_c10_cli_demo_fixture_no_mabel():
    """Mabel was a benchmark-persona name; it should not appear in
    shipping cli.py outside a comment."""
    cli_text = (SIEVE_SRC / "cli.py").read_text()
    for lineno, line in enumerate(cli_text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"\bMabel\b", line):
            pytest.fail(
                f"cli.py:{lineno} still references 'Mabel': {stripped}"
            )

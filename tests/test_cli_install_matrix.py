"""Install-test matrix — verify the CLI surface is complete and every
command has working --help.

These tests catch regressions where a new click.command decorator is
forgotten, a subcommand is dropped, or help text fails to render.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner


EXPECTED_TOP_LEVEL = {
    "init",
    "start",
    "stop",
    "restart",
    "status",
    "demo",
    "uninstall",
    "store",       # group
    "config",      # group
    "key",         # group
    "backup",      # group
}

EXPECTED_STORE = {
    "init", "status", "migrate",
    "facts", "entities", "relationships", "episodes",
    "stats", "export", "wipe",
}

EXPECTED_CONFIG = {"show", "set", "reset", "edit"}
EXPECTED_KEY = {"show", "rotate", "export", "import"}
EXPECTED_BACKUP = {"create", "list", "restore"}


def test_top_level_help_lists_all_commands():
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in EXPECTED_TOP_LEVEL:
        assert cmd in result.output, f"'{cmd}' missing from top-level --help"


@pytest.mark.parametrize("cmd", sorted(EXPECTED_TOP_LEVEL))
def test_every_top_level_command_has_help(cmd):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, [cmd, "--help"])
    assert result.exit_code == 0, (
        f"'sieve {cmd} --help' failed: {result.output}"
    )
    assert "Usage:" in result.output


@pytest.mark.parametrize(
    "group,expected",
    [
        ("store", EXPECTED_STORE),
        ("config", EXPECTED_CONFIG),
        ("key", EXPECTED_KEY),
        ("backup", EXPECTED_BACKUP),
    ],
)
def test_group_surfaces_all_subcommands(group, expected):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, [group, "--help"])
    assert result.exit_code == 0, result.output
    for sub in expected:
        assert sub in result.output, (
            f"'{sub}' missing from 'sieve {group} --help'"
        )


@pytest.mark.parametrize(
    "args",
    [
        ["config", "set"],                 # missing both args
        ["config", "set", "a"],            # missing value
        ["store", "export"],               # missing --output
        ["key", "import"],                 # missing keyfile argument
        ["backup", "restore"],             # missing backup id
    ],
)
def test_missing_args_exit_nonzero_with_usage(args):
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, args)
    assert result.exit_code != 0
    assert (
        "Usage:" in result.output
        or "Error:" in result.output
        or "Missing" in result.output
    )


def test_version_flag_works():
    from sieve import cli as cli_mod
    runner = CliRunner()
    result = runner.invoke(cli_mod.cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert "sieve" in result.output.lower()

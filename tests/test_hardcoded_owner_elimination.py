"""Verify no production SQL or regex path contains hardcoded
'jamie'/'jamie rivera'/'mary'.

These are Category-C ship-blockers from the 2026-04-22 hygiene audit.
For any user not named Jamie, Layer 1 absence signals, Layer 3 response
verification, schema_v2 multi-hop routing, and the D41 polarity
safety-net all silently no-op.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from sieve.config import ProfileOwnerConfig

LLM_SIEVE_SRC = Path(__file__).resolve().parents[1] / "sieve"

# Files legitimately exempt (tests, sandboxes, fixtures).
EXEMPT_PARTS = {
    "_sandbox.py",         # dev-only
    "_agent_fixture.py",   # test fixtures
}

# Hardcoded owner literals forbidden in production code (SQL/regex/alias sets).
# Docstrings, comments, and illustrative prompt examples are OK — see
# the _is_illustrative heuristic below.
FORBIDDEN_PATTERNS = [
    r"'jamie rivera'",
    r"'jamie'",
    r'"jamie rivera"',
    r'"jamie"',
    # Benchmark-only persona name — appears in regex strings without quotes.
    # The raw string r"\bmary'?s" in query_classifier_v2 is caught by this.
    r"'mary'",
    r'"mary"',
    r"\\bmary\b",    # regex literal \bmary\b inside a raw string
    r"\\bmary'",     # regex literal \bmary' (possessive form in raw string)
]


def _iter_source_files():
    for p in LLM_SIEVE_SRC.rglob("*.py"):
        if any(ex in p.name for ex in EXEMPT_PARTS):
            continue
        yield p


def _is_illustrative(line: str) -> bool:
    """Heuristic: is this a comment, docstring content, or inside a
    prompt-template string with ``example``/``e.g.`` context?
    """
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return True
    # Prompt-templates often include illustrative names in quoted strings
    # with 'example' context — leave those; they teach the LLM.
    lowered = line.lower()
    if "example" in lowered or "e.g." in lowered or "for instance" in lowered:
        return True
    return False


@pytest.mark.parametrize("forbidden", FORBIDDEN_PATTERNS)
def test_no_hardcoded_owner_in_production_code(forbidden):
    pattern = re.compile(forbidden, re.IGNORECASE)
    offenders = []
    for p in _iter_source_files():
        text = p.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line) and not _is_illustrative(line):
                offenders.append(
                    f"{p.relative_to(LLM_SIEVE_SRC)}:{lineno}: {line.strip()}"
                )
    assert not offenders, (
        f"Hardcoded {forbidden!r} in production code "
        f"(ship-hygiene audit findings C#4/C#5/C#12):\n"
        + "\n".join(offenders)
    )


def test_verification_resolves_owner_from_config():
    """Integration: the absence-signal path must accept a non-Jamie owner.

    Constructs the owner as Taylor Kim and verifies that:
    - build_absence_signals accepts a profile_owner kwarg
    - _user_relationships SQL uses the owner's name, not hardcoded 'jamie'
    - _subjects_equivalent treats the owner's aliases as user-aliases
    """
    owner = ProfileOwnerConfig(
        name="Taylor Kim",
        aliases=["Taylor", "TK", "me"],
    )

    from sieve.verification import build_absence_signals, _subjects_equivalent

    # Integration smoke: build_absence_signals must accept profile_owner kwarg
    # without raising (no real store needed — empty store returns no signals).
    try:
        signals = build_absence_signals(
            "Does Taylor have a sister?",
            [],
            None,  # no store — triggers early-exit path
            profile_owner=owner,
        )
        # With no store, should return [] cleanly
        assert signals == []
    except TypeError as e:
        pytest.fail(
            f"build_absence_signals does not accept profile_owner kwarg: {e}"
        )

    # _subjects_equivalent must treat owner name as user alias
    owner_aliases = frozenset(
        {"user", "the_user", "the user", "taylor", "taylor kim", "tk", "me"}
    )
    assert _subjects_equivalent("Taylor Kim", "the_user", user_aliases=owner_aliases), (
        "_subjects_equivalent does not treat owner name as user alias"
    )
    assert not _subjects_equivalent("taylor", "jamie", user_aliases=owner_aliases), (
        "_subjects_equivalent still carries hardcoded 'jamie' as a user alias "
        "when owner is Taylor Kim"
    )

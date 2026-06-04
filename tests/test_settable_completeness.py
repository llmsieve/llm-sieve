"""Regression test: every dataclass field in RecallConfig is reachable
via the YAML surface — either in _SETTABLE (CLI-settable) or in
PRODUCTION_YAML_ONLY_KEYS (YAML-only).

Without this check, adding a new dataclass field without updating
_SETTABLE or PRODUCTION_YAML_ONLY_KEYS would silently re-introduce
ProductionKeyViolation bugs for any user YAML that uses those fields.
"""
from __future__ import annotations

import dataclasses

import pytest

from sieve.config import RecallConfig
from sieve.config_surfaces import PRODUCTION_KEYS, PRODUCTION_YAML_ONLY_KEYS
from sieve.cli_config import _SETTABLE


def _walk_instance(obj, prefix=""):
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        name = f"{prefix}{f.name}"
        if dataclasses.is_dataclass(v):
            yield from _walk_instance(v, name + ".")
        else:
            yield name


# Dataclass field names that deliberately differ from their YAML/CLI key.
# The CLI surface uses the YAML key (left side); the dataclass uses the
# right side. Both are intentional and correct — do not remove.
_YAML_TO_DATACLASS_ALIASES = {
    # YAML key                        : dataclass field name
    "retrieval.temporal_dedup":         "retrieval.temporal_dedup_enabled",
}
_ALIASED_DATACLASS_FIELDS: frozenset[str] = frozenset(
    _YAML_TO_DATACLASS_ALIASES.values()
)

# Dataclass fields that hold container values (list / dict) and are
# therefore walked as a single terminal node whose name does not appear
# in PRODUCTION_KEYS directly. Their YAML paths ARE covered by
# PRODUCTION_YAML_ONLY_KEYS (e.g. "profile_owner.aliases") but the
# walk_instance traversal reaches them under a different path (e.g.
# "profile_owner.aliases" itself, which IS in PRODUCTION_YAML_ONLY_KEYS).
# We list them here only as documentation; they are NOT skipped from the
# completeness check — they will match via PRODUCTION_YAML_ONLY_KEYS.
_CONTAINER_FIELDS = {
    "provider.options",          # dict — covered by PRODUCTION_YAML_ONLY_KEYS
    "profile_owner.aliases",     # list[str] — covered by PRODUCTION_YAML_ONLY_KEYS
    "security.allowed_origins",  # list[str] — covered by PRODUCTION_YAML_ONLY_KEYS
}


def test_every_dataclass_path_is_in_production_surface():
    """Every dotted-path reachable from RecallConfig() defaults must
    exist in PRODUCTION_KEYS (or it will fail production-mode load).

    Exceptions:
    - Dataclass fields that are covered via a YAML alias key
      (e.g. temporal_dedup_enabled is accessible via temporal_dedup).
    - Container fields (list/dict) that appear directly in
      PRODUCTION_YAML_ONLY_KEYS under their own dotted name.
    """
    cfg = RecallConfig()
    all_paths = set(_walk_instance(cfg))

    # Paths covered by an alias key in _SETTABLE are exempt from the direct
    # membership check — their YAML key IS in PRODUCTION_KEYS.
    paths_to_check = all_paths - _ALIASED_DATACLASS_FIELDS

    missing = sorted(paths_to_check - PRODUCTION_KEYS)
    assert not missing, (
        "Dataclass fields not in PRODUCTION_KEYS — add them to "
        "_SETTABLE (if CLI-settable) or to PRODUCTION_YAML_ONLY_KEYS "
        "(if YAML-only):\n  "
        + "\n  ".join(missing)
    )


def test_no_overlap_between_cli_settable_and_yaml_only():
    """A key must not appear in both _SETTABLE and PRODUCTION_YAML_ONLY_KEYS
    — that would indicate a design inconsistency."""
    overlap = frozenset(_SETTABLE.keys()) & PRODUCTION_YAML_ONLY_KEYS
    assert not overlap, (
        f"Keys appear in both _SETTABLE and PRODUCTION_YAML_ONLY_KEYS: "
        f"{sorted(overlap)}"
    )


def test_production_keys_equals_cli_plus_yaml_only():
    """PRODUCTION_KEYS must be exactly the union of _SETTABLE and
    PRODUCTION_YAML_ONLY_KEYS — no extra entries, no missing ones."""
    expected = frozenset(_SETTABLE.keys()) | PRODUCTION_YAML_ONLY_KEYS
    assert PRODUCTION_KEYS == expected, (
        f"PRODUCTION_KEYS diverges from expected union.\n"
        f"Extra in PRODUCTION_KEYS: {sorted(PRODUCTION_KEYS - expected)}\n"
        f"Missing from PRODUCTION_KEYS: {sorted(expected - PRODUCTION_KEYS)}"
    )

"""Sieve production vs test config surfaces.

Production keys ship to users via `sieve config set` and sieve.yaml.
Test keys require SIEVE_MODE=test to be honoured; if found in sieve.yaml
while in production mode, startup fails.

Source of truth:
- Production keys: sieve.cli_config._SETTABLE
- Advanced keys: ADVANCED_KEYS below
- Any YAML key not in either set is rejected in either mode.

Adding a key to ADVANCED_KEYS is the ONLY way to expose new dials
without the production-mode startup assertion blocking them.
Graduating a key from advanced -> production is a code change:
add to _SETTABLE, document in sieve.example.yaml, update this module.
"""
from __future__ import annotations

from sieve.cli_config import _SETTABLE

# Every key that ships to the user-visible CLI surface.
_CLI_SETTABLE_KEYS: frozenset[str] = frozenset(_SETTABLE.keys())

# Keys that are valid in a production sieve.yaml but cannot be set via
# `sieve config set` because they hold list or nested-dict values that
# the CLI _coerce function cannot parse from a single string argument.
# Accepted from YAML; rejected when someone tries to set them via CLI.
PRODUCTION_YAML_ONLY_KEYS: frozenset[str] = frozenset({
    "profile_owner.aliases",      # list[str] — populated by YAML or wizard
    "security.allowed_origins",   # list[str] — CORS allowlist
    "provider.options",           # dict — raw Ollama options block
    "provider.options.think",     # nested dict leaf — think-mode flag
})

# Union: all keys accepted in production-mode YAML.
PRODUCTION_KEYS: frozenset[str] = _CLI_SETTABLE_KEYS | PRODUCTION_YAML_ONLY_KEYS

# Hidden-from-users but override-able when SIEVE_MODE=test.
# Source: ship-hygiene audit (SHIP_HYGIENE_AUDIT_2026_04_22.md) Category A
# findings — dials deemed "should be user-tunable" but not yet exposed in
# the production surface. Users experimenting can set these via a
# ~/.sieve/sieve.test.yaml alongside SIEVE_MODE=test.
ADVANCED_KEYS: frozenset[str] = frozenset({
    # pipeline — budget scaling
    "pipeline.budget_scale_threshold",
    "pipeline.budget_floor",
    "pipeline.budget_ceiling",
    "pipeline.extreme_threshold_tokens",
    "pipeline.extreme_target_tokens",
    # retrieval — knobs
    "retrieval.max_facts",
    "retrieval.context_tokens",
    "retrieval.mmr_lambda",
    "retrieval.mmr_enabled",
    "retrieval.confidence_floor",
    "retrieval.dedup_similarity",
    "retrieval.temporal_dedup_similarity",
    "retrieval.graph_reserve",
    "retrieval.decompose_max_subqueries",
    "retrieval.decompose_min_subqueries",
    "retrieval.decompose_max_tokens",
    "retrieval.absence_coverage_gate",
    "retrieval.absence_coverage_denominator",
    # writer — knobs
    "writer.conflict_similarity",
    "writer.dedup_distance",
    "writer.s2_timeout_s",
    "writer.episode_max_tokens",
    "writer.episode_timeout_s",
    "writer.session_coherence_window",
    "writer.max_sessions",
    # classifier / tools
    "classifier.min_retrieval_length",
    "classifier.l1_threshold",
    "tools.fallback_min_words",
    "tools.recall_max_rounds",
    # provider
    "provider.read_timeout_s",
})

# Sanity: no key is both production and advanced (would indicate drift
# between _SETTABLE and this file). If this fails at import time, fix
# ADVANCED_KEYS.
_overlap = PRODUCTION_KEYS & ADVANCED_KEYS
assert not _overlap, (
    f"keys in both production and advanced surfaces: {_overlap}; "
    f"remove duplicates from ADVANCED_KEYS in sieve/config_surfaces.py"
)


def is_production_key(dotted_path: str) -> bool:
    """True if the key is part of the shipping / `sieve config set` surface."""
    return dotted_path in PRODUCTION_KEYS


def is_advanced_key(dotted_path: str) -> bool:
    """True if the key is hidden but overridable under SIEVE_MODE=test."""
    return dotted_path in ADVANCED_KEYS


def flatten_yaml(raw: dict, prefix: str = "") -> list[str]:
    """Return every dotted key present in a nested YAML dict.

    Example:
        >>> flatten_yaml({"writer": {"num_ctx": 8192}, "listen": {"host": "x"}})
        ['writer.num_ctx', 'listen.host']

    Leaves (non-dict values) become terminal keys; intermediate dicts
    recurse. Empty dicts contribute no keys.
    """
    out: list[str] = []
    for k, v in raw.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(flatten_yaml(v, path))
        else:
            out.append(path)
    return out

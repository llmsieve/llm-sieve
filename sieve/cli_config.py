"""Runtime config management for `sieve config show/set/reset/edit`.

Keeps all YAML-walking logic in pure functions so it can be tested
without the CLI runner. The CLI layer in cli.py wraps these.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from sieve.config import RecallConfig


# Module-level handle so tests can monkeypatch. Callers resolve this
# lazily through `current_config_path()`.
SIEVE_CFG = Path("~/.sieve/sieve.yaml").expanduser()


def current_config_path() -> Path:
    return SIEVE_CFG


# ── Field-type registry for `set_path` validation/coercion ────────────────

# Every settable path → (python_type, optional enum-set). Paths not in
# the table cannot be set via `config set` (the whitelist prevents users
# from writing random keys that the dataclass loader would silently
# ignore). Add a row here when exposing a new option.
_BOOL_TRUE = {"1", "true", "yes", "on", "y", "t"}
_BOOL_FALSE = {"0", "false", "no", "off", "n", "f"}

_SETTABLE: dict[str, tuple[type, set | None]] = {
    "listen.host": (str, None),
    "listen.port": (int, None),

    "provider.type": (str, None),
    "provider.base_url": (str, None),
    "provider.api_key": (str, None),
    "provider.default_model": (str, None),

    "embeddings.provider": (str, {"fastembed", "ollama"}),
    "embeddings.ollama_url": (str, None),
    "embeddings.ollama_model": (str, None),

    "store.path": (str, None),
    "store.embedding_model": (str, None),
    "store.embedding_dimensions": (int, None),

    "pipeline.conversation_turns": (int, None),
    "pipeline.max_rounds": (int, None),
    "pipeline.core_facts_size": (int, None),
    "pipeline.max_outbound_tokens": (int, None),
    "pipeline.context_format": (str, {"flat", "structured", "auto"}),
    "pipeline.pre_populate_top_k": (int, None),

    "writer.model": (str, None),
    "writer.fallback_model": (str, None),
    "writer.num_ctx": (int, None),
    "writer.ghost_validator_enabled": (bool, None),

    "retrieval.temporal_dedup": (bool, None),
    "retrieval.reranker_enabled": (bool, None),
    "retrieval.query_decomposition_enabled": (bool, None),

    "learning.tune_interval": (int, None),
    "learning.relevance_threshold": (float, None),
    "learning.core_facts_size": (int, None),

    "security.auth_token": (str, None),

    "tools.enabled": (bool, None),
    "tools.compression": (str, {"none", "moderate", "aggressive"}),
    "tools.l1_threshold": (float, None),
    "tools.fallback_include_all": (bool, None),
    "tools.max_tools_injected": (int, None),

    "ablation.absence_signal": (bool, None),
    "ablation.closed_world": (bool, None),
    "ablation.response_verification": (bool, None),
    "ablation.schema_v2": (bool, None),
    "ablation.tier2_classifier": (bool, None),
    "ablation.extreme_summary": (bool, None),

    "profile_owner.name": (str, None),
    "profile_owner.pin": (str, None),

    "mode_override": (str, {"auto", "A", "B"}),
}


def _coerce(value: Any, py_type: type) -> Any:
    if py_type is bool:
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
        raise ValueError(f"cannot interpret {value!r} as boolean")
    if py_type is int:
        return int(value)
    if py_type is float:
        return float(value)
    return str(value)


def set_path(data: dict, path: str, value: Any) -> None:
    """Set ``data[path]`` in place with validation and coercion."""
    if path not in _SETTABLE:
        raise ValueError(f"unknown config path {path!r}")
    py_type, enum = _SETTABLE[path]
    coerced = _coerce(value, py_type)
    if enum is not None and coerced not in enum:
        raise ValueError(
            f"invalid value for {path}: {coerced!r} not in {sorted(enum)}"
        )

    # Walk / create nested dicts.
    parts = path.split(".")
    cursor = data
    for p in parts[:-1]:
        if p not in cursor or not isinstance(cursor[p], dict):
            cursor[p] = {}
        cursor = cursor[p]
    cursor[parts[-1]] = coerced


def load_raw(path: Path | None = None) -> dict:
    p = path or current_config_path()
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def write_raw(data: dict, path: Path | None = None) -> None:
    p = path or current_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False))


def validate_raw(data: dict) -> None:
    """Raise if the YAML can't be loaded as a valid RecallConfig."""
    # Re-parse through the dataclass loader — any malformed value
    # triggers either an exception or a logger warning we ignore here.
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(data, f)
        tmp = f.name
    try:
        RecallConfig.load(tmp)
    finally:
        os.unlink(tmp)


# ── Diffing for `config show` ──────────────────────────────────────────────

def _walk_dataclass(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if is_dataclass(obj):
        for f in fields(obj):
            _walk_dataclass(
                f"{prefix}.{f.name}" if prefix else f.name,
                getattr(obj, f.name),
                out,
            )
    else:
        out[prefix] = obj


def diff_from_defaults(cfg: RecallConfig) -> list[tuple[str, Any, Any]]:
    """Return [(path, current, default), ...] for fields whose current
    value differs from the defaults."""
    current: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    _walk_dataclass("", cfg, current)
    _walk_dataclass("", RecallConfig(), defaults)
    diffs = []
    for k, cur in current.items():
        dflt = defaults.get(k)
        if cur != dflt:
            diffs.append((k, cur, dflt))
    return diffs

"""SIEVE_MODE detection + mode-aware YAML loader.

Production mode (default, when SIEVE_MODE is unset or empty):
  - Loads sieve.yaml only.
  - Any key in sieve.yaml must be in PRODUCTION_KEYS, else raise
    ProductionKeyViolation. Typos are caught loud; advanced dials are
    rejected with a hint pointing at SIEVE_MODE=test.

Test mode (SIEVE_MODE=test, case-insensitive):
  - Loads sieve.yaml first.
  - Merges sieve.test.yaml on top (test wins on collision).
  - Any key in either file must be in PRODUCTION_KEYS | ADVANCED_KEYS,
    else raise. Emits a warning listing every advanced key active.

Both modes return the merged raw dict; `sieve.config._build_config` turns
that into the typed RecallConfig.
"""
from __future__ import annotations

import enum
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from sieve.config_surfaces import (
    ADVANCED_KEYS,
    PRODUCTION_KEYS,
    flatten_yaml,
)

logger = logging.getLogger("recall.config_modes")


class Mode(enum.Enum):
    PRODUCTION = "production"
    TEST = "test"


class ProductionKeyViolation(Exception):
    """Raised when a YAML key is outside the allowed surface for the active mode."""
    pass


def current_mode() -> Mode:
    """Return the active mode, read once from the SIEVE_MODE env var.

    Production is the default. 'test' (case-insensitive) selects test
    mode. Any other value raises ValueError so misconfiguration is
    loud instead of silently downgraded to production.
    """
    raw = os.environ.get("SIEVE_MODE", "").strip().lower() or "production"
    try:
        return Mode(raw)
    except ValueError:
        raise ValueError(
            f"SIEVE_MODE={raw!r} is not valid. "
            f"Use 'production' (default) or 'test'."
        )


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into `base`; overlay wins on conflict.

    Only dict-vs-dict keys are recursed; any other type collision results
    in overlay replacing base.
    """
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path | None) -> dict:
    """Return the parsed YAML at `path` as a dict, or {} if the file is
    missing / the path is None / the YAML is empty.
    """
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        # Top-level YAML isn't a mapping — treat as empty.
        return {}
    return data


def load_config_for_mode(
    yaml_path: Path | None = None,
    test_yaml_path: Path | None = None,
    mode: Mode | None = None,
) -> dict:
    """Load and merge YAML per the active mode, enforcing the surface.

    Args:
        yaml_path: primary YAML (sieve.yaml). Optional; missing = empty.
        test_yaml_path: overlay YAML (sieve.test.yaml). Only read when
            mode is TEST. Optional; missing = base YAML only.
        mode: override the env-based mode. Omit in production use.

    Returns:
        A merged raw dict suitable for `sieve.config._build_config`.

    Raises:
        ProductionKeyViolation: if the merged config contains keys
            outside the surface allowed in the active mode.
    """
    mode = mode or current_mode()

    raw = _load_yaml(yaml_path)

    if mode is Mode.TEST:
        overlay = _load_yaml(test_yaml_path)
        if overlay:
            raw = _deep_merge(raw, overlay)

    # Enforce the surface.
    allowed = (
        PRODUCTION_KEYS | ADVANCED_KEYS if mode is Mode.TEST else PRODUCTION_KEYS
    )
    offenders = sorted(k for k in flatten_yaml(raw) if k not in allowed)
    if offenders:
        surface_name = mode.value
        hint = (
            " Set SIEVE_MODE=test (and put them in ~/.sieve/sieve.test.yaml) "
            "to override advanced dials."
            if mode is Mode.PRODUCTION
            else ""
        )
        raise ProductionKeyViolation(
            f"Keys not in {surface_name} surface: "
            + ", ".join(offenders)
            + "."
            + hint
        )

    # Warn on advanced keys active in test mode so the run isn't surprising.
    if mode is Mode.TEST:
        advanced_active = sorted(
            k for k in flatten_yaml(raw) if k in ADVANCED_KEYS
        )
        if advanced_active:
            logger.warning(
                "SIEVE_MODE=test: advanced overrides active: %s",
                ", ".join(advanced_active),
            )

    return raw

"""Tool schema compression — shrink verbose tool schemas before injection.

Three modes:
  - none        : passthrough (deep copy only, no changes)
  - moderate    : first-sentence description, drop param description/examples/default
  - aggressive  : drop all descriptions entirely, keep only name + types + required

Operates on both OpenAI shape (`{"type":"function","function":{...}}`) and
Ollama shape (`{"name":..., "description":..., "parameters":{...}}`).
Pure functions — never mutates input.
"""

from __future__ import annotations

import copy
from typing import Any

VALID_MODES = ("none", "moderate", "aggressive")


def compress_schema(schema: dict, mode: str = "moderate") -> dict:
    """Return a compressed copy of the tool schema per the given mode.

    Args:
        schema: either OpenAI-shape (top-level "function" key) or Ollama
                flat-shape ({"name":..., "description":..., "parameters":...}).
        mode:   "none" | "moderate" | "aggressive".

    Returns:
        A deep-copied schema with compression applied. The input is never
        modified.

    Raises:
        ValueError: if `mode` is not one of VALID_MODES.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown compression mode: {mode!r} (valid: {VALID_MODES})")

    out = copy.deepcopy(schema)
    if mode == "none":
        return out

    # Locate the "function body" — either nested under "function" or top-level
    if "function" in out and isinstance(out["function"], dict):
        body = out["function"]
    else:
        body = out

    if mode == "moderate":
        _apply_moderate(body)
    elif mode == "aggressive":
        _apply_aggressive(body)

    return out


def _first_sentence(text: str) -> str:
    """Return the first sentence of `text`. Simple rule: split on '. '."""
    if not text:
        return text
    # Split on the first ". " (period + space) to preserve abbreviations like "U.S."
    idx = text.find(". ")
    if idx == -1:
        # No period-space; try just a trailing period
        return text.rstrip()
    return text[: idx + 1]


def _apply_moderate(body: dict) -> None:
    """Mutate `body` in place: shorten description, strip param extras."""
    if isinstance(body.get("description"), str):
        body["description"] = _first_sentence(body["description"])

    params = body.get("parameters")
    if not isinstance(params, dict):
        return

    props = params.get("properties")
    if isinstance(props, dict):
        for name, prop in list(props.items()):
            if not isinstance(prop, dict):
                continue
            trimmed: dict[str, Any] = {}
            for key in ("type", "enum", "items"):
                if key in prop:
                    trimmed[key] = prop[key]
            props[name] = trimmed


def _apply_aggressive(body: dict) -> None:
    """Mutate `body` in place: drop all descriptions, keep types + required only."""
    body.pop("description", None)

    params = body.get("parameters")
    if not isinstance(params, dict):
        return

    props = params.get("properties")
    if isinstance(props, dict):
        for name, prop in list(props.items()):
            if not isinstance(prop, dict):
                continue
            trimmed: dict[str, Any] = {}
            for key in ("type", "items"):
                if key in prop:
                    trimmed[key] = prop[key]
            props[name] = trimmed

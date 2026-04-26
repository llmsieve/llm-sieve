"""Sieve test-mode module — internal protocol surface for evaluation.

Loaded ONLY when ``SIEVE_TEST_MODE=on`` env var is set. In production mode
this module is not imported and the ``/test/*`` endpoints are not mounted
(404 on any request).

This module exposes a stable versioned protocol used by an internal test
harness (separate from this repo) for benchmark and evaluation work:

- **Control plane:** REST endpoints under ``/test/control/*`` and ``/test/state``
- **Telemetry plane:** Server-Sent Events stream at ``/test/events``

CARDINAL RULE: this module is *protocol surface only*, not test logic.
No scenario YAMLs, no graders, no report generators here. The harness
that consumes this surface lives in a separate internal repo.
"""
from __future__ import annotations

import os

PROTOCOL_VERSION = 1
"""Protocol major version."""


def is_test_mode_enabled() -> bool:
    """Return True if SIEVE_TEST_MODE is set to a truthy value.

    Truthy values: 'on', '1', 'true', 'yes' (case-insensitive).
    Anything else (including unset) returns False.
    """
    val = os.environ.get("SIEVE_TEST_MODE", "").strip().lower()
    return val in {"on", "1", "true", "yes"}


__all__ = ["PROTOCOL_VERSION", "is_test_mode_enabled"]

"""Sieve test-mode module — opt-in protocol surface for external harnesses.

Loaded ONLY when ``SIEVE_TEST_MODE=on`` env var is set. In production mode
this module is not imported and the ``/test/*`` endpoints are not mounted
(404 on any request).

This module exposes a stable versioned protocol that an external test
harness can drive to script Sieve for benchmark and evaluation work:

- **Control plane:** REST endpoints under ``/test/control/*`` and ``/test/state``
- **Telemetry plane:** Server-Sent Events stream at ``/test/events``

This module is *protocol surface only*, not test logic. No scenario
runners, no graders, no report generators — those belong in the
harness that drives this protocol.
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

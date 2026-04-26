"""Sieve test-mode module — protocol surface for sieve-test research harness.

Loaded ONLY when ``SIEVE_TEST_MODE=on`` env var is set. In production mode
this module is not imported and the ``/test/*`` endpoints are not mounted
(404 on any request).

This module exposes a stable versioned protocol:

- **Control plane:** REST endpoints under ``/test/control/*`` and ``/test/state``
- **Telemetry plane:** Server-Sent Events stream at ``/test/events``

The schema is mirrored from the canonical sieve-test repo at
https://github.com/azardhosein/sieve-test (private). Pydantic models in
this module MUST match `sieve-test/sieve_test/protocol/{events,control}.py`.
Drift is detected by sieve-test on consumption (Pydantic validation
failure); the runtime keeps working.

CARDINAL RULE: this module is *protocol surface only*, not test logic.
No scenario YAMLs, no graders, no report generators here. Those live in
sieve-test exclusively.

Spec: see ``docs/test-mode-protocol-v1.md`` (mirrored from
``sieve-test/docs/specs/protocol-spec-v1.md``).
"""
from __future__ import annotations

import os

PROTOCOL_VERSION = 1
"""Protocol major version. Must match sieve-test's PROTOCOL_VERSION."""


def is_test_mode_enabled() -> bool:
    """Return True if SIEVE_TEST_MODE is set to a truthy value.

    Truthy values: 'on', '1', 'true', 'yes' (case-insensitive).
    Anything else (including unset) returns False.
    """
    val = os.environ.get("SIEVE_TEST_MODE", "").strip().lower()
    return val in {"on", "1", "true", "yes"}


__all__ = ["PROTOCOL_VERSION", "is_test_mode_enabled"]

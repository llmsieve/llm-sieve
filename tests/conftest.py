"""Session-wide pytest configuration for llm-sieve tests.

Isolates tests from the developer's real ~/.sieve/sieve.yaml so that
module-level imports (e.g. ``sieve.main``'s ``app = create_app()``) are
not contaminated by keys outside the production surface.

Without this guard, any test file that imports from ``sieve.main`` at
collection time would trigger ``ProductionKeyViolation`` if the real
sieve.yaml contains advanced-only or unknown keys.  The fix is to set
SIEVE_CONFIG to a non-existent path before collection so
``load_config_for_mode`` sees an empty YAML and falls back to defaults.
"""
from __future__ import annotations

import os

# Before any test module is imported, ensure SIEVE_CONFIG points nowhere.
# Tests that need a specific YAML file monkeypatch SIEVE_CONFIG themselves.
# This sentinel path will not exist, so RecallConfig.load() will use
# ship-safe defaults for the module-level create_app() call in sieve.main.
if "SIEVE_CONFIG" not in os.environ:
    # Point at a path that will never exist; load_config_for_mode treats a
    # missing file as an empty YAML and falls back to ship-safe defaults.
    os.environ["SIEVE_CONFIG"] = "/tmp/.sieve-test-sentinel-nonexistent.yaml"

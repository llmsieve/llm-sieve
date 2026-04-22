"""Tests for the dynamic max_outbound_tokens budget scaling (Fix 1)."""
from __future__ import annotations

import pytest

from sieve.config import PipelineConfig, _build_config


def test_max_outbound_tokens_default_is_8000():
    """Back-compat: unset means 8000 (legacy behaviour)."""
    c = PipelineConfig()
    assert c.max_outbound_tokens == 8000


def test_resolve_budget_uses_explicit_override():
    """When max_outbound_tokens is set to non-default, it wins."""
    c = PipelineConfig(max_outbound_tokens=12000)
    assert c.resolve_budget(upstream_ctx=40960) == 12000


def test_resolve_budget_scales_with_upstream_ctx_when_default():
    """When max_outbound_tokens is at default (8000) AND upstream_ctx > 16384,
    budget scales to min(upstream_ctx // 2, 32768)."""
    c = PipelineConfig()  # default 8000
    # qwen3:14b = 40960 → 20480
    assert c.resolve_budget(upstream_ctx=40960) == 20480
    # qwen3.6:35b = 131072 → 32768 (capped)
    assert c.resolve_budget(upstream_ctx=131072) == 32768
    # small model = 8192 → stays at 8000 (unchanged; below scale threshold)
    assert c.resolve_budget(upstream_ctx=8192) == 8000


def test_resolve_budget_explicit_honored_exactly():
    """Explicit (non-default) max_outbound_tokens values are honoured as-is,
    including values below the _BUDGET_FLOOR constant. The floor only guards
    the auto-scaling path, not intentional programmatic/YAML overrides."""
    # Tiny explicit override is returned unchanged (not floored).
    c = PipelineConfig(max_outbound_tokens=1000)
    assert c.resolve_budget(upstream_ctx=2048) == 1000
    assert c.resolve_budget(upstream_ctx=40960) == 1000
    # Default (8000) with tiny ctx stays at 8000 — already above _BUDGET_FLOOR.
    c_default = PipelineConfig()
    assert c_default.resolve_budget(upstream_ctx=2048) == 8000


def test_build_config_reads_yaml_override():
    """Explicit YAML value overrides the scaling logic."""
    c = _build_config({"pipeline": {"max_outbound_tokens": 5000}})
    assert c.pipeline.max_outbound_tokens == 5000
    assert c.pipeline.resolve_budget(upstream_ctx=40960) == 5000

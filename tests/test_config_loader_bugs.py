"""Regression tests for the silent-config-drop bugs found in the 2026-04-22
ship-hygiene audit. Each YAML override below was written by
`sieve config set` but never read back by `_build_config`.
"""
from __future__ import annotations

import pytest

from sieve.config import _build_config


def test_c1_writer_num_ctx_yaml_override_honoured():
    """C#1: writer.num_ctx must be readable from YAML, not dropped."""
    raw = {"writer": {"num_ctx": 8192}}
    c = _build_config(raw)
    assert c.writer.num_ctx == 8192, \
        f"writer.num_ctx YAML override dropped; got {c.writer.num_ctx}"


def test_c1_writer_num_ctx_default_preserved_when_unset():
    """Back-compat: when YAML omits writer.num_ctx, the dataclass default (4096) applies."""
    c = _build_config({})
    assert c.writer.num_ctx == 4096


def test_c2_pipeline_upstream_ctx_default_yaml_override_honoured():
    """C#2: pipeline.upstream_ctx_default must be readable from YAML."""
    raw = {"pipeline": {"upstream_ctx_default": 32768}}
    c = _build_config(raw)
    assert c.pipeline.upstream_ctx_default == 32768, \
        f"pipeline.upstream_ctx_default YAML override dropped; got {c.pipeline.upstream_ctx_default}"


def test_c2_pipeline_upstream_ctx_default_unset_preserves_default():
    """Back-compat: 8192 default when unset."""
    c = _build_config({})
    assert c.pipeline.upstream_ctx_default == 8192


def test_c2_upstream_ctx_default_feeds_resolve_budget():
    """Integration: when YAML raises upstream_ctx_default, resolve_budget activates scaling.

    Without the loader fix, even if the YAML value arrived, resolve_budget
    would still see default 8192 (which is below the 16384 scale threshold)
    and refuse to scale. With the fix, 32768 > 16384, so budget scales to
    min(32768 // 2, 32768) = 16384.
    """
    raw = {"pipeline": {"upstream_ctx_default": 32768}}
    c = _build_config(raw)
    assert c.pipeline.resolve_budget(upstream_ctx=c.pipeline.upstream_ctx_default) == 16384

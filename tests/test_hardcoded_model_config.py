"""Verify extreme_summary and tier2_classifier models are config-routed,
not hardcoded. Ship-hygiene audit findings C#3 and C#7.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import pytest

from sieve.config import _build_config


def test_c7_ablation_tier2_classifier_model_defaults_to_auto():
    """The new dial must exist on AblationConfig with default 'auto'."""
    c = _build_config({})
    assert c.ablation.tier2_classifier_model == "auto"


def test_c7_ablation_tier2_classifier_model_yaml_override():
    """YAML override must flow through _build_config."""
    c = _build_config({"ablation": {"tier2_classifier_model": "gemma4:e4b"}})
    assert c.ablation.tier2_classifier_model == "gemma4:e4b"


def test_c7_ablation_tier2_classifier_model_in_settable():
    """The dial must be in _SETTABLE so `sieve config set` works."""
    from sieve.cli_config import _SETTABLE
    assert "ablation.tier2_classifier_model" in _SETTABLE


def test_c3_extreme_summary_model_kwarg_is_required():
    """sieve/extreme_summary.py should no longer have a hardcoded default
    for `model`. Calling without model must raise TypeError."""
    from sieve import extreme_summary
    sig = inspect.signature(extreme_summary.summarise_async)
    model_param = sig.parameters.get("model")
    assert model_param is not None, "summarise_async must take a model arg"
    assert model_param.default is inspect.Parameter.empty, (
        f"summarise_async.model still has a hardcoded default "
        f"({model_param.default!r}); C#3 not fixed"
    )


def test_c7_classify_fact_async_model_kwarg_is_required():
    """sieve/fact_classifier_v2.py should no longer hardcode a model default."""
    from sieve import fact_classifier_v2
    # Find the callable that takes a model arg. The audit named it
    # classify_fact_async; be tolerant of a rename.
    target = None
    for name in ("classify_fact_async", "classify_fact", "classify_async"):
        candidate = getattr(fact_classifier_v2, name, None)
        if candidate is not None:
            target = candidate
            break
    assert target is not None, \
        "No classify_fact_async (or alias) in fact_classifier_v2"
    sig = inspect.signature(target)
    model_param = sig.parameters.get("model")
    assert model_param is not None, f"{target.__name__} must take a model arg"
    assert model_param.default is inspect.Parameter.empty, (
        f"{target.__name__}.model still has a hardcoded default "
        f"({model_param.default!r}); C#7 not fixed"
    )


def test_c3_no_hardcoded_qwen35_4b_in_extreme_summary():
    """Sanity grep: the literal 'qwen3.5:4b' must not appear in the
    extreme_summary module (it was the hardcoded default)."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "sieve" / "extreme_summary.py"
    text = src.read_text()
    # Allow the string in a comment explaining the change history, not in code.
    for lineno, line in enumerate(text.splitlines(), 1):
        if "qwen3.5:4b" in line.lower():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            pytest.fail(
                f"{src.name}:{lineno}: 'qwen3.5:4b' still present in "
                f"non-comment code: {stripped}"
            )


def test_c7_no_hardcoded_gemma4_e4b_in_fact_classifier():
    """Sanity grep: 'gemma4:e4b' must not appear in non-comment lines."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "sieve" / "fact_classifier_v2.py"
    text = src.read_text()
    for lineno, line in enumerate(text.splitlines(), 1):
        if "gemma4:e4b" in line.lower():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            pytest.fail(
                f"{src.name}:{lineno}: 'gemma4:e4b' still present in "
                f"non-comment code: {stripped}"
            )

"""S5 — guard against drift between sieve.example.yaml and the
RecallConfig dataclass.

If a key is added to the YAML without a corresponding dataclass field,
the loader silently drops it; if a dataclass field is renamed without
updating the example, users who follow the docs hit confusing errors.

This test loads the shipped example end-to-end and verifies:
  1. The YAML is structurally valid
  2. RecallConfig.load() accepts every section without raising
  3. Every top-level YAML key has a matching attribute on the resulting
     config object (catches removed-field drift)
  4. A few sentinel keys (writer.skip_empty_turns, embeddings.provider,
     progression.phase_1_threshold) round-trip cleanly — caught the
     temporal_dedup rename in this exact way (see commit 3597c34).

Runs in <50ms; no LLM, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sieve.config import RecallConfig


EXAMPLE_YAML = Path(__file__).parent.parent / "sieve.example.yaml"


@pytest.fixture
def raw_yaml() -> dict:
    """Parsed example.yaml as a raw dict — independent of the dataclass."""
    return yaml.safe_load(EXAMPLE_YAML.read_text())


@pytest.fixture
def loaded_config() -> RecallConfig:
    """Example YAML loaded through the real config parser."""
    return RecallConfig.load(EXAMPLE_YAML)


def test_example_yaml_is_parseable(raw_yaml: dict):
    """The shipped example YAML must be valid YAML."""
    assert isinstance(raw_yaml, dict)
    assert raw_yaml, "example.yaml is empty?"


def test_example_yaml_loads_through_dataclass(loaded_config: RecallConfig):
    """RecallConfig.load() must accept the shipped example without
    raising — this is the smoke test that catches most drift."""
    assert loaded_config is not None
    assert isinstance(loaded_config, RecallConfig)


def test_top_level_keys_have_matching_config_attrs(
    raw_yaml: dict, loaded_config: RecallConfig
):
    """Every top-level YAML key should map to a RecallConfig attribute.

    Catches: someone removes / renames a dataclass field but forgets
    to update the example, so the YAML key is silently dropped.
    """
    yaml_top_keys = set(raw_yaml.keys())
    config_attrs = {f for f in dir(loaded_config) if not f.startswith("_")}
    missing = yaml_top_keys - config_attrs
    assert not missing, (
        f"sieve.example.yaml has top-level keys with no matching "
        f"RecallConfig attribute: {missing}. Either add the dataclass "
        f"field or remove the key from the example."
    )


def test_writer_skip_empty_turns_round_trips(loaded_config: RecallConfig):
    """Sentinel: writer.skip_empty_turns is a v1-rc addition. Confirm
    the example loads it correctly (catches typos in the YAML key)."""
    assert isinstance(loaded_config.writer.skip_empty_turns, bool)


def test_embeddings_provider_round_trips(loaded_config: RecallConfig):
    """Sentinel: the FastEmbed default. If this fails the
    embeddings provider key was renamed without updating the example."""
    assert loaded_config.embeddings.provider in ("fastembed", "ollama")


def test_progression_phase_1_threshold_round_trips(loaded_config: RecallConfig):
    """Sentinel: progressive activation thresholds — caught real drift in
    the past."""
    assert isinstance(loaded_config.progression.phase_1_threshold, int)
    assert loaded_config.progression.phase_1_threshold > 0

import yaml
import pytest

from sieve.config import RecallConfig, ToolsConfig


def test_tools_config_defaults():
    cfg = RecallConfig()
    assert isinstance(cfg.tools, ToolsConfig)
    assert cfg.tools.enabled is True
    assert cfg.tools.compression == "moderate"
    assert cfg.tools.l1_threshold == 0.5
    assert cfg.tools.fallback_include_all is True
    assert cfg.tools.max_tools_injected == 10


def test_tools_config_loads_from_yaml(tmp_path):
    path = tmp_path / "sieve.yaml"
    path.write_text(yaml.safe_dump({
        "tools": {
            "enabled": False,
            "compression": "aggressive",
            "l1_threshold": 0.75,
            "fallback_include_all": False,
            "max_tools_injected": 3,
        }
    }))
    cfg = RecallConfig.load(path)
    assert cfg.tools.enabled is False
    assert cfg.tools.compression == "aggressive"
    assert cfg.tools.l1_threshold == 0.75
    assert cfg.tools.fallback_include_all is False
    assert cfg.tools.max_tools_injected == 3


def test_tools_config_rejects_bad_compression_mode(tmp_path):
    path = tmp_path / "sieve.yaml"
    path.write_text(yaml.safe_dump({"tools": {"compression": "ludicrous"}}))
    cfg = RecallConfig.load(path)
    # Invalid mode should fall back to default with a warning
    assert cfg.tools.compression == "moderate"


def test_pipeline_config_max_outbound_tokens_default():
    cfg = RecallConfig()
    assert cfg.pipeline.max_outbound_tokens == 8000


def test_pipeline_config_max_outbound_tokens_loads_from_yaml(tmp_path):
    path = tmp_path / "sieve.yaml"
    path.write_text(yaml.safe_dump({
        "pipeline": {
            "conversation_turns": 5,
            "max_outbound_tokens": 4000,
        }
    }))
    cfg = RecallConfig.load(path)
    assert cfg.pipeline.conversation_turns == 5
    assert cfg.pipeline.max_outbound_tokens == 4000


def test_profile_owner_defaults():
    from sieve.config import RecallConfig
    cfg = RecallConfig()
    assert cfg.profile_owner.name == ""
    assert cfg.profile_owner.aliases == []


def test_profile_owner_loaded_from_yaml(tmp_path):
    import yaml
    from sieve.config import RecallConfig
    p = tmp_path / "sieve.yaml"
    p.write_text(yaml.safe_dump({
        "profile_owner": {
            "name": "Jamie Rivera",
            "aliases": ["Jamie", "I", "me", "the user"],
        }
    }))
    cfg = RecallConfig.load(p)
    assert cfg.profile_owner.name == "Jamie Rivera"
    assert "Jamie" in cfg.profile_owner.aliases
    assert "the user" in cfg.profile_owner.aliases


def test_profile_owner_aliases_string_wrapped_to_list(tmp_path):
    import yaml
    from sieve.config import RecallConfig
    p = tmp_path / "sieve.yaml"
    p.write_text(yaml.safe_dump({
        "profile_owner": {
            "name": "John Doe",
            "aliases": "Johnny",
        }
    }))
    cfg = RecallConfig.load(p)
    assert cfg.profile_owner.aliases == ["Johnny"]


def test_writer_ghost_validator_default_enabled():
    from sieve.config import RecallConfig
    cfg = RecallConfig()
    assert cfg.writer.ghost_validator_enabled is True


def test_writer_ghost_validator_loaded_from_yaml(tmp_path):
    import yaml
    from sieve.config import RecallConfig
    p = tmp_path / "sieve.yaml"
    p.write_text(yaml.safe_dump({"writer": {"ghost_validator_enabled": False}}))
    cfg = RecallConfig.load(p)
    assert cfg.writer.ghost_validator_enabled is False


def test_retrieval_config_defaults():
    from sieve.config import RecallConfig
    cfg = RecallConfig()
    assert cfg.retrieval.temporal_dedup_enabled is True


def test_retrieval_temporal_dedup_can_be_disabled(tmp_path):
    import yaml
    from sieve.config import RecallConfig
    p = tmp_path / "sieve.yaml"
    p.write_text(yaml.safe_dump({"retrieval": {"temporal_dedup": False}}))
    cfg = RecallConfig.load(p)
    assert cfg.retrieval.temporal_dedup_enabled is False


def test_schema_v2_flag_default_off():
    """schema_v2 must default to False — new code paths gated."""
    from sieve.config import RecallConfig
    cfg = RecallConfig()
    assert cfg.ablation.schema_v2 is False


def test_schema_v2_flag_loaded_from_yaml(tmp_path):
    import yaml
    from sieve.config import RecallConfig
    p = tmp_path / "sieve.yaml"
    p.write_text(yaml.safe_dump({"ablation": {"schema_v2": True}}))
    cfg = RecallConfig.load(p)
    assert cfg.ablation.schema_v2 is True


def test_empty_yaml_produces_dataclass_defaults(tmp_path):
    """Loading an empty YAML must yield the exact same config as RecallConfig().

    This guards against loader defaults drifting from dataclass defaults — a
    bug that previously flipped closed_world, response_verification, and
    extreme_summary from their documented shipping values, because
    ``sieve init`` writes a minimal yaml that exercises every absent-key
    branch of the loader.
    """
    p = tmp_path / "sieve.yaml"
    p.write_text("")  # completely empty config
    loaded = RecallConfig.load(p)
    dataclass_default = RecallConfig()

    # Compare ablation block field-by-field so a failure names the field.
    for field_name in dataclass_default.ablation.__dataclass_fields__:
        assert getattr(loaded.ablation, field_name) == getattr(
            dataclass_default.ablation, field_name
        ), (
            f"ablation.{field_name}: loader default "
            f"{getattr(loaded.ablation, field_name)!r} != dataclass default "
            f"{getattr(dataclass_default.ablation, field_name)!r}"
        )
    # And the other config sections that have the same drift risk.
    assert loaded.writer.model == dataclass_default.writer.model
    assert (
        loaded.writer.ghost_validator_enabled
        == dataclass_default.writer.ghost_validator_enabled
    )
    assert loaded.tools.enabled == dataclass_default.tools.enabled
    assert (
        loaded.retrieval.temporal_dedup_enabled
        == dataclass_default.retrieval.temporal_dedup_enabled
    )

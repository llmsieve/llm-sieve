"""WriterConfig.model default tests — validates the 2026-04-22 flip
from 'auto' to qwen3.5:4b."""
from __future__ import annotations

from sieve.config import WriterConfig, RecallConfig
from sieve.writer import resolve_writer_model


def test_writer_model_default_is_qwen3_5_4b():
    """Default changed 2026-04-22 based on 20-msg fact-extraction smoke."""
    assert WriterConfig().model == "qwen3.5:4b"


def test_writer_fallback_model_default_is_auto():
    """Fallback retains 'auto' — when the primary explicit model fails,
    fall back to provider.default_model."""
    assert WriterConfig().fallback_model == "auto"


def test_yaml_override_writer_model_to_auto_works():
    """Users can opt back into 'auto' (main model) via YAML."""
    import yaml
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "sieve.yaml"
        yaml_path.write_text(yaml.safe_dump({"writer": {"model": "auto"}}))
        c = RecallConfig.load(yaml_path)
        assert c.writer.model == "auto"


def test_yaml_override_writer_model_to_explicit_works():
    """Users can override with an explicit model name via YAML."""
    import yaml
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "sieve.yaml"
        yaml_path.write_text(yaml.safe_dump({"writer": {"model": "qwen3.5:2b"}}))
        c = RecallConfig.load(yaml_path)
        assert c.writer.model == "qwen3.5:2b"


def test_resolve_writer_model_auto_uses_provider_default():
    """When writer.model='auto', resolve_writer_model routes to
    provider.default_model."""
    from sieve.config import ProviderConfig

    cfg = RecallConfig(
        provider=ProviderConfig(default_model="qwen3:14b"),
        writer=WriterConfig(model="auto"),
    )
    assert resolve_writer_model(cfg) == "qwen3:14b"


def test_resolve_writer_model_explicit_wins():
    """Explicit model wins over provider.default_model."""
    from sieve.config import ProviderConfig

    cfg = RecallConfig(
        provider=ProviderConfig(default_model="qwen3:14b"),
        writer=WriterConfig(model="qwen3.5:4b"),
    )
    assert resolve_writer_model(cfg) == "qwen3.5:4b"

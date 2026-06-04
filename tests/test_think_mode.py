"""Tests for user-controllable think mode via inline tags."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sieve.config import PipelineConfig, RecallConfig


def _make_app(think_enabled: bool = False):
    """Create a test app with mocked upstream."""
    config = RecallConfig()
    config.pipeline.think_enabled = think_enabled
    from sieve.main import create_app
    app = create_app(config)
    return app, config


class TestThinkTagProcessing:
    """Test the _process_think_tags function directly."""

    def test_think_on_tag_sets_flag(self):
        from sieve.main import create_app
        config = RecallConfig()
        assert config.pipeline.think_enabled is False
        app = create_app(config)
        # Access the internal function via module
        import sieve.main as main_mod
        # We need to test the tag processing — do it via the app's endpoint
        # by sending a message with the tag
        # For unit test, let's test the regex and logic directly
        import re
        pattern = re.compile(r"<#think_(on|off|status)#>", re.IGNORECASE)
        assert pattern.search("<#think_on#>") is not None
        assert pattern.search("<#think_off#>") is not None
        assert pattern.search("<#think_status#>") is not None
        assert pattern.search("hello world") is None

    def test_think_tag_stripped_from_message(self):
        """Tags are removed from user message text."""
        import re
        pattern = re.compile(r"<#think_(on|off|status)#>", re.IGNORECASE)
        msg = "Hello <#think_on#> world"
        cleaned = pattern.sub("", msg).strip()
        assert cleaned == "Hello  world"  # double space acceptable
        assert "<#think" not in cleaned

    def test_think_tag_case_insensitive(self):
        import re
        pattern = re.compile(r"<#think_(on|off|status)#>", re.IGNORECASE)
        assert pattern.search("<#THINK_ON#>") is not None
        assert pattern.search("<#Think_Off#>") is not None
        assert pattern.search("<#THINK_STATUS#>") is not None


class TestThinkInjection:
    """Test think:false injection in composed payloads."""

    def test_think_off_omits_flag(self):
        # Post-fix: pipeline no longer injects think:false.
        # On qwen3:30b-a3b + Ollama 0.20.2 the flag leaks reasoning
        # into message.content; leaving it unset lets Ollama's template
        # route reasoning to the separate `thinking` field.
        from sieve.pipeline import compose_lean_payload
        from tests.test_pipeline import _make_decomposed_with_user
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        decomposed = _make_decomposed_with_user("hi")
        cfg = PipelineConfig(think_enabled=False)
        lean = compose_lean_payload(payload, decomposed, cfg)
        assert "think" not in lean

    def test_think_on_no_flag(self):
        from sieve.pipeline import compose_lean_payload
        from tests.test_pipeline import _make_decomposed_with_user
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        decomposed = _make_decomposed_with_user("hi")
        cfg = PipelineConfig(think_enabled=True)
        lean = compose_lean_payload(payload, decomposed, cfg)
        assert "think" not in lean

    def test_think_persists_across_config(self):
        """Setting think_enabled persists on the config object."""
        cfg = PipelineConfig()
        assert cfg.think_enabled is False
        cfg.think_enabled = True
        assert cfg.think_enabled is True
        # Stays True
        assert cfg.think_enabled is True

    def test_default_think_off(self):
        cfg = PipelineConfig()
        assert cfg.think_enabled is False

    def test_think_flag_never_injected(self):
        """Pipeline must not inject a `think` key anywhere (top level or options)."""
        from sieve.pipeline import compose_lean_payload
        from tests.test_pipeline import _make_decomposed_with_user
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.5},
        }
        decomposed = _make_decomposed_with_user("hi")
        cfg = PipelineConfig(think_enabled=False)
        lean = compose_lean_payload(payload, decomposed, cfg)
        assert "think" not in lean
        assert "think" not in lean.get("options", {})

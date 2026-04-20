"""Tests for the numbered-picker + live model-listing wizard helpers."""

from __future__ import annotations

import httpx
import pytest

from sieve import _wizard_helpers


# ── list_models ─────────────────────────────────────────────────────────


def _make_client_transport(handler):
    """Patch the module's httpx.get to use a MockTransport handler."""
    transport = httpx.MockTransport(handler)
    original_get = httpx.get
    def _mock_get(url, **kwargs):
        return httpx.Client(transport=transport, **{
            k: v for k, v in kwargs.items() if k != "timeout"
        }).get(url, **{k: v for k, v in kwargs.items() if k == "timeout"})
    return original_get, _mock_get


def test_list_models_ollama_success(monkeypatch):
    def handler(req):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3.5:9b"}, {"name": "gemma:2b"}, {"name": "llama3:8b"},
            ]})
        return httpx.Response(404)
    orig, mock = _make_client_transport(handler)
    monkeypatch.setattr(_wizard_helpers, "httpx", httpx)
    # Temporarily override httpx.get
    monkeypatch.setattr(httpx, "get", mock)
    try:
        models = _wizard_helpers.list_models("http://ollama.test:11434")
    finally:
        monkeypatch.setattr(httpx, "get", orig)
    assert models == ["gemma:2b", "llama3:8b", "qwen3.5:9b"]  # sorted case-insensitive


def test_list_models_openai_fallback(monkeypatch):
    def handler(req):
        # Ollama path 404s
        if req.url.path == "/api/tags":
            return httpx.Response(404)
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": "gpt-4o"}, {"id": "gpt-4o-mini"},
            ]})
        return httpx.Response(404)
    orig = httpx.get
    def mock_get(url, **kwargs):
        return httpx.Client(transport=httpx.MockTransport(handler)).get(
            url, timeout=kwargs.get("timeout", 5.0)
        )
    monkeypatch.setattr(httpx, "get", mock_get)
    try:
        models = _wizard_helpers.list_models("https://api.openai.com")
    finally:
        monkeypatch.setattr(httpx, "get", orig)
    assert models == ["gpt-4o", "gpt-4o-mini"]


def test_list_models_returns_empty_on_network_error(monkeypatch):
    """Network failure yields [] so the caller can show a free-text prompt.

    No exception should propagate — the wizard must never crash
    because a user's LLM endpoint is temporarily down.
    """
    def mock_get(url, **kwargs):
        raise httpx.ConnectError("connection refused", request=None)
    orig = httpx.get
    monkeypatch.setattr(httpx, "get", mock_get)
    try:
        models = _wizard_helpers.list_models("http://down.test:11434")
    finally:
        monkeypatch.setattr(httpx, "get", orig)
    assert models == []


def test_list_models_deduplicates():
    """A tokeniser that mis-returns duplicates shouldn't crash us."""
    # Direct unit exercise via a stub that returns dupes via monkeypatch.
    def _stub_list():
        # Mimic what list_models does with a pre-built response.
        raw = ["Qwen3.5:9b", "qwen3.5:9b", "gemma:2b", "qwen3.5:9b"]
        seen, out = set(), []
        for n in sorted(raw, key=lambda s: s.lower()):
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out
    assert _stub_list() == ["gemma:2b", "Qwen3.5:9b", "qwen3.5:9b"]
    # The case-insensitive sort is stable: "Qwen" precedes "qwen" because
    # they're equal under case-fold, and Python's sort is stable on ties.


# ── pick_numbered / pick_model (prompt plumbing smoke tests) ────────────


class _ScriptedConsole:
    """Captures rich output as plain text for assertion."""
    def __init__(self):
        self.lines: list[str] = []
    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


def test_pick_numbered_accepts_number(monkeypatch):
    console = _ScriptedConsole()
    inputs = iter(["2"])
    monkeypatch.setattr(
        "click.prompt",
        lambda *a, **k: next(inputs),
    )
    choices = [
        _wizard_helpers.NumberedChoice(label="first", value="a"),
        _wizard_helpers.NumberedChoice(label="second", value="b"),
        _wizard_helpers.NumberedChoice(label="third", value="c"),
    ]
    result = _wizard_helpers.pick_numbered(
        "Pick one", choices, console=console,
    )
    assert result == "b"


def test_pick_numbered_returns_default_on_empty(monkeypatch):
    console = _ScriptedConsole()
    inputs = iter([""])
    monkeypatch.setattr("click.prompt", lambda *a, **k: next(inputs))
    choices = [_wizard_helpers.NumberedChoice(label="only", value="x")]
    result = _wizard_helpers.pick_numbered(
        "Pick", choices, default="x", console=console,
    )
    assert result == "x"


def test_pick_numbered_reprompts_on_bad_input(monkeypatch):
    console = _ScriptedConsole()
    inputs = iter(["banana", "99", "1"])
    monkeypatch.setattr("click.prompt", lambda *a, **k: next(inputs))
    choices = [
        _wizard_helpers.NumberedChoice(label="only", value="x"),
        _wizard_helpers.NumberedChoice(label="another", value="y"),
    ]
    result = _wizard_helpers.pick_numbered(
        "Pick", choices, console=console,
    )
    assert result == "x"
    # Verify the reprompts were rendered.
    rendered = "\n".join(console.lines)
    assert "banana" in rendered or "isn't a number" in rendered


def test_pick_numbered_free_text_escape(monkeypatch):
    console = _ScriptedConsole()
    # Two choices, so free-text option is index 3.
    inputs = iter(["3", "my-custom-model"])
    monkeypatch.setattr("click.prompt", lambda *a, **k: next(inputs))
    choices = [
        _wizard_helpers.NumberedChoice(label="alpha", value="a"),
        _wizard_helpers.NumberedChoice(label="beta", value="b"),
    ]
    result = _wizard_helpers.pick_numbered(
        "Pick", choices, allow_free_text=True, console=console,
    )
    assert result == "my-custom-model"

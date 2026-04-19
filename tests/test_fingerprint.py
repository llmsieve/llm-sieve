"""Tests for Phase 3: Payload fingerprinting and decomposition."""

from __future__ import annotations

import json
import logging

import pytest

from sieve.config import StoreConfig
from sieve.fingerprint import (
    DecomposedPayload,
    FingerprintCache,
    Section,
    _estimate_tokens,
    _hash_content,
    decompose,
)
from sieve.store import MemoryStore


# --- Helpers ---


def _ollama_payload(
    user_msg: str = "Hello",
    system_prompt: str | None = None,
    tools: list | None = None,
    history: list | None = None,
    stream: bool = True,
) -> dict:
    """Build a realistic Ollama /api/chat payload."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    payload = {"model": "qwen3.5:35b", "messages": messages, "stream": stream}
    if tools:
        payload["tools"] = tools
    return payload


def _openai_payload(
    user_msg: str = "Hello",
    system_prompt: str | None = None,
    tools: list | None = None,
) -> dict:
    """Build a realistic OpenAI /v1/chat/completions payload."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "model": "gpt-4",
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    return payload


SAMPLE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. You help users with their tasks. "
    "Always be polite and thorough in your responses. " * 50
)  # ~600 tokens

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
]

SAMPLE_HISTORY = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have access to weather data."},
    {"role": "user", "content": "Can you help me with Python?"},
    {"role": "assistant", "content": "Of course! What do you need help with?"},
]


@pytest.fixture
def cache():
    return FingerprintCache(store=None)


# --- Token estimation ---


def test_estimate_tokens_short():
    assert _estimate_tokens("hello") >= 1


def test_estimate_tokens_long():
    text = "word " * 1000  # ~5000 chars → ~1250 tokens
    tokens = _estimate_tokens(text)
    assert 1000 < tokens < 2000


# --- Hashing ---


def test_hash_deterministic():
    assert _hash_content("hello") == _hash_content("hello")


def test_hash_different_for_different_content():
    assert _hash_content("hello") != _hash_content("world")


# --- FingerprintCache ---


def test_cache_first_access_is_changed(cache):
    assert cache.check_and_update("key1", "hash1") is True


def test_cache_same_hash_unchanged(cache):
    cache.check_and_update("key1", "hash1")
    assert cache.check_and_update("key1", "hash1") is False


def test_cache_different_hash_changed(cache):
    cache.check_and_update("key1", "hash1")
    assert cache.check_and_update("key1", "hash2") is True


def test_cache_with_store(tmp_path):
    """FingerprintCache should persist to store when available."""
    config = StoreConfig(path=str(tmp_path / "test.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test")
    ms.open()
    ms.init_schema()

    cache = FingerprintCache(store=ms)
    cache.check_and_update("system_prompt", "abc123")

    # Verify it was persisted
    fp = ms.get_fingerprint("system_prompt")
    assert fp is not None
    assert fp["hash"] == "abc123"

    # New cache instance should load from store
    cache2 = FingerprintCache(store=ms)
    assert cache2.check_and_update("system_prompt", "abc123") is False  # unchanged
    assert cache2.check_and_update("system_prompt", "xyz789") is True   # changed

    ms.close()


# --- Decomposition: Ollama format ---


def test_decompose_simple_message(cache):
    payload = _ollama_payload("Hello world")
    result = decompose(payload, cache, api_format="ollama")

    assert result.format == "ollama"
    user_msg = result.section_by_name("user_message")
    assert user_msg is not None
    assert user_msg.content == "Hello world"
    assert user_msg.changed is True


def test_decompose_with_system_prompt(cache):
    payload = _ollama_payload("Hi", system_prompt=SAMPLE_SYSTEM_PROMPT)
    result = decompose(payload, cache, api_format="ollama")

    sp = result.section_by_name("system_prompt")
    assert sp is not None
    assert sp.token_estimate > 100
    assert sp.changed is True  # first time


def test_decompose_with_tools(cache):
    payload = _ollama_payload("Hi", tools=SAMPLE_TOOLS)
    result = decompose(payload, cache, api_format="ollama")

    tools = result.section_by_name("tools")
    assert tools is not None
    assert tools.token_estimate > 10


def test_decompose_with_history(cache):
    payload = _ollama_payload("New question", history=SAMPLE_HISTORY)
    result = decompose(payload, cache, api_format="ollama")

    hist = result.section_by_name("conversation_history")
    assert hist is not None
    assert hist.token_estimate > 10


def test_decompose_full_payload(cache):
    """Full payload with all sections."""
    payload = _ollama_payload(
        "Tell me about Dubai",
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
        history=SAMPLE_HISTORY,
    )
    result = decompose(payload, cache, api_format="ollama")

    names = {s.name for s in result.sections}
    assert "system_prompt" in names
    assert "tools" in names
    assert "conversation_history" in names
    assert "user_message" in names
    assert "options" in names  # model, stream
    assert result.total_tokens > 0


# --- Change detection ---


def test_unchanged_sections_detected(cache):
    """Second request with same system prompt + tools should show unchanged."""
    payload = _ollama_payload(
        "First question",
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
    )
    result1 = decompose(payload, cache, api_format="ollama")
    assert result1.section_by_name("system_prompt").changed is True
    assert result1.section_by_name("tools").changed is True

    # Second request — same system prompt & tools, different user message
    payload2 = _ollama_payload(
        "Second question",
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
    )
    result2 = decompose(payload2, cache, api_format="ollama")
    assert result2.section_by_name("system_prompt").changed is False  # unchanged!
    assert result2.section_by_name("tools").changed is False          # unchanged!
    assert result2.section_by_name("user_message").changed is True    # always new


def test_changed_system_prompt_detected(cache):
    """Changing the system prompt should be detected."""
    payload1 = _ollama_payload("Hi", system_prompt="You are a helpful assistant.")
    decompose(payload1, cache, api_format="ollama")

    payload2 = _ollama_payload("Hi", system_prompt="You are a code reviewer.")
    result2 = decompose(payload2, cache, api_format="ollama")
    assert result2.section_by_name("system_prompt").changed is True


def test_unchanged_tokens_counted(cache):
    """unchanged_tokens should reflect sections that didn't change."""
    payload = _ollama_payload("Q1", system_prompt=SAMPLE_SYSTEM_PROMPT, tools=SAMPLE_TOOLS)
    decompose(payload, cache, api_format="ollama")

    payload2 = _ollama_payload("Q2", system_prompt=SAMPLE_SYSTEM_PROMPT, tools=SAMPLE_TOOLS)
    result = decompose(payload2, cache, api_format="ollama")

    assert result.unchanged_tokens > 0
    assert result.changed_tokens > 0
    assert result.unchanged_tokens > result.changed_tokens  # system prompt is large


# --- OpenAI format ---


def test_decompose_openai_format(cache):
    payload = _openai_payload("Hello", system_prompt="You are helpful.", tools=SAMPLE_TOOLS)
    result = decompose(payload, cache, api_format="openai")

    assert result.format == "openai"
    assert result.section_by_name("system_prompt") is not None
    assert result.section_by_name("tools") is not None
    assert result.section_by_name("user_message") is not None


# --- Workspace file detection ---


def test_workspace_files_extracted(cache):
    """System messages with file-like content should be detected as workspace files."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": "# AGENTS.md\n\nThis agent is designed to...\n```python\nprint('hello')\n```\n" * 10},
        {"role": "user", "content": "Hi"},
    ]
    payload = {"model": "qwen3.5:35b", "messages": messages}
    result = decompose(payload, cache, api_format="ollama")

    ws = result.section_by_name("workspace_files")
    assert ws is not None
    assert ws.token_estimate > 50


# --- Logging ---


def test_log_breakdown(cache, caplog):
    """decompose should log the section breakdown."""
    payload = _ollama_payload(
        "Tell me about Dubai",
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        tools=SAMPLE_TOOLS,
    )
    with caplog.at_level(logging.INFO, logger="recall.fingerprint"):
        decompose(payload, cache, api_format="ollama")

    assert "Payload breakdown" in caplog.text
    assert "system_prompt" in caplog.text
    assert "tokens" in caplog.text


# --- Section dataclass ---


def test_section_log_line():
    s = Section(name="system_prompt", content="...", token_estimate=9600, hash="abc", changed=False)
    assert "system_prompt" in s.log_line()
    assert "9,600" in s.log_line()
    assert "unchanged" in s.log_line()


def test_section_log_line_new():
    s = Section(name="user_message", content="Hi", token_estimate=1, hash="xyz", changed=True)
    assert "new" in s.log_line()


# --- Edge cases ---


def test_empty_payload(cache):
    result = decompose({}, cache, api_format="ollama")
    assert result.total_tokens == 0
    assert len(result.sections) == 0


def test_no_system_prompt(cache):
    payload = _ollama_payload("Hi")
    result = decompose(payload, cache, api_format="ollama")
    assert result.section_by_name("system_prompt") is None
    assert result.section_by_name("user_message") is not None


def test_decompose_time_tracked(cache):
    payload = _ollama_payload("Hi", system_prompt=SAMPLE_SYSTEM_PROMPT)
    result = decompose(payload, cache, api_format="ollama")
    assert result.decompose_time_us >= 0


# --- Simulated OpenClaw bloated payload ---


def test_openclaw_bloated_payload(cache):
    """Simulate a realistic OpenClaw payload (~47k tokens) and verify decomposition."""
    big_system = "You are an AI assistant with extensive capabilities. " * 500  # ~4000 tokens
    big_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool number {i} that does something useful " * 5,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"},
                        "arg2": {"type": "integer"},
                    },
                },
            },
        }
        for i in range(30)  # 30 tools → ~4000 tokens
    ]
    big_history = []
    for i in range(20):
        big_history.append({"role": "user", "content": f"Question {i}: " + "context " * 50})
        big_history.append({"role": "assistant", "content": f"Answer {i}: " + "response " * 50})

    payload = _ollama_payload(
        "What's the weather in Dubai?",
        system_prompt=big_system,
        tools=big_tools,
        history=big_history,
    )

    result1 = decompose(payload, cache, api_format="ollama")
    assert result1.total_tokens > 5000
    # All sections should be new on first request
    for s in result1.sections:
        assert s.changed is True

    # Second request — same bloat, different question
    payload2 = _ollama_payload(
        "How is Sara doing?",
        system_prompt=big_system,
        tools=big_tools,
        history=big_history,
    )
    result2 = decompose(payload2, cache, api_format="ollama")

    # system_prompt and tools should be unchanged
    assert result2.section_by_name("system_prompt").changed is False
    assert result2.section_by_name("tools").changed is False
    assert result2.section_by_name("user_message").changed is True

    # The unchanged portion should be the majority
    assert result2.unchanged_tokens > result2.changed_tokens

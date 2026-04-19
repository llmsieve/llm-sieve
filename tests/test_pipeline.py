"""Tests for Phase 4: Payload stripping and lean composition."""

from __future__ import annotations

import json
import logging

import pytest

from sieve.config import PipelineConfig
from sieve.fingerprint import FingerprintCache, decompose
from sieve.fingerprint import DecomposedPayload, Section
from sieve.pipeline import (
    GENERAL_LEAN_SYSTEM_PROMPT,
    LEAN_SYSTEM_PROMPT,
    RECALL_TOOL,
    _last_n_turns,
    compose_lean_payload,
)


@pytest.fixture
def config():
    return PipelineConfig(conversation_turns=3)


@pytest.fixture
def cache():
    return FingerprintCache(store=None)


def _decompose(payload: dict, cache: FingerprintCache) -> tuple:
    """Helper: decompose + return both."""
    decomposed = decompose(payload, cache, api_format="ollama")
    return payload, decomposed


# --- _last_n_turns ---


def test_last_n_turns_empty():
    assert _last_n_turns([], 3) == []


def test_last_n_turns_fewer_than_n():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    result = _last_n_turns(history, 3)
    assert len(result) == 2


def test_last_n_turns_exact():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]
    result = _last_n_turns(history, 3)
    assert len(result) == 6  # All 3 turns


def test_last_n_turns_trims_old():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
        {"role": "user", "content": "Q4"},
        {"role": "assistant", "content": "A4"},
        {"role": "user", "content": "Q5"},
        {"role": "assistant", "content": "A5"},
    ]
    result = _last_n_turns(history, 2)
    # Keeps 2 user messages + their assistants, plus the assistant reply
    # that immediately precedes the first kept user message
    assert result[-1]["content"] == "A5"
    # The last 2 user messages (Q4, Q5) must be present
    user_contents = [m["content"] for m in result if m["role"] == "user"]
    assert user_contents == ["Q4", "Q5"]


def test_last_n_turns_zero():
    history = [{"role": "user", "content": "Q1"}]
    assert _last_n_turns(history, 0) == []


# --- compose_lean_payload ---


def test_lean_payload_has_lean_system_prompt(config, cache):
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "Bloated system prompt " * 500},
            {"role": "user", "content": "Hello"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    # System prompt should be the lean one, not the bloated one
    system_msgs = [m for m in lean["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == LEAN_SYSTEM_PROMPT
    assert "Bloated" not in system_msgs[0]["content"]


def test_owner_pin_preserved_from_explicit_arg(config, cache):
    """Explicit profile_owner_pin arg must be appended to LEAN_SYSTEM_PROMPT."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "Bloated " * 100},
            {"role": "user", "content": "Who am I?"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(
        payload, decomposed, config,
        profile_owner_pin="Jamie Rivera is a 41-year-old civil engineer.",
    )
    sys_text = lean["messages"][0]["content"]
    assert sys_text.startswith(LEAN_SYSTEM_PROMPT)
    assert "Jamie Rivera is a 41-year-old civil engineer." in sys_text


def test_owner_pin_detected_from_inbound_system_prompt(config, cache):
    """When no explicit pin passed, detector picks up "The person speaking is X." """
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": (
                "You are a helpful AI assistant. "
                "The person speaking is Jamie Rivera, a 41-year-old civil "
                "engineer living in Bristol. Answer clearly."
            )},
            {"role": "user", "content": "What's on my calendar?"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)
    sys_text = lean["messages"][0]["content"]
    assert "Jamie Rivera" in sys_text


def test_no_owner_pin_leaves_lean_prompt_clean(config, cache):
    """With no pin detectable and no explicit arg, the lean system
    prompt is emitted verbatim (no stray whitespace, no extra newlines)."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)
    assert lean["messages"][0]["content"] == LEAN_SYSTEM_PROMPT


def test_pure_general_uses_general_prompt(config, cache):
    """pure_general=True swaps to GENERAL_LEAN_SYSTEM_PROMPT.

    Pure general-knowledge queries should never see memory-focused framing;
    the model answers from its own knowledge.
    """
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "Bloated " * 100},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config, pure_general=True)
    sys_text = lean["messages"][0]["content"]
    assert sys_text == GENERAL_LEAN_SYSTEM_PROMPT
    # Guard: the memory-framing phrase must not leak into the general prompt
    assert "personal memory" not in sys_text
    assert "recall tool" not in sys_text


def test_pure_general_drops_owner_pin(config, cache):
    """pure-G queries drop the owner pin — identity grounding is
    irrelevant when the question is general knowledge."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": (
                "You are a helpful AI assistant. "
                "The person speaking is Jamie Rivera, a 41-year-old civil "
                "engineer living in Bristol. Answer clearly."
            )},
            {"role": "user", "content": "What is 2+2?"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(
        payload, decomposed, config,
        profile_owner_pin="Jamie Rivera is a 41-year-old civil engineer.",
        pure_general=True,
    )
    sys_text = lean["messages"][0]["content"]
    assert sys_text == GENERAL_LEAN_SYSTEM_PROMPT
    assert "Jamie Rivera" not in sys_text


def test_pure_general_default_false_uses_memory_prompt(config, cache):
    """Regression guard: the new param defaults to False, so existing callers
    that do not opt in still get the memory-focused LEAN_SYSTEM_PROMPT."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "bloated"},
            {"role": "user", "content": "Where do I live?"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)
    assert lean["messages"][0]["content"] == LEAN_SYSTEM_PROMPT


def test_lean_payload_has_recall_tool(config, cache):
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "big_tool"}} for _ in range(20)],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    # BUG-001 fix: recall tool prepended, agent tools preserved
    assert lean["tools"][0]["function"]["name"] == "recall"
    assert len(lean["tools"]) == 21  # 1 recall + 20 agent tools


def test_lean_payload_has_user_message(config, cache):
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "What is Dubai like?"}],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    user_msgs = [m for m in lean["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "What is Dubai like?"


def test_lean_payload_preserves_model(config, cache):
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)
    assert lean["model"] == "qwen3.5:35b"


def test_lean_payload_preserves_stream_flag(config, cache):
    for stream_val in [True, False]:
        payload = {
            "model": "qwen3.5:35b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": stream_val,
        }
        decomposed = decompose(payload, cache, api_format="ollama")
        lean = compose_lean_payload(payload, decomposed, config)
        assert lean["stream"] == stream_val


def test_lean_payload_preserves_options(config, cache):
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "Hi"}],
        "options": {"temperature": 0.7, "top_p": 0.9},
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)
    assert lean["options"]["temperature"] == 0.7


def test_lean_payload_trims_history(cache):
    """Should keep only last 2 turns when configured."""
    config = PipelineConfig(conversation_turns=2)
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
            {"role": "assistant", "content": "A3"},
            {"role": "user", "content": "Current question"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    msgs = lean["messages"]
    # Should have: system + 2 turns (4 msgs) + user message = 6
    user_msgs = [m for m in msgs if m["role"] == "user"]
    # Last user message + 2 from history = 3
    assert len(user_msgs) == 3
    # First user in history should be Q2 (Q1 trimmed)
    assert user_msgs[0]["content"] == "Q2"
    assert user_msgs[-1]["content"] == "Current question"


def test_lean_payload_strips_workspace_files(config, cache):
    """Workspace file content should not appear in lean payload."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "# AGENTS.md\n" + "workspace content " * 200},
            {"role": "user", "content": "Hello"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    lean_text = json.dumps(lean)
    assert "AGENTS.md" not in lean_text
    assert "workspace content" not in lean_text


# --- Token reduction ---


def test_massive_reduction(config, cache):
    """Simulate a ~47k token payload and verify massive reduction."""
    big_system = "You are an AI assistant with capabilities. " * 500
    big_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"Tool {i} description " * 10,
                "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
            },
        }
        for i in range(30)
    ]
    big_history = []
    for i in range(20):
        big_history.append({"role": "user", "content": f"Question {i}: " + "context " * 50})
        big_history.append({"role": "assistant", "content": f"Answer {i}: " + "detail " * 50})

    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": big_system},
            *big_history,
            {"role": "user", "content": "Hello"},
        ],
        "tools": big_tools,
        "stream": False,
    }

    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    original_size = len(json.dumps(payload))
    lean_size = len(json.dumps(lean))

    # Tools are now passed through (BUG-001 fix), so reduction is from system+history stripping
    reduction = (1 - lean_size / original_size) * 100
    assert reduction > 50, f"Expected >50% reduction, got {reduction:.0f}%"

    # System and history tokens should be stripped (tools are now passed through per BUG-001 fix)
    from sieve.fingerprint import _estimate_tokens
    lean_tokens = _estimate_tokens(json.dumps(lean))
    original_tokens = _estimate_tokens(json.dumps(payload))
    # Tools pass through, but system bloat + old history should be gone
    assert lean_tokens < original_tokens * 0.5, f"Expected significant reduction, got {lean_tokens}/{original_tokens} tokens"


def test_reduction_logged(config, cache, caplog):
    """compose_lean_payload should log the reduction."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [
            {"role": "system", "content": "Big prompt " * 200},
            {"role": "user", "content": "Hi"},
        ],
    }
    decomposed = decompose(payload, cache, api_format="ollama")

    with caplog.at_level(logging.INFO, logger="recall.pipeline"):
        compose_lean_payload(payload, decomposed, config)

    assert "Strip:" in caplog.text
    assert "reduction" in caplog.text


# --- Edge cases ---


def test_no_history(config, cache):
    """Payload with no conversation history should work fine."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    msgs = lean["messages"]
    assert len(msgs) == 2  # system + user
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_no_system_prompt_in_original(config, cache):
    """Original payload without system prompt should still get lean prompt."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    decomposed = decompose(payload, cache, api_format="ollama")
    lean = compose_lean_payload(payload, decomposed, config)

    system_msgs = [m for m in lean["messages"] if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == LEAN_SYSTEM_PROMPT


def test_openai_format_works(config, cache):
    """OpenAI format payloads should be composed correctly."""
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "Big system prompt " * 200},
            {"role": "user", "content": "Hello"},
        ],
        "tools": [{"type": "function", "function": {"name": "web_search"}}],
        "stream": True,
    }
    decomposed = decompose(payload, cache, api_format="openai")
    lean = compose_lean_payload(payload, decomposed, config)

    assert lean["model"] == "gpt-4"
    assert lean["stream"] is True
    assert lean["tools"][0]["function"]["name"] == "recall"


def _make_decomposed_with_user(text: str) -> DecomposedPayload:
    d = DecomposedPayload(format="ollama")
    d.sections.append(Section(
        name="user_message", content=text,
        token_estimate=1, hash="x", changed=True,
    ))
    return d


def test_tools_passthrough_merges_agent_and_recall_tools():
    """Layer 1 / BUG-001: agent tools must appear in outbound alongside recall tool."""
    agent_tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [agent_tool],
    }
    decomposed = _make_decomposed_with_user("hi")
    lean = compose_lean_payload(payload, decomposed, PipelineConfig())

    tool_names = [t["function"]["name"] for t in lean["tools"]]
    assert "recall" in tool_names, "recall tool must always be injected"
    assert "web_search" in tool_names, "agent tools must be preserved (BUG-001)"
    # Recall should be first
    assert tool_names[0] == "recall"


def test_tools_passthrough_drops_recall_collision():
    """If agent sends a tool named 'recall', ours wins and theirs is dropped."""
    colliding = {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Agent's own recall tool",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    real_tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [colliding, real_tool],
    }
    decomposed = _make_decomposed_with_user("hi")
    lean = compose_lean_payload(payload, decomposed, PipelineConfig())

    tool_names = [t["function"]["name"] for t in lean["tools"]]
    # Exactly one "recall" (ours), and web_search preserved
    assert tool_names.count("recall") == 1
    assert "web_search" in tool_names
    # Ours is the one with description matching RECALL_TOOL
    recall_entry = next(t for t in lean["tools"] if t["function"]["name"] == "recall")
    assert recall_entry["function"]["description"] == RECALL_TOOL["function"]["description"]


def test_tools_passthrough_no_inbound_tools_still_gets_recall():
    """No tools in payload — recall tool still injected."""
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    decomposed = _make_decomposed_with_user("hi")
    lean = compose_lean_payload(payload, decomposed, PipelineConfig())
    tool_names = [t["function"]["name"] for t in lean["tools"]]
    assert tool_names == ["recall"]


def test_strip_preserves_model_stream_options():
    """Strip phase must preserve model, stream, options untouched."""
    payload = {
        "model": "qwen3.5:35b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"temperature": 0.3, "num_ctx": 8192},
    }
    decomposed = _make_decomposed_with_user("hi")
    lean = compose_lean_payload(payload, decomposed, PipelineConfig())
    assert lean["model"] == "qwen3.5:35b"
    assert lean["stream"] is True
    assert lean["options"] == {"temperature": 0.3, "num_ctx": 8192}


def test_tools_passthrough_handles_malformed_items():
    """Non-dict items in tools array should not crash; they pass through unchanged."""
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            "bogus_string",
            None,
            42,
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ],
    }
    decomposed = _make_decomposed_with_user("hi")
    lean = compose_lean_payload(payload, decomposed, PipelineConfig())

    # Outbound must include recall (first) + all 4 original items
    assert len(lean["tools"]) == 5
    assert lean["tools"][0]["function"]["name"] == "recall"
    assert "bogus_string" in lean["tools"]
    assert None in lean["tools"]
    assert 42 in lean["tools"]
    # The real web_search is also present
    assert any(
        isinstance(t, dict) and t.get("function", {}).get("name") == "web_search"
        for t in lean["tools"]
    )


# ── Layer 2: selective injection tests (Task 7) ────────────────────────────

class _StubSelection:
    def __init__(self, tools, level=0, reason="stub"):
        self.tools = tools
        self.level = level
        self.reason = reason
        self.confidence = 1.0


class _StubToolClassifier:
    """Stub ToolClassifier that returns whatever it was told to."""
    def __init__(self, tools_to_return):
        self._tools = tools_to_return
        self.last_query = None

    async def select(self, query):
        self.last_query = query
        return _StubSelection(self._tools)


def test_selective_injection_weather_query():
    """With a classifier returning web_search, that + recall must be in outbound."""
    import asyncio
    from sieve.pipeline import compose_with_tool_selection, RECALL_TOOL

    web_tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    unrelated_tool = {
        "type": "function",
        "function": {
            "name": "random_tool",
            "description": "Not relevant",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "weather in Tokyo"}],
        "tools": [web_tool, unrelated_tool],
    }
    decomposed = _make_decomposed_with_user("weather in Tokyo")
    classifier = _StubToolClassifier(tools_to_return=[web_tool])

    lean = asyncio.run(compose_with_tool_selection(
        payload, decomposed, PipelineConfig(),
        tool_classifier=classifier, user_query="weather in Tokyo",
    ))

    names = [t["function"]["name"] for t in lean["tools"]]
    assert names == ["recall", "web_search"]
    assert classifier.last_query == "weather in Tokyo"


def test_selective_injection_empty_selection_still_has_recall():
    """Classifier returns no tools — outbound still has recall."""
    import asyncio
    from sieve.pipeline import compose_with_tool_selection

    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "what is 2+2"}],
        "tools": [{"type": "function", "function": {"name": "x",
                   "description": "x", "parameters": {"type": "object",
                   "properties": {}, "required": []}}}],
    }
    decomposed = _make_decomposed_with_user("what is 2+2")
    classifier = _StubToolClassifier(tools_to_return=[])

    lean = asyncio.run(compose_with_tool_selection(
        payload, decomposed, PipelineConfig(),
        tool_classifier=classifier, user_query="what is 2+2",
    ))

    names = [t["function"]["name"] for t in lean["tools"]]
    assert names == ["recall"]


def test_selective_injection_classifier_exception_falls_back_to_passthrough():
    """If the classifier raises, fall back to Layer 1 passthrough (all agent tools)."""
    import asyncio
    from sieve.pipeline import compose_with_tool_selection

    class _BrokenClassifier:
        async def select(self, query):
            raise RuntimeError("oops")

    agent_tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [agent_tool],
    }
    decomposed = _make_decomposed_with_user("hi")

    lean = asyncio.run(compose_with_tool_selection(
        payload, decomposed, PipelineConfig(),
        tool_classifier=_BrokenClassifier(), user_query="hi",
    ))

    names = [t["function"]["name"] for t in lean["tools"]]
    # Both recall and web_search present (fell back to passthrough)
    assert "recall" in names
    assert "web_search" in names


# ── Token budget / conversation trimming (max_outbound_tokens) ────────────

def test_conversation_history_trimmed_before_compose():
    """Regression guard: history with 8 turns must be trimmed to last N turns
    BEFORE composing the lean payload. The lean messages list should contain
    exactly N user turns from history + the current user message.
    """
    # Build 8 prior turns + the current user query (17 messages total w/ system)
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(8):
        messages.append({"role": "user", "content": f"user turn {i}"})
        messages.append({"role": "assistant", "content": f"assistant reply {i}"})
    messages.append({"role": "user", "content": "current query"})

    payload = {"model": "m", "messages": messages}

    from sieve.fingerprint import FingerprintCache, decompose
    cache = FingerprintCache(None)
    decomposed = decompose(payload, cache, api_format="ollama")

    lean = compose_lean_payload(
        payload, decomposed, PipelineConfig(conversation_turns=3),
    )

    # The trim walks backwards and stops after seeing N user messages; the
    # assistant reply that precedes the oldest kept user turn is also kept
    # (it's valuable context for that turn). So with N=3 and 8 turns, we
    # expect user turns 5/6/7 + the current query, and assistant replies
    # 4/5/6/7 (asst 4 precedes user 5 and is kept by the walk).
    user_contents = [
        m["content"] for m in lean["messages"] if m.get("role") == "user"
    ]
    assert user_contents == [
        "user turn 5", "user turn 6", "user turn 7", "current query",
    ]
    assistant_contents = [
        m["content"] for m in lean["messages"] if m.get("role") == "assistant"
    ]
    assert assistant_contents == [
        "assistant reply 4", "assistant reply 5",
        "assistant reply 6", "assistant reply 7",
    ]


def test_token_budget_no_trim_when_under_budget():
    """Small payloads should pass through unchanged by the budget step."""
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    decomposed = _make_decomposed_with_user("hi")
    cfg = PipelineConfig(max_outbound_tokens=8000)
    lean = compose_lean_payload(payload, decomposed, cfg)

    # Token count should be tiny; no trimming should have happened
    from sieve.pipeline import _count_lean_tokens
    assert _count_lean_tokens(lean) < 8000
    # The current user message must still be present
    assert lean["messages"][-1] == {"role": "user", "content": "hi"}


def test_token_budget_reduces_history_to_one_turn():
    """When over budget, step 1 reduces history to last 1 turn."""
    # Build a big history — 10 long turns — exceeding a tight budget
    big_text = "x" * 1200  # ~300 tokens each
    messages = [{"role": "system", "content": "sys"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"{big_text} user{i}"})
        messages.append({"role": "assistant", "content": f"{big_text} asst{i}"})
    messages.append({"role": "user", "content": "current"})

    payload = {"model": "m", "messages": messages}

    from sieve.fingerprint import FingerprintCache, decompose
    cache = FingerprintCache(None)
    decomposed = decompose(payload, cache, api_format="ollama")

    # Give compose a generous history window; budget does the trimming
    cfg = PipelineConfig(conversation_turns=10, max_outbound_tokens=800)
    lean = compose_lean_payload(payload, decomposed, cfg)

    # After budget step 1, only the last 1 turn of history + current query
    history_user_msgs = [
        m for m in lean["messages"]
        if m.get("role") == "user" and m.get("content", "") != "current"
    ]
    assert len(history_user_msgs) == 1, (
        f"expected 1 historical user msg after budget trim, got {len(history_user_msgs)}"
    )
    # Current user query is always preserved
    assert lean["messages"][-1]["content"] == "current"


def test_token_budget_truncates_retrieved_context():
    """After reducing history to 1 turn, if still over budget, truncate the
    retrieved-context system message."""
    # Tiny history, no budget pressure from history alone
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    # But a huge retrieved context pushing us over the budget
    big_context = "context line. " * 2000  # ~7k tokens

    payload = {"model": "m", "messages": messages}

    from sieve.fingerprint import FingerprintCache, decompose
    cache = FingerprintCache(None)
    decomposed = decompose(payload, cache, api_format="ollama")

    cfg = PipelineConfig(max_outbound_tokens=1500)
    lean = compose_lean_payload(
        payload, decomposed, cfg, retrieved_context=big_context,
    )

    from sieve.pipeline import _count_lean_tokens
    # Should be within budget after truncation
    assert _count_lean_tokens(lean) <= 1500
    # The retrieved context should still exist but be much shorter
    system_msgs = [m for m in lean["messages"] if m.get("role") == "system"]
    # First system is lean prompt; second (if present) is the truncated context
    if len(system_msgs) >= 2:
        assert len(system_msgs[1]["content"]) < len(big_context)


def test_token_budget_warning_when_still_over_after_trim(caplog):
    """If the current user query alone is bigger than the budget, log a warning."""
    # A huge current user query — budget can't trim around this
    huge_query = "q " * 5000  # ~2500 tokens alone
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": huge_query}],
    }
    decomposed = _make_decomposed_with_user(huge_query)

    cfg = PipelineConfig(max_outbound_tokens=500)
    with caplog.at_level("WARNING", logger="recall.pipeline"):
        lean = compose_lean_payload(payload, decomposed, cfg)

    # The user message is untouched (we never truncate it)
    assert lean["messages"][-1]["content"] == huge_query
    # And the warning was logged
    assert any(
        "Token budget still exceeded" in rec.message
        for rec in caplog.records
    ), "expected warning about budget still exceeded"


def test_token_budget_applies_through_tool_selection_wrapper():
    """Budget must also apply in the compose_with_tool_selection path."""
    import asyncio
    from sieve.pipeline import compose_with_tool_selection

    class _PassthroughClassifier:
        async def select(self, query):
            class S: pass
            s = S()
            s.tools = []
            s.level = -2
            s.reason = "stub"
            s.confidence = 1.0
            return s

    # Build a payload with heavy history
    big_text = "y" * 1200
    messages = [{"role": "system", "content": "sys"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"{big_text} u{i}"})
        messages.append({"role": "assistant", "content": f"{big_text} a{i}"})
    messages.append({"role": "user", "content": "current"})

    payload = {"model": "m", "messages": messages, "tools": []}

    from sieve.fingerprint import FingerprintCache, decompose
    cache = FingerprintCache(None)
    decomposed = decompose(payload, cache, api_format="ollama")

    cfg = PipelineConfig(conversation_turns=10, max_outbound_tokens=800)
    lean = asyncio.run(compose_with_tool_selection(
        payload, decomposed, cfg,
        tool_classifier=_PassthroughClassifier(),
        user_query="current",
    ))

    history_user_msgs = [
        m for m in lean["messages"]
        if m.get("role") == "user" and m.get("content", "") != "current"
    ]
    assert len(history_user_msgs) == 1
    assert lean["messages"][-1]["content"] == "current"


# ── Think mode injection tests ───────────────────────────────────────────────


def test_think_off_does_not_inject_flag():
    """Post-fix: the pipeline no longer injects `think` under any config.
    See src/pipeline.py: think:false leaks reasoning into content on qwen3:30b-a3b."""
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    decomposed = _make_decomposed_with_user("hi")
    cfg = PipelineConfig(think_enabled=False)
    lean = compose_lean_payload(payload, decomposed, cfg)
    assert "think" not in lean


def test_think_on_no_flag():
    """With think_enabled=True, outbound body does NOT have think:false."""
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    decomposed = _make_decomposed_with_user("hi")
    cfg = PipelineConfig(think_enabled=True)
    lean = compose_lean_payload(payload, decomposed, cfg)
    assert "think" not in lean


def test_default_think_off():
    """Fresh PipelineConfig defaults to think_enabled=False."""
    cfg = PipelineConfig()
    assert cfg.think_enabled is False

"""Tests for the OpenClaw history-preamble adapter.

Real payload fixtures are drawn from the 2026-04-18 30-day Albert
validation run (validation_metrics.db). The adapter must lift the
[Chat messages since your last reply ...] preamble out of the user
content into proper message-level turns and leave only the current
question in the user field.
"""
from __future__ import annotations

import copy

from sieve.openclaw_adapter import (
    adapt_openclaw_payload,
    has_openclaw_preamble,
)


# ── Fixture: real Day-1 Q2 OpenClaw payload ─────────────────────────
#
# Captured verbatim from validation_metrics.db
# (path='openclaw_recall', query_id=2, simulated_day=1).
DAY1_Q2_USER = (
    "[Chat messages since your last reply - for context]\n"
    "User: What's the weather forecast for Bristol this week?\n"
    "Assistant: I don't have access to real-time weather data or "
    "external APIs. For current weather forecasts in Bristol, please "
    "check a weather service or app.\n\n"
    "[Current message - respond to this]\n"
    "User: I need to leave work early today to pick up my daughter "
    "from school. Can you help me draft a quick message to my "
    "manager?"
)


def _make_payload(user_content: str) -> dict:
    return {
        "model": "qwen3:30b-a3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ],
    }


def test_has_openclaw_preamble_detects_marker():
    assert has_openclaw_preamble(DAY1_Q2_USER)
    assert not has_openclaw_preamble("Plain user question")
    assert not has_openclaw_preamble("")


def test_adapter_lifts_history_turns_out_of_user_content():
    payload = _make_payload(DAY1_Q2_USER)
    changed = adapt_openclaw_payload(payload)
    assert changed

    roles = [m["role"] for m in payload["messages"]]
    # system, user (prior), assistant (prior), user (current)
    assert roles == ["system", "user", "assistant", "user"]

    # Prior turn content preserved:
    assert "weather forecast for Bristol" in payload["messages"][1]["content"]
    assert "real-time weather data" in payload["messages"][2]["content"]

    # Final user message holds only the current question, no markers:
    last = payload["messages"][-1]
    assert last["role"] == "user"
    assert "[Chat messages" not in last["content"]
    assert "[Current message" not in last["content"]
    assert "leave work early today" in last["content"]


def test_adapter_noop_on_plain_payload():
    """A vanilla chat payload without the OpenClaw markers is untouched."""
    payload = _make_payload("What's the weather like?")
    snapshot = copy.deepcopy(payload)
    assert adapt_openclaw_payload(payload) is False
    assert payload == snapshot


def test_adapter_noop_on_missing_messages():
    """Malformed payloads (no messages list) survive without error."""
    payload = {"model": "qwen"}
    assert adapt_openclaw_payload(payload) is False


def test_adapter_handles_multi_turn_history():
    """A preamble with many alternating turns is parsed in order."""
    user_content = (
        "[Chat messages since your last reply - for context]\n"
        "User: Q1\nAssistant: A1\n"
        "User: Q2\nAssistant: A2\n"
        "User: Q3\nAssistant: A3\n\n"
        "[Current message - respond to this]\n"
        "User: Q4 current"
    )
    payload = _make_payload(user_content)
    assert adapt_openclaw_payload(payload) is True

    non_system = [m for m in payload["messages"] if m["role"] != "system"]
    # 3 user / 3 assistant history turns + 1 new user = 7 msgs
    assert len(non_system) == 7
    assert non_system[0] == {"role": "user", "content": "Q1"}
    assert non_system[1] == {"role": "assistant", "content": "A1"}
    assert non_system[-1]["content"] == "Q4 current"


def test_adapter_without_current_marker_falls_back_to_last_user():
    """Some OpenClaw variants omit the [Current message ...] marker.
    Fallback: treat the final "User:" line as the current question."""
    user_content = (
        "[Chat messages since your last reply - for context]\n"
        "User: prior\nAssistant: a1\n"
        "User: the new question"
    )
    payload = _make_payload(user_content)
    assert adapt_openclaw_payload(payload) is True
    assert payload["messages"][-1]["content"] == "the new question"
    # One historical exchange should remain as messages:
    hist = [m for m in payload["messages"] if m["role"] != "system"][:-1]
    assert any(m["role"] == "assistant" and m["content"] == "a1" for m in hist)


def test_adapter_leaves_tools_intact():
    """Tools/top-level keys must pass through unmodified."""
    payload = _make_payload(DAY1_Q2_USER)
    payload["tools"] = [{"type": "function", "function": {"name": "foo"}}]
    payload["stream"] = True
    adapt_openclaw_payload(payload)
    assert payload["tools"] == [{"type": "function", "function": {"name": "foo"}}]
    assert payload["stream"] is True

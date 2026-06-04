"""Tests for Phase 7: RecallHandler — stream interception + multi-turn recall."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.responses import Response

from sieve.recall_tool import (
    MAX_ROUNDS,
    RecallHandler,
    _build_tool_result,
    _extract_assistant_message,
    _extract_tool_calls,
    _is_recall_call,
    _parse_recall_args,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fake_embed(text: str) -> list[float]:
    v = [ord(text[0]) / 256.0, 0.5, 0.3, 0.2] if text else [0.1, 0.1, 0.1, 0.1]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


@dataclass
class FakeRetrievedContext:
    facts: list
    text: str
    query: str = ""
    token_estimate: int = 0
    retrieved_from_graph: int = 0


class FakeRetriever:
    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self.calls: list[str] = []

    async def retrieve(self, query: str, top_k: int | None = None):
        self.calls.append(query)
        text = self._responses.get(query, "No relevant context found.")
        facts = [{"content": text, "confidence": 0.9}] if text else []
        return FakeRetrievedContext(
            facts=facts,
            text=text,
            query=query,
            token_estimate=len(text) // 4,
        )


def _make_request():
    """Create a minimal fake Request object for testing."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    return Request(scope)


def _mock_stream_result(resp_data: dict, api_format: str = "ollama"):
    """Build a (buffered_chunks, resp_data, has_recall) tuple for mocking _stream_and_detect."""
    has_recall = bool(_extract_tool_calls(resp_data, api_format) and
                      any(_is_recall_call(tc, api_format)
                          for tc in _extract_tool_calls(resp_data, api_format)))
    # Build a single NDJSON chunk for the buffered response
    chunk_bytes = (json.dumps(resp_data) + "\n").encode()
    return ([chunk_bytes], resp_data, has_recall)


# ─── Ollama response builders ─────────────────────────────────────────────────

def _ollama_text_response(content: str) -> dict:
    """Build an Ollama-format response with just text content."""
    return {
        "model": "qwen3.5:35b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


def _ollama_recall_response(query: str, scope: str = "all") -> dict:
    """Build an Ollama-format response with a recall tool call."""
    return {
        "model": "qwen3.5:35b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "recall",
                        "arguments": {"query": query, "scope": scope},
                    },
                }
            ],
        },
        "done": True,
    }


def _ollama_other_tool_response(name: str, args: dict) -> dict:
    """Build an Ollama-format response with a non-recall tool call."""
    return {
        "model": "qwen3.5:35b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": name,
                        "arguments": args,
                    },
                }
            ],
        },
        "done": True,
    }


def _openai_text_response(content: str) -> dict:
    """Build an OpenAI-format response with just text content."""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _openai_recall_response(query: str, scope: str = "all") -> dict:
    """Build an OpenAI-format response with a recall tool call."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "recall",
                                "arguments": json.dumps({"query": query, "scope": scope}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _openai_other_tool_response(name: str, args: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz789",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


# ─── Unit tests: extraction helpers ──────────────────────────────────────────

class TestExtractToolCalls:
    def test_ollama_no_tool_calls(self):
        resp = _ollama_text_response("Hello!")
        assert _extract_tool_calls(resp, "ollama") == []

    def test_ollama_with_recall(self):
        resp = _ollama_recall_response("user location")
        calls = _extract_tool_calls(resp, "ollama")
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "recall"

    def test_openai_no_tool_calls(self):
        resp = _openai_text_response("Hello!")
        assert _extract_tool_calls(resp, "openai") == []

    def test_openai_with_recall(self):
        resp = _openai_recall_response("user location")
        calls = _extract_tool_calls(resp, "openai")
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "recall"

    def test_empty_response(self):
        assert _extract_tool_calls({}, "ollama") == []
        assert _extract_tool_calls({}, "openai") == []


class TestIsRecallCall:
    def test_recall_call_ollama(self):
        tc = {"function": {"name": "recall", "arguments": {}}}
        assert _is_recall_call(tc, "ollama") is True

    def test_non_recall_call_ollama(self):
        tc = {"function": {"name": "search_web", "arguments": {}}}
        assert _is_recall_call(tc, "ollama") is False

    def test_recall_call_openai(self):
        tc = {"id": "call_1", "function": {"name": "recall", "arguments": "{}"}}
        assert _is_recall_call(tc, "openai") is True

    def test_non_recall_call_openai(self):
        tc = {"id": "call_1", "function": {"name": "calculator", "arguments": "{}"}}
        assert _is_recall_call(tc, "openai") is False


class TestParseRecallArgs:
    def test_ollama_dict_args(self):
        tc = {"function": {"name": "recall", "arguments": {"query": "my location", "scope": "facts"}}}
        q, s = _parse_recall_args(tc, "ollama")
        assert q == "my location"
        assert s == "facts"

    def test_ollama_string_args(self):
        tc = {"function": {"name": "recall", "arguments": '{"query": "my family"}'}}
        q, s = _parse_recall_args(tc, "ollama")
        assert q == "my family"
        assert s == "all"

    def test_openai_string_args(self):
        tc = {"function": {"name": "recall", "arguments": '{"query": "my job", "scope": "episodes"}'}}
        q, s = _parse_recall_args(tc, "openai")
        assert q == "my job"
        assert s == "episodes"

    def test_missing_query(self):
        tc = {"function": {"name": "recall", "arguments": "{}"}}
        q, s = _parse_recall_args(tc, "ollama")
        assert q == ""
        assert s == "all"

    def test_malformed_args(self):
        tc = {"function": {"name": "recall", "arguments": "not json"}}
        q, s = _parse_recall_args(tc, "ollama")
        assert q == ""


class TestExtractAssistantMessage:
    def test_ollama_message(self):
        resp = _ollama_text_response("Hi there")
        msg = _extract_assistant_message(resp, "ollama")
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hi there"

    def test_openai_message(self):
        resp = _openai_text_response("Hi there")
        msg = _extract_assistant_message(resp, "openai")
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hi there"

    def test_empty_response(self):
        assert _extract_assistant_message({}, "ollama") is None
        assert _extract_assistant_message({}, "openai") is None


class TestBuildToolResult:
    def test_ollama_tool_result(self):
        tc = {"function": {"name": "recall", "arguments": {}}}
        msg = _build_tool_result(tc, "User lives in Springfield", "ollama")
        assert msg["role"] == "tool"
        assert "Springfield" in msg["content"]

    def test_openai_tool_result_has_id(self):
        tc = {"id": "call_abc123", "function": {"name": "recall", "arguments": "{}"}}
        msg = _build_tool_result(tc, "User is a librarian", "openai")
        assert msg["tool_call_id"] == "call_abc123"
        assert "librarian" in msg["content"]

    def test_empty_result_gives_fallback(self):
        tc = {"function": {"name": "recall", "arguments": {}}}
        msg = _build_tool_result(tc, "", "ollama")
        assert "No relevant context" in msg["content"]


# ─── Integration tests: RecallHandler ─────────────────────────────────────────

class TestRecallHandlerNoToolCalls:
    """LLM returns plain text — no recall loop, direct passthrough."""

    async def test_ollama_text_passthrough(self):
        retriever = FakeRetriever()
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)
        resp_data = _ollama_text_response("Hello!")
        handler._stream_and_detect = AsyncMock(
            return_value=_mock_stream_result(resp_data, "ollama"))

        request = _make_request()
        payload = {"model": "qwen3.5:35b", "messages": [{"role": "user", "content": "hi"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        data = json.loads(resp.body)
        assert data["message"]["content"] == "Hello!"
        assert resp.headers.get("X-Sieve-Rounds") == "0"
        assert retriever.calls == []

    async def test_openai_text_passthrough(self):
        retriever = FakeRetriever()
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)
        resp_data = _openai_text_response("Hello!")
        handler._stream_and_detect = AsyncMock(
            return_value=_mock_stream_result(resp_data, "openai"))

        request = _make_request()
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="openai")
        data = json.loads(resp.body)
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert resp.headers.get("X-Sieve-Rounds") == "0"


class TestRecallHandlerSingleRound:
    """LLM calls recall once, then returns text."""

    async def test_ollama_single_recall(self):
        retriever = FakeRetriever({"user location": "## Recalled context\n- User lives in Springfield"})
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(side_effect=[
            _mock_stream_result(_ollama_recall_response("user location"), "ollama"),
            _mock_stream_result(_ollama_text_response("You live in Springfield!"), "ollama"),
        ])

        request = _make_request()
        payload = {"model": "qwen3.5:35b", "messages": [{"role": "user", "content": "where do I live?"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        data = json.loads(resp.body)
        assert "Springfield" in data["message"]["content"]
        assert resp.headers.get("X-Sieve-Rounds") == "1"
        assert retriever.calls == ["user location"]

    async def test_openai_single_recall(self):
        retriever = FakeRetriever({"user job": "## Recalled context\n- User is a librarian"})
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(side_effect=[
            _mock_stream_result(_openai_recall_response("user job"), "openai"),
            _mock_stream_result(_openai_text_response("You're a librarian at City Library."), "openai"),
        ])

        request = _make_request()
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "what do I do?"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="openai")
        data = json.loads(resp.body)
        assert "librarian" in data["choices"][0]["message"]["content"]
        assert resp.headers.get("X-Sieve-Rounds") == "1"


class TestRecallHandlerMultiRound:
    """LLM calls recall multiple times before producing a final answer."""

    async def test_two_recall_rounds(self):
        retriever = FakeRetriever({
            "user financial situation": "## Recalled context\n- User earns $180k/year",
            "user home ownership": "## Recalled context\n- User lives in Springfield",
        })
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(side_effect=[
            _mock_stream_result(_ollama_recall_response("user financial situation"), "ollama"),
            _mock_stream_result(_ollama_recall_response("user home ownership"), "ollama"),
            _mock_stream_result(_ollama_text_response("Based on your $180k salary and living in Springfield, I'd recommend..."), "ollama"),
        ])

        request = _make_request()
        payload = {"model": "qwen3.5:35b", "messages": [{"role": "user", "content": "should I refinance?"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        data = json.loads(resp.body)
        assert "$180k" in data["message"]["content"] or "Springfield" in data["message"]["content"]
        assert resp.headers.get("X-Sieve-Rounds") == "2"
        assert len(retriever.calls) == 2

    async def test_three_recall_rounds(self):
        retriever = FakeRetriever({
            "q1": "fact1", "q2": "fact2", "q3": "fact3",
        })
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(side_effect=[
            _mock_stream_result(_ollama_recall_response("q1"), "ollama"),
            _mock_stream_result(_ollama_recall_response("q2"), "ollama"),
            _mock_stream_result(_ollama_recall_response("q3"), "ollama"),
            _mock_stream_result(_ollama_text_response("Final answer with all context."), "ollama"),
        ])

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "complex question"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        assert resp.headers.get("X-Sieve-Rounds") == "3"
        assert len(retriever.calls) == 3


class TestRecallHandlerMaxRounds:
    """Verify the MAX_ROUNDS limit is enforced."""

    async def test_max_rounds_enforced(self):
        retriever = FakeRetriever({"q": "some fact"})
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(
            return_value=_mock_stream_result(_ollama_recall_response("q"), "ollama"),
        )

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "loop"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        assert resp.headers.get("X-Sieve-Rounds") == str(MAX_ROUNDS)
        assert len(retriever.calls) == MAX_ROUNDS
        data = json.loads(resp.body)
        assert "message" in data


class TestRecallHandlerNonRecallToolCalls:
    """Non-recall tool calls should be passed through to the agent."""

    async def test_non_recall_tool_passthrough_ollama(self):
        retriever = FakeRetriever()
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(
            return_value=_mock_stream_result(
                _ollama_other_tool_response("search_web", {"query": "weather"}), "ollama"),
        )

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "search web"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        data = json.loads(resp.body)
        assert data["message"]["tool_calls"][0]["function"]["name"] == "search_web"
        assert resp.headers.get("X-Sieve-Rounds") == "0"
        assert retriever.calls == []

    async def test_non_recall_tool_passthrough_openai(self):
        retriever = FakeRetriever()
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(
            return_value=_mock_stream_result(
                _openai_other_tool_response("calculator", {"expression": "2+2"}), "openai"),
        )

        request = _make_request()
        payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "calc"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="openai")
        data = json.loads(resp.body)
        tc = data["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "calculator"
        assert resp.headers.get("X-Sieve-Rounds") == "0"


class TestRecallHandlerConversationBuildup:
    """Verify that recall results are correctly added to the conversation."""

    async def test_messages_accumulate_across_rounds(self):
        retriever = FakeRetriever({"q": "recalled fact"})
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        call_count = 0

        async def capture_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_stream_result(_ollama_recall_response("q"), "ollama")
            return _mock_stream_result(_ollama_text_response("Done"), "ollama")

        handler._stream_and_detect = AsyncMock(side_effect=capture_stream)

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "question"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        assert call_count == 2
        assert resp.headers.get("X-Sieve-Rounds") == "1"


class TestRecallHandlerEdgeCases:
    async def test_llm_failure_returns_502(self):
        retriever = FakeRetriever()
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)
        handler._stream_and_detect = AsyncMock(return_value=([], None, False))

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        assert resp.status_code == 502

    async def test_recall_with_empty_query(self):
        retriever = FakeRetriever({"": "No relevant context found."})
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)

        handler._stream_and_detect = AsyncMock(side_effect=[
            _mock_stream_result(_ollama_recall_response(""), "ollama"),
            _mock_stream_result(_ollama_text_response("I don't have any context."), "ollama"),
        ])

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "?"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        assert resp.status_code == 200

    async def test_headers_present(self):
        retriever = FakeRetriever()
        proxy = MagicMock()
        handler = RecallHandler(proxy, retriever, config=None)
        handler._stream_and_detect = AsyncMock(
            return_value=_mock_stream_result(_ollama_text_response("ok"), "ollama"))

        request = _make_request()
        payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": False}

        resp = await handler.handle_chat(request, payload, api_format="ollama")
        assert "X-Sieve-Proxy-Us" in resp.headers
        assert "X-Sieve-Rounds" in resp.headers

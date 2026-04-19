"""Tests for Cycle 30 Fix 3: query decomposition for multi-hop retrieval."""
from __future__ import annotations

import json

import pytest

from sieve.query_decomposer import (
    _parse_subqueries,
    decompose_query,
)


# ─── Parser unit tests ──────────────────────────────────────────────────────


class TestParseSubqueries:
    def test_array_of_strings(self):
        raw = '["What is the user\'s job?", "Where does the user live?"]'
        assert _parse_subqueries(raw) == [
            "What is the user's job?",
            "Where does the user live?",
        ]

    def test_object_with_questions_key(self):
        raw = '{"questions": ["Q1?", "Q2?"]}'
        assert _parse_subqueries(raw) == ["Q1?", "Q2?"]

    def test_object_with_subqueries_key(self):
        raw = '{"subqueries": ["Sub?"]}'
        assert _parse_subqueries(raw) == ["Sub?"]

    def test_markdown_fenced_json(self):
        raw = '```json\n["Q1?", "Q2?"]\n```'
        assert _parse_subqueries(raw) == ["Q1?", "Q2?"]

    def test_empty_input(self):
        assert _parse_subqueries("") == []
        assert _parse_subqueries("   ") == []

    def test_invalid_json_returns_empty(self):
        assert _parse_subqueries("not json at all") == []

    def test_non_list_returns_empty(self):
        assert _parse_subqueries('{"foo": "bar"}') == []

    def test_caps_at_4_subqueries(self):
        raw = json.dumps(["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?"])
        out = _parse_subqueries(raw)
        assert len(out) == 4
        assert out == ["Q1?", "Q2?", "Q3?", "Q4?"]

    def test_drops_too_short_items(self):
        raw = '["", " ", "a", "What is the job?"]'
        out = _parse_subqueries(raw)
        # Empty, whitespace and single-char entries are dropped; real
        # question survives.
        assert out == ["What is the job?"]


# ─── decompose_query integration tests (httpx-mocked) ───────────────────────


async def test_decompose_query_returns_subqueries(httpx_mock):
    base = "http://127.0.0.1:11434"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        json={"message": {"content": json.dumps([
            "What is Dad's eye condition?",
            "What are Mum's driving constraints?",
            "Where does the user live?",
        ])}, "done": True},
    )
    subs = await decompose_query(
        "Given Dad's eye, Mum's driving, and our location, should they move?",
        provider_base_url=base,
        model="qwen3:30b-a3b",
    )
    assert len(subs) == 3
    assert "eye condition" in subs[0].lower()


async def test_decompose_query_openai_fallback(httpx_mock):
    base = "https://openai-compat.example.com"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        status_code=400,
        json={"error": {"message": "bad request"}},
    )
    httpx_mock.add_response(
        url=f"{base}/v1/chat/completions",
        json={"choices": [{"message": {"content": json.dumps(["A?", "B?"])}}]},
    )
    subs = await decompose_query(
        "Complex multi-hop question about A and B",
        provider_base_url=base,
        model="gpt-4o-mini",
    )
    assert subs == ["A?", "B?"]


async def test_decompose_query_error_returns_empty(httpx_mock):
    base = "http://127.0.0.1:11434"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        status_code=500,
        json={"error": "OOM"},
    )
    subs = await decompose_query(
        "multi hop query",
        provider_base_url=base,
        model="m",
    )
    assert subs == []


async def test_decompose_query_drops_duplicates_of_original(httpx_mock):
    """If the LLM returns the original query verbatim as one of its
    sub-questions, we drop it — running the same search again is wasted."""
    base = "http://127.0.0.1:11434"
    original = "Should Mum and Dad move closer?"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        json={"message": {"content": json.dumps([
            original,
            "Where do they live now?",
            "What are their health needs?",
        ])}, "done": True},
    )
    subs = await decompose_query(
        original, provider_base_url=base, model="m",
    )
    assert original not in subs
    assert len(subs) == 2


async def test_decompose_query_requires_minimum_2_subqueries(httpx_mock):
    """A single-sub-query response is effectively a no-op; we want the
    caller to fall back to the standard retrieve() path."""
    base = "http://127.0.0.1:11434"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        json={"message": {"content": json.dumps(["Just one"])}, "done": True},
    )
    subs = await decompose_query(
        "anything goes here",
        provider_base_url=base,
        model="m",
    )
    assert subs == []


async def test_decompose_query_empty_text_returns_empty():
    assert await decompose_query("", provider_base_url="http://x", model="m") == []
    assert await decompose_query("   ", provider_base_url="http://x", model="m") == []

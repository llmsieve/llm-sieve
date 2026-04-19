"""Tests for S2 writer routing — the feature that removes the separate
CPU-pinned qwen3.5:2b writer and instead sends Stage-2 extraction calls
to the user's configured main model.

Two user-visible behaviors this suite pins down:
  1. When WriterConfig.model == "auto", MemoryWriter resolves it to
     provider.default_model so the proxy uses a single model for both
     main inference and S2 fact extraction (no extra model load, no
     VRAM contention).
  2. S2 survives across Ollama and OpenAI-compatible endpoints. Ollama
     needs `format: "json"`; OpenAI-style servers reject that field with
     400/422 and want `response_format: {"type": "json_object"}` instead.
     The writer tries Ollama format first (cheap happy path for local
     users) and falls back to OpenAI format on a 4xx.
"""

from __future__ import annotations

import json

import httpx
import pytest

import sieve.writer as writer_mod
from sieve.config import (
    AblationConfig,
    EmbeddingsConfig,
    ListenConfig,
    ProviderConfig,
    RecallConfig,
    SecurityConfig,
    StoreConfig,
    WriterConfig,
)
from sieve.writer import extract_facts_s2, resolve_writer_model, summarize_episode


# ─── auto → default_model resolution ────────────────────────────────────────


def test_resolve_writer_model_auto_uses_provider_default():
    """writer.model=='auto' must resolve to provider.default_model so
    the S2 writer and the user's main query hit the same Ollama/cloud
    model."""
    cfg = RecallConfig(
        provider=ProviderConfig(default_model="qwen3:30b-a3b"),
        writer=WriterConfig(model="auto"),
    )
    assert resolve_writer_model(cfg) == "qwen3:30b-a3b"


def test_resolve_writer_model_explicit_name_preserved():
    """Explicit writer.model override still wins so advanced users
    can keep a separate CPU-pinned writer if they want."""
    cfg = RecallConfig(
        provider=ProviderConfig(default_model="qwen3:30b-a3b"),
        writer=WriterConfig(model="recall-writer-cpu"),
    )
    assert resolve_writer_model(cfg) == "recall-writer-cpu"


# ─── Protocol fallback: Ollama first, OpenAI on 4xx ─────────────────────────


def _mock_ollama_ok_response(httpx_mock, base_url: str, *, content: str):
    """Simulate a successful Ollama /api/chat response."""
    httpx_mock.add_response(
        url=f"{base_url}/api/chat",
        json={"message": {"content": content}, "done": True},
    )


def _mock_openai_ok_response(httpx_mock, base_url: str, *, content: str):
    """Simulate a successful OpenAI /v1/chat/completions response."""
    httpx_mock.add_response(
        url=f"{base_url}/v1/chat/completions",
        json={"choices": [{"message": {"content": content}}]},
    )


async def test_s2_uses_ollama_format_first(httpx_mock):
    """Happy path for local Ollama users: the first request should use
    Ollama's /api/chat with top-level format='json' and think=False."""
    base = "http://127.0.0.1:11434"
    _mock_ollama_ok_response(
        httpx_mock, base,
        content=json.dumps({
            "facts": [
                {
                    "content": "User lives in Bristol.",
                    "entities": ["user", "Bristol"],
                    "fact_type": "objective",
                    "confidence": 0.9,
                },
            ],
        }),
    )

    facts = await extract_facts_s2(
        text="I live in Bristol.",
        provider_base_url=base,
        model="qwen3:30b-a3b",
        fallback_model="qwen3:30b-a3b",
        owner_name="Albert",
    )

    assert len(facts) == 1
    assert facts[0].content == "User lives in Bristol."

    sent = httpx_mock.get_request()
    assert sent.url == f"{base}/api/chat"
    body = json.loads(sent.content)
    assert body["format"] == "json"
    assert body["think"] is False
    assert body["stream"] is False
    assert body["model"] == "qwen3:30b-a3b"


async def test_s2_falls_back_to_openai_format_on_400(httpx_mock):
    """When the upstream rejects the Ollama-specific format field
    (OpenAI-compatible servers return 400/422), we retry the same
    prompt against /v1/chat/completions with response_format set."""
    base = "https://openai-compat.example.com"

    # Ollama attempt rejected by the server.
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        status_code=400,
        json={"error": {"message": "unknown field 'format'"}},
    )
    # Fallback to OpenAI-style.
    _mock_openai_ok_response(
        httpx_mock, base,
        content=json.dumps({
            "facts": [
                {
                    "content": "User works as a civil engineer.",
                    "entities": ["user", "civil engineer"],
                    "fact_type": "objective",
                    "confidence": 0.8,
                },
            ],
        }),
    )

    facts = await extract_facts_s2(
        text="I work as a civil engineer.",
        provider_base_url=base,
        model="gpt-4o-mini",
        fallback_model="gpt-4o-mini",
        owner_name="Albert",
    )

    assert len(facts) == 1
    assert "civil engineer" in facts[0].content

    requests = httpx_mock.get_requests()
    assert requests[0].url == f"{base}/api/chat"  # first try
    assert requests[1].url == f"{base}/v1/chat/completions"  # fallback

    fallback_body = json.loads(requests[1].content)
    assert fallback_body["model"] == "gpt-4o-mini"
    assert fallback_body["response_format"] == {"type": "json_object"}
    assert "format" not in fallback_body
    assert "think" not in fallback_body


async def test_s2_falls_back_on_422(httpx_mock):
    """Some OpenAI-compatible servers (vLLM, LM Studio) return 422
    rather than 400. The fallback must trigger on the full 4xx range."""
    base = "https://vllm.example.com"

    httpx_mock.add_response(
        url=f"{base}/api/chat",
        status_code=422,
        json={"error": "Unprocessable Entity"},
    )
    _mock_openai_ok_response(
        httpx_mock, base,
        content=json.dumps({"facts": []}),
    )

    facts = await extract_facts_s2(
        text="no facts here",
        provider_base_url=base,
        model="llama-3-8b",
        fallback_model="llama-3-8b",
        owner_name="",
    )
    assert facts == []


# ─── Episode summary (Cycle 30 Fix 2) ──────────────────────────────────────


async def test_summarize_episode_returns_llm_sentence(httpx_mock):
    """Happy path: main-model Ollama call returns the one-sentence summary."""
    base = "http://127.0.0.1:11434"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        json={"message": {
            "content": "User is considering a mortgage and leaning toward locking in."
        }, "done": True},
    )
    out = await summarize_episode(
        user_text="We spoke with the broker about fixed vs variable.",
        assistant_text="Locking in at 5% gives certainty for 5 years.",
        provider_base_url=base,
        model="qwen3:30b-a3b",
    )
    assert "mortgage" in out
    sent = httpx_mock.get_request()
    body = json.loads(sent.content)
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0
    # System prompt must instruct a single sentence
    sys_msg = body["messages"][0]
    assert sys_msg["role"] == "system"
    assert "ONE sentence" in sys_msg["content"]


async def test_summarize_episode_empty_on_llm_failure(httpx_mock):
    """A 5xx or network error must fall open with an empty string so the
    caller can use the legacy truncation."""
    base = "http://127.0.0.1:11434"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        status_code=500,
        json={"error": "upstream OOM"},
    )
    out = await summarize_episode(
        user_text="hello",
        assistant_text="hi",
        provider_base_url=base,
        model="qwen3:30b-a3b",
    )
    assert out == ""


async def test_summarize_episode_falls_back_to_openai(httpx_mock):
    """When Ollama endpoint 4xx's, we retry on /v1/chat/completions."""
    base = "https://openai-compat.example.com"
    httpx_mock.add_response(
        url=f"{base}/api/chat",
        status_code=404,
        json={"error": {"message": "not found"}},
    )
    httpx_mock.add_response(
        url=f"{base}/v1/chat/completions",
        json={"choices": [{"message": {
            "content": "User plans to finalise the mortgage next week.",
        }}]},
    )
    out = await summarize_episode(
        user_text="Let's finalise next week.",
        assistant_text="Agreed.",
        provider_base_url=base,
        model="gpt-4o-mini",
    )
    assert "mortgage" in out


async def test_summarize_episode_empty_user_text_noop(httpx_mock):
    """Empty user_text returns "" without any network call."""
    out = await summarize_episode(
        user_text="",
        assistant_text="sure",
        provider_base_url="http://127.0.0.1:11434",
        model="m",
    )
    assert out == ""
    # No requests should have been made
    assert httpx_mock.get_requests() == []


async def test_s2_surfaces_5xx_instead_of_falling_back(httpx_mock):
    """5xx isn't a protocol problem — it's an upstream outage. We
    should not mask it by routing to a different endpoint; let the
    caller fall back to regex-only extraction."""
    base = "http://127.0.0.1:11434"

    # _s2_call_with_retries returns None on any exception (5xx raises)
    # rather than retrying the same endpoint, so the primary model takes
    # one call, the fallback model takes one more, total = 2.
    for _ in range(2):
        httpx_mock.add_response(
            url=f"{base}/api/chat",
            status_code=500,
            json={"error": "upstream OOM"},
        )

    facts = await extract_facts_s2(
        text="I live in Bristol.",
        provider_base_url=base,
        model="qwen3:30b-a3b",
        fallback_model="qwen3:30b-a3b",
        owner_name="",
    )
    assert facts == []

    # Every request must have been to /api/chat — we didn't spuriously
    # switch protocols on a 5xx.
    for req in httpx_mock.get_requests():
        assert req.url == f"{base}/api/chat"

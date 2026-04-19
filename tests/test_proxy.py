"""Tests for Phase 1: Skeleton proxy passthrough + streaming."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from sieve.config import (
    EmbeddingsConfig,
    ListenConfig,
    ProviderConfig,
    RecallConfig,
    SecurityConfig,
    StoreConfig,
)
from sieve.main import create_app

TEST_AUTH_TOKEN = "test-token-for-proxy-tests"


@pytest.fixture
def mock_config(httpx_mock, tmp_path):
    """Config pointing at a mock upstream with an isolated tmp store.

    Path isolation is required so create_app doesn't touch the
    developer's real ~/.sieve/memory.db — which also means we won't
    trip the EmbeddingDimensionMismatchError guard on a store that was
    built with a different embedding backend.
    """
    return RecallConfig(
        listen=ListenConfig(host="127.0.0.1", port=11435),
        provider=ProviderConfig(base_url="https://upstream"),
        store=StoreConfig(path=str(tmp_path / "proxy_test.db")),
        # Use the Ollama embedding path with a mocked upstream so the
        # test doesn't trigger a 50MB FastEmbed download inside pytest.
        embeddings=EmbeddingsConfig(
            provider="ollama",
            ollama_url="https://upstream",
            ollama_model="nomic-embed-text",
        ),
        security=SecurityConfig(auth_token=TEST_AUTH_TOKEN),
    )


@pytest.fixture
def app(mock_config):
    return create_app(mock_config)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _auth_headers():
    return {"X-Sieve-Token": TEST_AUTH_TOKEN}


# --- Health endpoint ---


def test_health(client):
    resp = client.get("/sieve/health", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# --- Passthrough: non-streaming ---


def test_passthrough_get_tags(client, httpx_mock):
    """GET /api/tags should passthrough to upstream."""
    upstream_body = {"models": [{"name": "qwen3.5:35b"}]}
    httpx_mock.add_response(
        url="https://upstream/api/tags",
        json=upstream_body,
    )

    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert resp.json() == upstream_body


def test_passthrough_post_embeddings(client, httpx_mock):
    """POST /api/embeddings should passthrough."""
    upstream_body = {"embedding": [0.1, 0.2, 0.3]}
    httpx_mock.add_response(
        url="https://upstream/api/embeddings",
        json=upstream_body,
    )

    resp = client.post("/api/embeddings", json={"model": "nomic-embed-text", "prompt": "hello"})
    assert resp.status_code == 200
    assert resp.json() == upstream_body


def test_passthrough_preserves_status_code(client, httpx_mock):
    """Upstream error codes should be forwarded."""
    httpx_mock.add_response(
        url="https://upstream/api/show",
        status_code=404,
        json={"error": "model not found"},
    )

    resp = client.post("/api/show", json={"name": "nonexistent"})
    assert resp.status_code == 404


def test_passthrough_adds_timing_header(client, httpx_mock):
    """Response should include X-Sieve-Proxy-Us timing header."""
    httpx_mock.add_response(url="https://upstream/api/tags", json={"models": []})

    resp = client.get("/api/tags")
    assert "x-sieve-proxy-us" in resp.headers
    elapsed = int(resp.headers["x-sieve-proxy-us"])
    assert elapsed >= 0


# --- Passthrough: streaming ---


@pytest.mark.skip(reason="Pipeline intercepts /api/chat — mock needs full pipeline setup. Streaming tested via live e2e.")
def test_streaming_ndjson_ollama(client, httpx_mock):
    """POST /api/chat with streaming NDJSON (Ollama format)."""
    chunks = [
        json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False}),
        json.dumps({"message": {"role": "assistant", "content": " world"}, "done": False}),
        json.dumps({"done": True, "total_duration": 1000}),
    ]
    body = "\n".join(chunks)

    httpx_mock.add_response(
        url="https://upstream/api/chat",
        content=body.encode(),
        headers={"content-type": "application/x-ndjson"},
    )

    resp = client.post(
        "/api/chat",
        json={"model": "qwen3.5:35b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    # Parse the streamed NDJSON lines
    lines = [line for line in resp.text.strip().split("\n") if line]
    assert len(lines) == 3
    last = json.loads(lines[-1])
    assert last["done"] is True


@pytest.mark.skip(reason="Pipeline intercepts /v1/chat/completions — mock needs full pipeline setup. Streaming tested via live e2e.")
def test_streaming_sse_openai(client, httpx_mock):
    """POST /v1/chat/completions with streaming SSE (OpenAI format)."""
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    body = "".join(chunks)

    httpx_mock.add_response(
        url="https://upstream/v1/chat/completions",
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
    )

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5:35b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "Hi" in resp.text
    assert "[DONE]" in resp.text


# --- Non-streaming chat ---


def test_non_streaming_chat(client, httpx_mock):
    """POST /api/chat with stream=false returns full JSON response."""
    upstream_body = {
        "model": "qwen3.5:35b",
        "message": {"role": "assistant", "content": "Hello!"},
        "done": True,
    }
    httpx_mock.add_response(
        url="https://upstream/api/chat",
        json=upstream_body,
        headers={"content-type": "application/json"},
    )
    # Classifier L1 may embed the user turn for tool selection.
    httpx_mock.add_response(
        url="https://upstream/api/embeddings",
        json={"embedding": [0.0] * 768},
        headers={"content-type": "application/json"},
        is_optional=True,
    )

    resp = client.post(
        "/api/chat",
        json={
            "model": "qwen3.5:35b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["message"]["content"] == "Hello!"


# --- Query string forwarding ---


def test_query_params_forwarded(client, httpx_mock):
    """Query parameters should be forwarded to upstream."""
    httpx_mock.add_response(url="https://upstream/v1/models?limit=5", json={"data": []})

    resp = client.get("/v1/models?limit=5")
    assert resp.status_code == 200


# --- Upstream failure ---


def test_upstream_connection_error(client, httpx_mock):
    """Connection error to upstream should return 502."""
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))

    resp = client.get("/api/tags")
    assert resp.status_code == 502
    assert "error" in resp.json()

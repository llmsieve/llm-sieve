"""Tests for the embedding client against a mocked Ollama endpoint."""

from __future__ import annotations

import pytest

from sieve.config import RecallConfig, ProviderConfig, StoreConfig, EmbeddingsConfig
from sieve.embeddings import EmbeddingClient


@pytest.fixture
def config():
    # These tests exercise the Ollama embedding path against a mocked
    # /api/embeddings endpoint. Force provider=ollama so we don't load
    # FastEmbed (the new shipping default).
    return RecallConfig(
        provider=ProviderConfig(base_url="https://test-ollama"),
        store=StoreConfig(embedding_model="nomic-embed-text", embedding_dimensions=4),
        embeddings=EmbeddingsConfig(provider="ollama"),
    )


@pytest.fixture
def embedding_client(config):
    return EmbeddingClient(config)


async def test_embed_single(embedding_client, httpx_mock):
    """Single text embedding should return a float vector."""
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        json={"embedding": [0.1, 0.2, 0.3, 0.4]},
    )

    await embedding_client.start()
    result = await embedding_client.embed("User lives in Springfield")
    await embedding_client.stop()

    # Embedding client now L2-normalizes the output
    assert len(result) == 4
    norm = sum(x * x for x in result) ** 0.5
    assert abs(norm - 1.0) < 1e-6, f"Expected unit vector, got norm={norm}"


async def test_embed_sends_correct_payload(embedding_client, httpx_mock):
    """Should send model name and prompt to Ollama."""
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        json={"embedding": [0.1, 0.2, 0.3, 0.4]},
    )

    await embedding_client.start()
    await embedding_client.embed("test prompt")
    await embedding_client.stop()

    request = httpx_mock.get_request()
    import json
    body = json.loads(request.content)
    assert body["model"] == "nomic-embed-text"
    assert body["prompt"] == "test prompt"


async def test_embed_batch(embedding_client, httpx_mock):
    """Batch embedding should return one vector per input text."""
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        json={"embedding": [1.0, 0.0, 0.0, 0.0]},
    )
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        json={"embedding": [0.0, 1.0, 0.0, 0.0]},
    )

    await embedding_client.start()
    results = await embedding_client.embed_batch(["text one", "text two"])
    await embedding_client.stop()

    assert len(results) == 2
    # Already unit vectors, normalization is a no-op
    assert results[0] == [1.0, 0.0, 0.0, 0.0]
    assert results[1] == [0.0, 1.0, 0.0, 0.0]


async def test_embed_error_propagates(embedding_client, httpx_mock):
    """HTTP errors from Ollama should propagate."""
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        status_code=404,
        json={"error": "model not found"},
    )

    await embedding_client.start()
    with pytest.raises(Exception):
        await embedding_client.embed("test")
    await embedding_client.stop()


async def test_embed_retries_on_500(embedding_client, httpx_mock, monkeypatch):
    """Transient 5xx responses must be retried (exponential backoff).

    The live embedder (recall-embed-cpu) intermittently returns 500s;
    without retry, each failure poisons retrieval for that query."""
    # Skip the real sleeps so the test stays fast.
    import sieve.embeddings_provider as embeddings_provider_mod
    monkeypatch.setattr(
        embeddings_provider_mod, "_OLLAMA_RETRY_DELAYS_S", (0.0, 0.0, 0.0)
    )

    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        status_code=500, json={"error": "oom"},
    )
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        status_code=500, json={"error": "oom"},
    )
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        json={"embedding": [0.1, 0.2, 0.3, 0.4]},
    )

    await embedding_client.start()
    result = await embedding_client.embed("flaky call")
    await embedding_client.stop()
    assert len(result) == 4


async def test_embed_gives_up_after_all_retries(embedding_client, httpx_mock, monkeypatch):
    """After exhausting retries, the final error must be re-raised so
    callers can fall back to keyword / graph retrieval."""
    import sieve.embeddings_provider as embeddings_provider_mod
    monkeypatch.setattr(
        embeddings_provider_mod, "_OLLAMA_RETRY_DELAYS_S", (0.0, 0.0, 0.0)
    )

    for _ in range(4):  # initial attempt + 3 retries
        httpx_mock.add_response(
            url="https://test-ollama/api/embeddings",
            status_code=500, json={"error": "perma-down"},
        )

    await embedding_client.start()
    with pytest.raises(Exception):
        await embedding_client.embed("always fails")
    await embedding_client.stop()

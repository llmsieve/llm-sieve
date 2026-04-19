"""Tests for EmbeddingService — the provider abstraction that lets Recall
use either built-in FastEmbed (default, self-contained) or Ollama
(opt-in, for users who prefer Ollama's embedding models)."""

from __future__ import annotations

import pytest

from sieve.embeddings_provider import EmbeddingService


# ── FastEmbed provider (self-contained default) ────────────────────────────


def test_fastembed_provider_dimension_is_384():
    """BAAI/bge-small-en-v1.5 is a 384-dim model. The service must
    advertise this so the store can size its vec0 virtual table correctly
    before any embedding is generated."""
    svc = EmbeddingService(provider="fastembed")
    assert svc.dimension == 384


async def test_fastembed_embed_returns_float_list_of_correct_dim():
    """embed() must return a python list[float] of the advertised
    dimension. FastEmbed's raw output is numpy.ndarray[float32]; callers
    (store serialization, cosine similarity) need plain list[float]."""
    svc = EmbeddingService(provider="fastembed")
    vec = await svc.embed("test embedding")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec[:5])


async def test_fastembed_embed_produces_unit_vector():
    """Cosine similarity assumes unit vectors. L2 norm must be ~1.0."""
    svc = EmbeddingService(provider="fastembed")
    vec = await svc.embed("the quick brown fox")
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-3, f"expected unit vector, got norm={norm}"


async def test_fastembed_embed_batch_returns_one_vec_per_text():
    svc = EmbeddingService(provider="fastembed")
    vecs = await svc.embed_batch(["alpha", "beta", "gamma"])
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


async def test_fastembed_embed_is_deterministic_for_same_text():
    """Same text in → same vector out. Required for cache correctness
    and reproducible retrieval."""
    svc = EmbeddingService(provider="fastembed")
    v1 = await svc.embed("hello world")
    v2 = await svc.embed("hello world")
    assert v1 == v2


# ── Ollama provider (opt-in, legacy) ───────────────────────────────────────


async def test_ollama_provider_dimension_defaults_768():
    """nomic-embed-text family is 768-dim. When provider=ollama the
    service must still advertise a dimension up-front so the store can
    size the vec table."""
    svc = EmbeddingService(
        provider="ollama",
        ollama_url="https://test-ollama",
        ollama_model="nomic-embed-text-v2-moe",
    )
    assert svc.dimension == 768


async def test_ollama_provider_hits_api_embeddings_endpoint(httpx_mock):
    """Ollama path must POST to /api/embeddings with {model, prompt}
    and return a normalized list[float]."""
    httpx_mock.add_response(
        url="https://test-ollama/api/embeddings",
        json={"embedding": [0.0, 3.0, 0.0, 4.0]},  # norm = 5 → /5
    )
    svc = EmbeddingService(
        provider="ollama",
        ollama_url="https://test-ollama",
        ollama_model="nomic-embed-text",
    )
    await svc.start()
    try:
        vec = await svc.embed("hello")
    finally:
        await svc.stop()

    # L2-normalized [0, 3, 0, 4] is [0, 0.6, 0, 0.8].
    assert len(vec) == 4
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6

    request = httpx_mock.get_request()
    import json
    body = json.loads(request.content)
    assert body["model"] == "nomic-embed-text"
    assert body["prompt"] == "hello"


# ── Provider validation ────────────────────────────────────────────────────


def test_unknown_provider_raises():
    """An unknown provider string is a config error — fail fast at
    construction rather than at first embed() call."""
    with pytest.raises(Exception):
        EmbeddingService(provider="magic-embed")

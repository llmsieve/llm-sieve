"""Embedding client used across Recall.

This module is a thin wrapper over ``EmbeddingService``
(src/embeddings_provider.py) that preserves the historical public
surface used throughout the codebase:

    client = EmbeddingClient(config)
    await client.start()
    vec = await client.embed("...")
    await client.stop()

The service it delegates to picks a backend from
``config.embeddings.provider`` — ``fastembed`` by default (self-contained,
384-dim, no external dependency) or ``ollama`` (legacy, 768-dim via
/api/embeddings). A small in-process LRU cache shields repeated lookups
of the same text during a single session regardless of backend.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from sieve.config import RecallConfig
from sieve.embeddings_provider import (
    FASTEMBED_DEFAULT_DIM,
    OLLAMA_DEFAULT_DIM,
    EmbeddingProvider,
    EmbeddingService,
)

logger = logging.getLogger("recall.embeddings")

_EMBED_CACHE_MAX = 512

# Re-exported for tests that monkeypatch the retry schedule on the Ollama
# path. Kept at module level to preserve the legacy test interface
# (tests/test_embeddings.py references src.embeddings._RETRY_DELAYS_S).
from sieve.embeddings_provider import _OLLAMA_RETRY_DELAYS_S as _RETRY_DELAYS_S  # noqa: E402,F401


class EmbeddingClient:
    """Caching facade over EmbeddingService.

    Kept as a distinct class from ``EmbeddingService`` because:
      * existing call sites (main.py, writer, retriever, classifier)
        pass ``client.embed`` as a bound method — that contract is
        stable here;
      * the per-text LRU cache is specific to proxy-side usage and is
        orthogonal to the backend.
    """

    def __init__(self, config: RecallConfig):
        self._config = config
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

        provider_name = config.embeddings.provider
        if provider_name == EmbeddingProvider.FASTEMBED.value:
            self._service = EmbeddingService(provider="fastembed")
        else:
            # Ollama backend. URL/model default to the values already
            # configured for the main provider + store so that existing
            # existing recall.yaml-style configs keep working unchanged.
            self._service = EmbeddingService(
                provider="ollama",
                ollama_url=config.embeddings.ollama_url or config.provider.base_url,
                ollama_model=(
                    config.embeddings.ollama_model or config.store.embedding_model
                ),
                ollama_dimension=config.store.embedding_dimensions or OLLAMA_DEFAULT_DIM,
            )

    @property
    def dimension(self) -> int:
        """Native dimension of the active embedding backend.

        Exposed so the store can size its vec0 virtual table from the
        provider rather than from a hand-maintained config field.
        """
        return self._service.dimension

    @property
    def provider_name(self) -> str:
        return self._service.provider.value

    async def start(self) -> None:
        await self._service.start()

    async def stop(self) -> None:
        await self._service.stop()

    async def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        vec = await self._service.embed(text)
        self._cache[text] = vec
        if len(self._cache) > _EMBED_CACHE_MAX:
            self._cache.popitem(last=False)
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Respect cache for hits; delegate misses to the service in a
        # single batch call (important for FastEmbed throughput).
        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                results.append(cached)
            else:
                results.append(None)
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            fresh = await self._service.embed_batch(miss_texts)
            for idx, text, vec in zip(miss_indices, miss_texts, fresh):
                self._cache[text] = vec
                if len(self._cache) > _EMBED_CACHE_MAX:
                    self._cache.popitem(last=False)
                results[idx] = vec

        assert all(r is not None for r in results)
        return [r for r in results]  # type: ignore[misc]

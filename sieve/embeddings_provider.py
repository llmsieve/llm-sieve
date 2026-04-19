"""Embedding provider abstraction.

Recall supports two embedding backends:

* ``fastembed`` (default) — ONNX Runtime, self-contained, no external
  service. Uses BAAI/bge-small-en-v1.5 (384-dim, ~50MB). Model
  auto-downloads on first use; pre-download during proxy init keeps the
  first query fast. This is the shipping default so a fresh
  ``pip install`` has zero Ollama dependency.
* ``ollama`` — calls the configured Ollama instance's /api/embeddings
  endpoint. Retained for users who already operate Ollama-based
  embedding pipelines (e.g. nomic-embed-text-v2-moe, 768-dim).

Callers only see the ``EmbeddingService`` interface (``dimension``,
``embed``, ``embed_batch``). The store sizes its vec0 table from
``dimension`` at schema creation time.
"""

from __future__ import annotations

import asyncio
import logging
import math
from enum import Enum

import httpx

logger = logging.getLogger("recall.embeddings_provider")

# FastEmbed default. Chosen because:
#   * 384-dim — half the storage of 768-dim nomic variants
#   * ~50MB ONNX — small enough to download during `sieve init`
#   * MIT-licensed, battle-tested (Qdrant, LangChain, LlamaIndex)
#   * Quality benchmarks competitive with nomic-embed for short-text retrieval
FASTEMBED_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_DEFAULT_DIM = 384

# Ollama embed default. nomic-embed-text-v2-moe returns 768-dim vectors.
OLLAMA_DEFAULT_DIM = 768

# Transient-failure retry schedule (Ollama path only). FastEmbed runs
# in-process with no network so no retry is needed there.
_OLLAMA_RETRY_DELAYS_S = (1.0, 2.0, 4.0)

# Cycle 30 Fix 5: cross-encoder reranker defaults. MiniLM-L-6-v2 is the
# smallest option supported by FastEmbed (~80MB), fast on CPU (~20-50ms
# for 10 candidates), and empirically strong for short-text re-ranking.
FASTEMBED_RERANK_DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class EmbeddingProvider(str, Enum):
    FASTEMBED = "fastembed"
    OLLAMA = "ollama"


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class EmbeddingService:
    """Unified embedding interface across providers."""

    def __init__(
        self,
        provider: str = "fastembed",
        *,
        fastembed_model: str = FASTEMBED_DEFAULT_MODEL,
        ollama_url: str | None = None,
        ollama_model: str | None = None,
        ollama_dimension: int = OLLAMA_DEFAULT_DIM,
    ):
        try:
            self.provider = EmbeddingProvider(provider)
        except ValueError as exc:
            raise ValueError(
                f"Unknown embeddings.provider={provider!r}. "
                f"Valid: {[p.value for p in EmbeddingProvider]}"
            ) from exc

        if self.provider is EmbeddingProvider.FASTEMBED:
            from fastembed import TextEmbedding
            self._fastembed_model_name = fastembed_model
            # Instantiating TextEmbedding triggers the ONNX download on
            # first construction of a given model. Callers who want to
            # pre-warm (e.g. during proxy startup) simply construct the
            # service early.
            self._fastembed = TextEmbedding(fastembed_model)
            self.dimension = FASTEMBED_DEFAULT_DIM
            self._ollama_client: httpx.AsyncClient | None = None
        else:
            if not ollama_url or not ollama_model:
                raise ValueError(
                    "provider=ollama requires ollama_url and ollama_model"
                )
            self._ollama_url = ollama_url.rstrip("/")
            self._ollama_model = ollama_model
            self.dimension = int(ollama_dimension)
            self._ollama_client: httpx.AsyncClient | None = None

    # ── Lifecycle (Ollama only; FastEmbed is in-process) ──────────────────

    async def start(self) -> None:
        if self.provider is EmbeddingProvider.OLLAMA:
            self._ollama_client = httpx.AsyncClient(
                base_url=self._ollama_url,
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
            )

    async def stop(self) -> None:
        if self._ollama_client is not None:
            await self._ollama_client.aclose()
            self._ollama_client = None

    # ── Embedding API ─────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        if self.provider is EmbeddingProvider.FASTEMBED:
            return self._fastembed_embed_one(text)
        return await self._ollama_embed_one(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider is EmbeddingProvider.FASTEMBED:
            # Real batching — FastEmbed's model.embed() consumes an
            # iterable and is substantially faster than per-text calls.
            results = list(self._fastembed.embed(texts))
            return [_l2_normalize([float(x) for x in vec]) for vec in results]
        return [await self._ollama_embed_one(t) for t in texts]

    # ── Providers ─────────────────────────────────────────────────────────

    def _fastembed_embed_one(self, text: str) -> list[float]:
        vec = list(self._fastembed.embed([text]))[0]
        return _l2_normalize([float(x) for x in vec])

    async def _ollama_embed_one(self, text: str) -> list[float]:
        assert self._ollama_client is not None, (
            "EmbeddingService(provider=ollama) not started"
        )
        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0,) + _OLLAMA_RETRY_DELAYS_S):
            if delay:
                await asyncio.sleep(delay)
            try:
                resp = await self._ollama_client.post(
                    "/api/embeddings",
                    json={"model": self._ollama_model, "prompt": text},
                )
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning(
                    "ollama embedder transport error (attempt %d/%d): %s",
                    attempt + 1, len(_OLLAMA_RETRY_DELAYS_S) + 1, exc,
                )
                continue
            if 400 <= resp.status_code < 500:
                resp.raise_for_status()
            if resp.status_code >= 500:
                last_exc = httpx.HTTPStatusError(
                    f"embedder returned {resp.status_code}",
                    request=resp.request, response=resp,
                )
                logger.warning(
                    "ollama embedder 5xx (attempt %d/%d): %d",
                    attempt + 1, len(_OLLAMA_RETRY_DELAYS_S) + 1, resp.status_code,
                )
                continue
            resp.raise_for_status()
            vec = _l2_normalize(resp.json()["embedding"])
            if attempt > 0:
                logger.info("ollama embedder: succeeded on retry attempt %d", attempt)
            return vec
        assert last_exc is not None
        raise last_exc


class RerankerService:
    """Cycle 30 Fix 5: cross-encoder re-ranker over vector search candidates.

    FastEmbed's TextCrossEncoder runs an ONNX cross-encoder model in-process
    on CPU. Typical MiniLM-L-6-v2 takes ~20-50ms for 10 (query, passage)
    pairs, negligible next to the model call that follows.

    Fail-open: if the model fails to load or a rerank call raises, the
    service returns the candidates unchanged so retrieval degrades to pure
    vector search rather than crashing the request.
    """

    def __init__(self, model_name: str = FASTEMBED_RERANK_DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._reranker: Any = None  # lazy init in load()
        self._available = False

    def load(self) -> bool:
        """Construct the cross-encoder. Triggers the ONNX download on first use.

        Returns True if loading succeeded. False means the reranker is
        disabled for this process — callers should see rerank() become a
        no-op.
        """
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            self._reranker = TextCrossEncoder(self._model_name)
            self._available = True
            logger.info("Reranker loaded: %s", self._model_name)
            return True
        except Exception as exc:
            logger.warning("Reranker load failed (%s) — disabled", exc)
            self._reranker = None
            self._available = False
            return False

    @property
    def available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        candidates: list[str],
    ) -> list[float] | None:
        """Score each candidate against *query*. Returns a list of floats
        aligned with *candidates* (higher = more relevant), or None if the
        reranker is unavailable or the call raises.
        """
        if not self._available or self._reranker is None:
            return None
        if not candidates:
            return []
        try:
            scores = list(self._reranker.rerank_pairs(
                [(query, c) for c in candidates],
            ))
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("Reranker call failed: %s — skipping rerank", exc)
            return None

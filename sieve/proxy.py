"""Transparent async HTTP proxy for LLM providers."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("recall.proxy")

# Headers that hop-by-hop and shouldn't be forwarded
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
})

# Timeout: 5min read (LLM generation can be slow), 10s connect
CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


class ProxyClient:
    """Manages a shared httpx.AsyncClient for forwarding requests."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=CLIENT_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        assert self._client is not None, "ProxyClient not started"
        return self._client


def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter out hop-by-hop headers."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


async def forward_request(
    request: Request,
    proxy_client: ProxyClient,
) -> Response | StreamingResponse:
    """Forward a request to the upstream LLM provider and return the response.

    Handles both streaming and non-streaming responses transparently.
    """
    start = time.perf_counter_ns()

    path = request.url.path
    query = str(request.url.query)
    url = path if not query else f"{path}?{query}"

    body = await request.body()
    headers = _forward_headers(dict(request.headers))

    upstream_req = proxy_client.client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )

    upstream_resp = await proxy_client.client.send(upstream_req, stream=True)

    content_type = upstream_resp.headers.get("content-type", "")
    is_streaming = (
        "text/event-stream" in content_type          # SSE (OpenAI/Anthropic)
        or "application/x-ndjson" in content_type     # NDJSON (Ollama)
        or "ndjson" in content_type
    )

    resp_headers = _forward_headers(dict(upstream_resp.headers))
    elapsed_us = (time.perf_counter_ns() - start) // 1000
    resp_headers["X-Sieve-Proxy-Us"] = str(elapsed_us)

    if is_streaming:
        return StreamingResponse(
            content=_stream_chunks(upstream_resp),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    # Non-streaming: read full body, close upstream
    response_body = await upstream_resp.aread()
    await upstream_resp.aclose()

    return Response(
        content=response_body,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=content_type,
    )


async def forward_payload(
    request: Request,
    proxy_client: ProxyClient,
    payload: dict[str, Any],
) -> Response | StreamingResponse:
    """Forward a modified payload to the upstream LLM provider.

    Like forward_request, but sends the given payload dict instead of the
    original request body. Used when the pipeline has stripped/composed
    a lean payload.
    """
    start = time.perf_counter_ns()

    path = request.url.path
    query = str(request.url.query)
    url = path if not query else f"{path}?{query}"

    body = json.dumps(payload).encode()
    # Debug: log outbound payload (truncated)
    logger.info("Outbound payload: %s", body[:500].decode(errors="replace"))
    headers = _forward_headers(dict(request.headers))
    # Update content-length for the new body
    headers["content-length"] = str(len(body))

    upstream_req = proxy_client.client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )

    upstream_resp = await proxy_client.client.send(upstream_req, stream=True)

    content_type = upstream_resp.headers.get("content-type", "")
    is_streaming = (
        "text/event-stream" in content_type
        or "application/x-ndjson" in content_type
        or "ndjson" in content_type
    )

    resp_headers = _forward_headers(dict(upstream_resp.headers))
    elapsed_us = (time.perf_counter_ns() - start) // 1000
    resp_headers["X-Sieve-Proxy-Us"] = str(elapsed_us)

    if is_streaming:
        return StreamingResponse(
            content=_stream_chunks(upstream_resp),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    response_body = await upstream_resp.aread()
    await upstream_resp.aclose()

    return Response(
        content=response_body,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=content_type,
    )


async def _stream_chunks(resp: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Yield chunks from the upstream response, then close it."""
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
    finally:
        await resp.aclose()

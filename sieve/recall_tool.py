"""Recall tool — stream interception and multi-turn internal recall.

When the LLM emits a `recall` tool call in its response, this module:
  1. Intercepts the response (does NOT stream it to the agent)
  2. Executes the recall query against the memory store
  3. Sends a follow-up request to the LLM with the tool result
  4. Repeats up to MAX_ROUNDS times
  5. Streams only the final text response back to the agent

Non-recall tool calls are passed through to the agent unchanged.

Streaming strategy:
  - First request sent with stream=true for low latency
  - Chunks buffered and inspected for tool calls
  - Common path (no recall call): buffered chunks flushed as streaming response
  - Recall path: stream consumed, recall executed, follow-up streamed to agent

Usage (from main.py)::

    handler = RecallHandler(proxy_client, retriever, config)
    response = await handler.handle_chat(
        request, lean_payload, api_format="ollama"
    )
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from sieve.proxy import ProxyClient, _forward_headers

logger = logging.getLogger("recall.tool")

MAX_ROUNDS = 5


class RecallHandler:
    """Handles LLM requests with internal recall tool call interception."""

    def __init__(
        self,
        proxy_client: ProxyClient,
        retriever: Any,
        config: Any,
        tool_registry: Any | None = None,
        slot_retriever: Any | None = None,
    ) -> None:
        self._proxy = proxy_client
        self._retriever = retriever
        self._config = config
        self._tool_registry = tool_registry
        # Cycle 28: SlotRetriever is optional. When schema_v2 is on, the
        # tool-call retrieval path consults it before falling back to the
        # legacy vector retriever.
        self._slot_retriever = slot_retriever

    async def handle_chat(
        self,
        request: Request,
        lean_payload: dict,
        api_format: str = "ollama",
    ) -> Response | StreamingResponse:
        """Send lean_payload to LLM, intercept recall tool calls, return response.

        Supports streaming: first request uses stream=true. If no recall tool
        calls are detected, chunks are forwarded to the agent in real-time.
        If a recall tool call is detected, it's handled internally and the
        final response is streamed back.
        """
        start = time.perf_counter_ns()
        path = request.url.path
        query = str(request.url.query)
        url = path if not query else f"{path}?{query}"

        # Determine if the client wants streaming
        client_wants_stream = lean_payload.get("stream", True)

        messages = list(lean_payload.get("messages", []))
        rounds = 0

        # First round: use streaming to detect tool calls with low latency
        payload = dict(lean_payload)
        payload["messages"] = messages
        payload["stream"] = True

        buffered_chunks, resp_data, has_recall = await self._stream_and_detect(
            url, payload, request, api_format,
        )

        if resp_data is None:
            return _error_response(502, "Upstream LLM request failed")

        # Track non-recall tool calls
        tool_calls = _extract_tool_calls(resp_data, api_format)
        recall_calls = [tc for tc in tool_calls if _is_recall_call(tc, api_format)]
        other_calls = [tc for tc in tool_calls if not _is_recall_call(tc, api_format)]
        if self._tool_registry is not None:
            for tc in other_calls:
                try:
                    name = _tool_call_name(tc, api_format)
                    if name:
                        self._tool_registry.record_usage(name)
                except Exception:
                    pass

        if not recall_calls:
            # Common path: no recall tool calls — return buffered stream
            elapsed_us = (time.perf_counter_ns() - start) // 1000
            if client_wants_stream and buffered_chunks:
                return _streaming_response(buffered_chunks, elapsed_us, 0, api_format)
            else:
                return _build_final_response(resp_data, api_format, elapsed_us, 0)

        # Recall path: handle tool calls internally
        while recall_calls and rounds < MAX_ROUNDS:
            rounds += 1
            logger.info(
                "Recall round %d/%d: %d recall call(s)",
                rounds, MAX_ROUNDS, len(recall_calls),
            )

            # Add assistant message with tool calls to conversation
            assistant_msg = _extract_assistant_message(resp_data, api_format)
            if assistant_msg:
                messages.append(assistant_msg)

            # Execute each recall call
            for tc in recall_calls:
                query_text, scope = _parse_recall_args(tc, api_format)
                logger.info("  recall(query=%r, scope=%r)", query_text[:80], scope)
                ctx = await self._retriever.retrieve(query_text)

                # Cycle 28 schema_v2: try SlotRetriever on the tool-call
                # path too. The pre-populate path does this already, but
                # tool-calling LLMs (qwen3.5:9b and up) prefer to call the
                # recall tool, so most real queries land here.
                tool_text = ctx.text
                ablation_cfg = getattr(self._config, "ablation", None)
                if (ablation_cfg is not None
                        and getattr(ablation_cfg, "schema_v2", False)
                        and self._slot_retriever is not None):
                    try:
                        slot_result = self._slot_retriever.retrieve(query_text)
                        if slot_result.is_hit:
                            from sieve.context_format_v2 import format_context_v2
                            owner_name = ""
                            owner_cfg = getattr(self._config, "profile_owner", None)
                            if owner_cfg is not None:
                                owner_name = getattr(owner_cfg, "name", "") or ""
                            v2_text, v2_tok = format_context_v2(
                                slot_result,
                                profile_owner_name=owner_name,
                                extra_facts=ctx.facts,
                                max_tokens=800,
                            )
                            logger.info(
                                "schema_v2 HIT (recall tool): class=%s slot=%s tokens=%d",
                                slot_result.query_class,
                                slot_result.slot_key,
                                v2_tok,
                            )
                            tool_text = v2_text
                        else:
                            logger.info(
                                "schema_v2 MISS (recall tool, class=%s) → legacy ctx",
                                slot_result.query_class,
                            )
                    except Exception as exc:
                        logger.warning("schema_v2 path failed (recall tool): %s", exc)

                # Cycle 19 Layer 1+2 — Absence signalling and closed-world framing
                # on the tool result. Use the original user query (not the LLM's
                # rewritten recall query) so absence signals reflect what the
                # user actually asked about.
                user_query = _last_user_query(messages) or query_text
                store = getattr(self._retriever, "_store", None)
                ablation = getattr(self._config, "ablation", None)
                if tool_text and ablation is not None:
                    if getattr(ablation, "absence_signal", False):
                        try:
                            from sieve.verification import build_absence_signals
                            # Pass the last-few conversation turns so the
                            # absence layer sees entities introduced
                            # earlier in the session (Q64 widening).
                            recent_turns = [
                                m for m in messages
                                if isinstance(m, dict)
                                and m.get("role") in ("user", "assistant")
                            ][-6:]
                            signals = build_absence_signals(
                                user_query, ctx.facts, store,
                                recent_turns=recent_turns,
                            )
                            if signals:
                                extra = "\n".join(s.text for s in signals)
                                tool_text = tool_text + "\n" + extra
                                logger.info(
                                    "ABL-AS (recall path): injected %d signal(s)",
                                    len(signals),
                                )
                        except Exception as exc:
                            logger.warning("ABL-AS recall path failed: %s", exc)
                    if getattr(ablation, "closed_world", False):
                        try:
                            from sieve.verification import CLOSED_WORLD_FRAMING
                            tool_text = tool_text + CLOSED_WORLD_FRAMING
                            logger.info("ABL-CW (recall path): closed-world framing appended")
                        except Exception as exc:
                            logger.warning("ABL-CW recall path failed: %s", exc)

                tool_result = _build_tool_result(tc, tool_text, api_format)
                messages.append(tool_result)
                logger.info("  → %d facts, ~%d tokens", len(ctx.facts), ctx.token_estimate)

            # Follow-up request — stream for the final response
            payload = dict(lean_payload)
            payload["messages"] = messages
            payload["stream"] = True

            buffered_chunks, resp_data, has_recall = await self._stream_and_detect(
                url, payload, request, api_format,
            )
            if resp_data is None:
                return _error_response(502, "Upstream LLM request failed")

            tool_calls = _extract_tool_calls(resp_data, api_format)
            recall_calls = [tc for tc in tool_calls if _is_recall_call(tc, api_format)]

        if recall_calls:
            logger.warning("Max recall rounds (%d) exceeded", MAX_ROUNDS)

        elapsed_us = (time.perf_counter_ns() - start) // 1000
        if client_wants_stream and buffered_chunks:
            return _streaming_response(buffered_chunks, elapsed_us, rounds, api_format)
        else:
            return _build_final_response(resp_data, api_format, elapsed_us, rounds)

    async def _stream_and_detect(
        self,
        url: str,
        payload: dict,
        request: Request,
        api_format: str,
    ) -> tuple[list[bytes], dict | None, bool]:
        """Send a streaming request, buffer chunks, detect recall tool calls.

        Returns:
            (buffered_chunks, accumulated_response_data, has_recall_call)
        """
        body = json.dumps(payload).encode()
        logger.info("Outbound to LLM: %s", body[:500].decode(errors="replace"))
        headers = _forward_headers(dict(request.headers))
        headers["content-length"] = str(len(body))

        upstream_req = self._proxy.client.build_request(
            method="POST", url=url, headers=headers, content=body,
        )

        try:
            resp = await self._proxy.client.send(upstream_req, stream=True)
        except Exception as exc:
            logger.error("LLM streaming request failed: %s", exc)
            return [], None, False

        buffered_chunks: list[bytes] = []
        accumulated: dict = {}
        has_recall = False

        try:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue

                raw_bytes = (line + "\n").encode()
                buffered_chunks.append(raw_bytes)

                # Parse JSON from NDJSON or SSE format
                json_str = line.strip()
                if json_str.startswith("data: "):
                    json_str = json_str[6:]
                if json_str == "[DONE]":
                    continue

                try:
                    chunk = json.loads(json_str)
                except (json.JSONDecodeError, TypeError):
                    continue

                if api_format == "ollama":
                    accumulated = _merge_ollama_chunk(accumulated, chunk)
                    msg = chunk.get("message", {})
                    if msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            if _is_recall_call(tc, api_format):
                                has_recall = True
                else:
                    accumulated = _merge_openai_chunk(accumulated, chunk)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                if _is_recall_call(tc, api_format):
                                    has_recall = True
        finally:
            await resp.aclose()

        return buffered_chunks, accumulated, has_recall

    async def _send_to_llm(
        self, url: str, payload: dict, request: Request,
    ) -> dict | None:
        """Send a non-streaming request to the LLM, return parsed JSON response."""
        body = json.dumps(payload).encode()
        logger.info("Outbound to LLM (non-stream): %s", body[:500].decode(errors="replace"))
        headers = _forward_headers(dict(request.headers))
        headers["content-length"] = str(len(body))

        upstream_req = self._proxy.client.build_request(
            method="POST", url=url, headers=headers, content=body,
        )

        try:
            resp = await self._proxy.client.send(upstream_req, stream=False)
            return resp.json()
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            return None


# ─── Ollama NDJSON stream accumulation ──────���────────────────────────────────

def _merge_ollama_chunk(accumulated: dict, chunk: dict) -> dict:
    """Merge an Ollama streaming chunk into the accumulated response."""
    if not accumulated:
        accumulated = dict(chunk)
        accumulated.setdefault("message", {"role": "assistant", "content": ""})
        return accumulated

    msg = chunk.get("message", {})
    content = msg.get("content", "")
    if content:
        accumulated.setdefault("message", {})
        accumulated["message"]["content"] = accumulated["message"].get("content", "") + content

    # Tool calls come in the final chunk
    if msg.get("tool_calls"):
        accumulated["message"]["tool_calls"] = msg["tool_calls"]

    # Copy done status and metadata from the final chunk
    if chunk.get("done"):
        for key in ("done", "done_reason", "total_duration", "load_duration",
                     "prompt_eval_count", "prompt_eval_duration",
                     "eval_count", "eval_duration", "model", "created_at"):
            if key in chunk:
                accumulated[key] = chunk[key]

    return accumulated


def _merge_openai_chunk(accumulated: dict, chunk: dict) -> dict:
    """Merge an OpenAI streaming chunk into the accumulated response."""
    if not accumulated:
        accumulated = {
            "id": chunk.get("id", ""),
            "object": "chat.completion",
            "created": chunk.get("created", 0),
            "model": chunk.get("model", ""),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
            "_tc_accum": {},  # tool call accumulator by index
        }

    choices = chunk.get("choices", [])
    if choices:
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            accumulated["choices"][0]["message"]["content"] += content

        # Accumulate tool calls across chunks (args arrive in fragments)
        if delta.get("tool_calls"):
            tc_accum = accumulated["_tc_accum"]
            for tc_delta in delta["tool_calls"]:
                idx = tc_delta.get("index", 0)
                if idx not in tc_accum:
                    tc_accum[idx] = {
                        "id": tc_delta.get("id", f"call_{idx}"),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    tc_accum[idx]["function"]["name"] = fn["name"]
                tc_accum[idx]["function"]["arguments"] += fn.get("arguments", "")
            # Rebuild tool_calls list from accumulator
            accumulated["choices"][0]["message"]["tool_calls"] = [
                tc_accum[i] for i in sorted(tc_accum)
            ]

    return accumulated


# ─── Tool call extraction ────────────��────────────────────────────────────────

def _last_user_query(messages: list[dict]) -> str | None:
    """Cycle 19: return the most recent user message text from a chat list."""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _extract_tool_calls(resp_data: dict, api_format: str) -> list[dict]:
    """Extract tool calls from an LLM response."""
    if api_format == "ollama":
        msg = resp_data.get("message", {})
        return msg.get("tool_calls", [])
    else:  # openai
        choices = resp_data.get("choices", [])
        if not choices:
            return []
        msg = choices[0].get("message", {})
        return msg.get("tool_calls", [])


def _is_recall_call(tc: dict, api_format: str) -> bool:
    """Check if a tool call is for the recall tool."""
    func = tc.get("function", {})
    return func.get("name") == "recall"


def _parse_recall_args(tc: dict, api_format: str) -> tuple[str, str]:
    """Extract query and scope from a recall tool call."""
    if api_format == "ollama":
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
    else:  # openai
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}

    return args.get("query", ""), args.get("scope", "all")


def _extract_assistant_message(resp_data: dict, api_format: str) -> dict | None:
    """Extract the assistant message (including tool calls) from a response."""
    if api_format == "ollama":
        msg = resp_data.get("message")
        return dict(msg) if msg else None
    else:  # openai
        choices = resp_data.get("choices", [])
        if choices:
            msg = choices[0].get("message")
            return dict(msg) if msg else None
        return None


def _build_tool_result(tc: dict, result_text: str, api_format: str) -> dict:
    """Build a tool result message for the conversation."""
    if not result_text:
        result_text = "No relevant context found in memory."

    if api_format == "ollama":
        return {"role": "tool", "content": result_text}
    else:  # openai
        return {
            "role": "tool",
            "tool_call_id": tc.get("id", "recall_0"),
            "content": result_text,
        }


def _streaming_response(
    buffered_chunks: list[bytes],
    elapsed_us: int,
    recall_rounds: int,
    api_format: str = "ollama",
) -> StreamingResponse:
    """Return a StreamingResponse that yields buffered chunks."""
    async def _yield_chunks() -> AsyncGenerator[bytes, None]:
        for chunk in buffered_chunks:
            yield chunk

    media_type = "text/event-stream" if api_format == "openai" else "application/x-ndjson"

    return StreamingResponse(
        content=_yield_chunks(),
        status_code=200,
        headers={
            "X-Sieve-Proxy-Us": str(elapsed_us),
            "X-Sieve-Rounds": str(recall_rounds),
        },
        media_type=media_type,
    )


def _build_final_response(
    resp_data: dict,
    api_format: str,
    elapsed_us: int,
    recall_rounds: int,
) -> Response:
    """Build a non-streaming FastAPI Response."""
    headers = {
        "X-Sieve-Proxy-Us": str(elapsed_us),
        "X-Sieve-Rounds": str(recall_rounds),
    }
    body = json.dumps(resp_data).encode()
    return Response(
        content=body,
        status_code=200,
        headers=headers,
        media_type="application/json",
    )


def _tool_call_name(tc: dict, api_format: str) -> str:
    """Extract the name from a tool call object."""
    func = tc.get("function")
    if isinstance(func, dict):
        return func.get("name", "")
    return tc.get("name", "")


def _error_response(status: int, message: str) -> Response:
    return Response(
        content=json.dumps({"error": message}).encode(),
        status_code=status,
        media_type="application/json",
    )

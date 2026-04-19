"""Query decomposition for multi-hop retrieval.

Multi-hop queries ("Given Dad's eye condition, Mum's driving, and our
location, should they move closer?") underperform when a single top-K
vector search returns similar facts rather than the logically required
mix.

Fix: ask a small LLM to break the query into 2-4 independent
sub-questions, run top-3 vector search per sub-question, deduplicate
the combined fact list, and hand the merged set to the usual context
formatter.

Cost profile: one extra small LLM call per complexity=2 query (~100ms
local, <500ms cloud), plus 2-4 additional vector searches (each
~5-15ms). Total latency budget: ~200-700ms per multi-hop query, a
fraction of the model-inference call that follows.

Fail-open: on any LLM error, parse failure, or empty sub-question list
we return an empty list so the caller can fall back to the single-query
path.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

logger = logging.getLogger("recall.query_decomposer")


_DECOMPOSE_READ_TIMEOUT_S = 30.0
_DECOMPOSE_MAX_TOKENS = 160
_DECOMPOSE_MAX_SUBQUERIES = 4
_DECOMPOSE_MIN_SUBQUERIES = 2

_DECOMPOSE_SYSTEM_PROMPT = (
    "You break a user's multi-hop question into 2-4 independent "
    "sub-questions, each focused on ONE fact that must be retrieved "
    "to answer the original. Output ONLY a JSON array of strings — no "
    "preamble, no keys. Keep each sub-question short (under 15 words) "
    "and self-contained."
)


def _build_ollama_body(model: str, messages: list[dict], num_ctx: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": _DECOMPOSE_MAX_TOKENS,
        },
    }


def _build_openai_body(model: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": _DECOMPOSE_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def _extract_content(data: dict) -> str:
    msg = data.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"] or ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message") or {}
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
    return ""


def _parse_subqueries(raw: str) -> list[str]:
    """Parse the LLM output into a clean list of sub-questions.

    Accepts either a bare JSON array or an object with a known wrapper
    key ("questions", "subqueries", etc.) so we tolerate small prompt
    drift without hard-failing.
    """
    text = (raw or "").strip()
    if not text:
        return []
    # Strip markdown fences the model may have added despite instructions.
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(parsed, dict):
        for key in ("questions", "subqueries", "sub_questions", "queries"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            s = item.strip()
            # Accept any non-trivial sub-question. 2-char threshold catches
            # pure whitespace / single-letter garbage without rejecting
            # terse valid forms like "Q1?".
            if len(s) >= 2:
                out.append(s)
        if len(out) >= _DECOMPOSE_MAX_SUBQUERIES:
            break
    return out


async def decompose_query(
    query: str,
    provider_base_url: str,
    model: str,
    num_ctx: int = 2048,
) -> list[str]:
    """Return 2-4 sub-queries, or [] on any failure.

    The caller must only invoke this for queries already tagged
    complexity=2 — running it on simple queries wastes a model call.
    """
    q = (query or "").strip()
    if not q:
        return []
    messages = [
        {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_DECOMPOSE_READ_TIMEOUT_S),
        ) as client:
            resp = await client.post(
                f"{provider_base_url}/api/chat",
                json=_build_ollama_body(model, messages, num_ctx),
            )
            if 400 <= resp.status_code < 500:
                resp = await client.post(
                    f"{provider_base_url}/v1/chat/completions",
                    json=_build_openai_body(model, messages),
                )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.info("Query decomposition call failed (model=%s): %s", model, exc)
        return []

    raw = _extract_content(data)
    subs = _parse_subqueries(raw)
    # Drop duplicates and exact matches of the original query (no value
    # running the same search again).
    seen: set[str] = set()
    deduped: list[str] = []
    orig_norm = re.sub(r"\s+", " ", q).lower()
    for s in subs:
        norm = re.sub(r"\s+", " ", s).lower()
        if norm == orig_norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(s)
    if len(deduped) < _DECOMPOSE_MIN_SUBQUERIES:
        return []
    return deduped[:_DECOMPOSE_MAX_SUBQUERIES]

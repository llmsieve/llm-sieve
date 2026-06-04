"""EXTREME summariser.

When an inbound request carries a very large static payload (>25K tokens
of identity / workspace files / prior turns that Recall is about to
strip), compress the stripped bloat into a ~500-token narrative summary
and include it in the composed context block as [NARRATIVE SUMMARY].

This gives the downstream model the temporal/causal reasoning depth
that purely slot-based retrieval can't provide, without shipping the
original 42K tokens. A ~500-token summary is still a >95% reduction.

WARNING: summaries are NOT authoritative. Validation testing caught
the summariser conflating "Dana is pregnant" into "Jamie is pregnant".
The cardinal rules header still forces the model to trust [CURRENT]
slot facts over anything in the summary, so conflicts resolve in favor
of structured data. The summary is additive, not primary.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("recall.extreme_summary")


# Threshold: only summarise when the stripped bloat exceeds this many
# estimated tokens. Below this, the existing retrieval path is enough.
EXTREME_THRESHOLD_TOKENS = 25_000

# Target length for the summary itself. ~500 tokens ≈ 2000 chars.
SUMMARY_TARGET_TOKENS = 500


_SUMMARY_PROMPT = """Summarise the following conversation or context \
into a concise {target_tokens}-token narrative summary.

Preserve:
- Key facts about the subject (names, roles, relationships, locations).
- Temporal sequence of events (what happened and when).
- How relationships between people evolved.

Do NOT:
- Invent facts not in the source.
- Conflate different people (if A is pregnant, do NOT write that B is \
pregnant).
- Include trivia or tangential detail — only what a downstream model \
would need to answer questions about the subject.

CONTEXT:
{text}

SUMMARY:"""


def rough_token_count(text: str) -> int:
    """Cheap token estimate — 1 token per 4 chars, rounded up."""
    return max(1, (len(text) + 3) // 4)


def should_summarise(stripped_text: str, *, enabled: bool) -> bool:
    """Return True if the stripped bloat warrants an EXTREME summary."""
    if not enabled:
        return False
    return rough_token_count(stripped_text) >= EXTREME_THRESHOLD_TOKENS


async def summarise_async(
    stripped_text: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str,  # required — caller must resolve via resolve_writer_model(config)
    num_ctx: int = 32768,
    target_tokens: int = SUMMARY_TARGET_TOKENS,
    timeout: float = 120.0,
) -> str | None:
    """Produce a ~target_tokens narrative summary of stripped bloat.

    ``model`` is required; callers should resolve it via
    ``resolve_writer_model(config)`` so the user's configured model is used
    rather than any hardcoded tag.

    Returns None on any failure — caller MUST fall back to no summary
    (never block the whole request on a summariser failure).
    """
    if not stripped_text.strip():
        return None

    prompt = _SUMMARY_PROMPT.format(
        target_tokens=target_tokens,
        text=stripped_text,
    )
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,  # qwen → safe to suppress reasoning tokens
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(f"{base_url}/api/generate", json=body)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
    except Exception as exc:
        logger.warning("extreme summariser failed: %s", exc)
        return None

    summary = (raw or "").strip()
    if not summary:
        return None
    return summary


def format_summary_section(summary: str | None) -> str:
    """Render a non-empty summary as a [NARRATIVE SUMMARY] context section."""
    if not summary:
        return ""
    return "[NARRATIVE SUMMARY]\n" + summary.strip() + "\n"

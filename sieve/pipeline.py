"""Payload stripping and lean composition (Phase 4).

Takes a decomposed payload and composes a minimal version:
- Replaces the bloated system prompt with a lean ~200 token prompt
- Injects the recall tool definition
- Keeps only the last N conversation turns
- Strips workspace files and tool schemas
- Preserves model, stream, and other options
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sieve.config import PipelineConfig
from sieve.fingerprint import DecomposedPayload, _estimate_tokens
from sieve.progression import PhaseDecision

logger = logging.getLogger("recall.pipeline")

# The lean system prompt that replaces the agent's bloated one.
#
# A1/A2/A4/A7/D6 rewrite (2026-04-21): the earlier prompt emphasised
# "direct and concise" + "under 200 words" and produced pathological
# terseness — seed acknowledgements became "Noted.", lookups returned
# bare values with no context ("£240,000" answering "how much do I owe
# on my mortgage"). Graders reading those replies scored them well
# below baseline responses that engaged with the question. The new
# prompt explicitly tells the model to:
#   * engage with new facts before confirming them (A1)
#   * pair factual answers with one sentence of useful context (A2)
#   * treat integrative queries as summaries, not lookups (A4)
#   * weave personal facts into general-knowledge answers (A7)
#   * never reply in fewer than a sentence (D6; 30B models were
#     emitting single tokens like "Noted." or "4.2%")
LEAN_SYSTEM_PROMPT = """\
You are the user's personal AI assistant with access to their persistent memory.
The [Recalled context] block above contains facts the user has shared with you
in prior turns. Treat it as authoritative: these are real details about the
user's life, not assumptions.

How to respond:
- Use the recalled context whenever it's relevant. Cite specifics (names,
  numbers, places) when you have them.
- When the user shares a NEW fact (statements, not questions), acknowledge
  it briefly AND engage with it — one or two sentences beyond "noted". Do
  not reply with just "Noted." or a single word.
- When the user asks a factual lookup ("What's my X?"), give the answer
  plus one sentence of useful surrounding context. Never reply with only
  a bare value.
- When the user asks an integrative question ("summarise my life",
  "plan a week", "what should I focus on"), synthesise from what IS
  in the recalled context. Do NOT invent categories or structure
  that isn't represented in the facts. If the user asks about a
  category you don't have ("temporal changes I've tracked", "my
  tracked habits", "my measured metrics"), say plainly that there
  is no such record rather than inventing one. A short honest answer
  beats an elaborate fabricated one.
- For questions that blend personal and general knowledge ("recommend a
  cookbook for date night"), use your general knowledge AND weave in what
  you know about the user.
- If you genuinely do not know something (it's not in recalled context
  and isn't general knowledge), use the recall tool. Do not guess.
- Do not repeat the question back. Do not add boilerplate disclaimers
  about being an AI. Keep responses under 300 words unless the question
  genuinely calls for more."""

# When the classifier confidently tags a query as pure general
# knowledge (Level 0, high confidence), swap the memory-focused framing
# above for a neutral "helpful, knowledgeable assistant" framing. The
# memory framing biases the model toward hedging or refusing on
# general-knowledge queries; a clean framing closes the G-category gap.
GENERAL_LEAN_SYSTEM_PROMPT = """\
You are a helpful, knowledgeable assistant.
Answer the user's question directly and concisely using your own knowledge. \
Do not repeat the question. Do not add unnecessary preamble or disclaimers. \
Keep responses under 200 words unless the question requires detailed explanation."""

# Patterns in the agent's original system prompt that identify the
# user by name. When present, the extracted sentence is appended to the
# lean system prompt so cold-start queries on an empty store still know
# who "I" is. Covers the harness runner's phrasing plus common variants.
_OWNER_PIN_PATTERNS = (
    re.compile(
        r"(?P<pin>The person (?:you'?re\s+(?:speaking\s+)?with|"
        r"(?:speaking|talking))\s+(?:to|with)?\s*(?:is|here is|"
        r"will be)?\s*[^.]+?\.)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<pin>You(?:'re|\s+are)\s+(?:a\s+|an\s+)?(?:helpful\s+)?"
        r"(?:assistant|AI\s+assistant)\s+(?:for|to|helping)\s+[^.]+?\.)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<pin>[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+\s+is\s+a\s+\d+[^.]+?\.)",
    ),
)


def _extract_owner_pin(system_text: str) -> str:
    """Find the one-sentence identity pin in the agent's system prompt.

    Returns the first matching sentence (e.g. "The person speaking is
    Jamie Rivera, a 41-year-old..."). Empty string when nothing matches.
    Used by compose_lean_payload to preserve identity grounding across
    the lean-prompt substitution.
    """
    if not system_text:
        return ""
    for rx in _OWNER_PIN_PATTERNS:
        m = rx.search(system_text)
        if m:
            pin = m.group("pin").strip()
            if len(pin) < 250:
                return pin
    return ""


# The recall tool definition injected into every lean payload
RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Retrieve relevant context from the user's personal memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What you need to know about the user",
                },
                "scope": {
                    "type": "string",
                    "enum": ["facts", "episodes", "all"],
                    "description": "Type of memory to search",
                },
            },
            "required": ["query"],
        },
    },
}


def _tool_name(tool: dict) -> str:
    """Extract tool name from either Ollama or OpenAI tool shape."""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("name", "")
    return tool.get("name", "")


def _last_n_turns(history: list[dict], n: int) -> list[dict]:
    """Keep the last N conversation turns (a turn = user + assistant pair).

    Walks backwards through messages, collecting up to n user messages
    and their corresponding assistant replies.
    """
    if n <= 0 or not history:
        return []

    # Walk backwards collecting turns
    kept: list[dict] = []
    user_count = 0

    for msg in reversed(history):
        role = msg.get("role", "")
        if role == "user":
            user_count += 1
            if user_count > n:
                break
        kept.append(msg)

    kept.reverse()
    return kept


def _count_lean_tokens(lean: dict) -> int:
    """Estimate the outbound token count of a composed lean payload."""
    return _estimate_tokens(json.dumps(lean))


def _apply_token_budget(lean: dict, max_tokens: int) -> dict:
    """Progressively trim the lean payload until it fits within `max_tokens`.

    Trim order (per spec):
        1. Reduce history to the last 1 turn (drop older user/assistant pairs).
        2. Halve the retrieved-context system message, repeatedly, until it
           either fits or is removed entirely.
        3. If still over budget, emit a WARNING. The payload is returned as-is
           in that case — we do not truncate the current user message or the
           tools array, because doing so silently could break the request.

    The lean dict is mutated in place (and also returned for convenience).
    No-op when `max_tokens <= 0` or the payload already fits.
    """
    if max_tokens <= 0:
        return lean

    tokens = _count_lean_tokens(lean)
    if tokens <= max_tokens:
        return lean

    over_by = tokens - max_tokens
    logger.info(
        "Token budget exceeded: %d > %d (over by %d), trimming",
        tokens, max_tokens, over_by,
    )

    # --- Step 1: reduce conversation history to the last 1 turn ---
    messages = lean.get("messages", [])
    # Layout after compose: [lean_system, (retrieved_context system)?, *history, user_message]
    # Identify boundaries:
    #   - the first system message is the lean system prompt (always at index 0)
    #   - an optional second system message at index 1 is the retrieved context
    #   - the last message is the current user query (never trimmed)
    #   - everything in between is history
    context_idx = 1 if len(messages) > 1 and messages[1].get("role") == "system" else None
    history_start = 2 if context_idx is not None else 1
    history_end = len(messages) - 1  # last message is the current user query

    if history_end > history_start:
        history_slice = messages[history_start:history_end]
        reduced = _last_n_turns(history_slice, 1)
        removed = len(history_slice) - len(reduced)
        if removed > 0:
            lean["messages"] = (
                messages[:history_start] + reduced + messages[history_end:]
            )
            logger.info(
                "Token budget: reduced history to last 1 turn (-%d messages)", removed,
            )
            tokens = _count_lean_tokens(lean)
            if tokens <= max_tokens:
                return lean

    # --- Step 2: halve retrieved-context message repeatedly ---
    if context_idx is not None:
        ctx_msg = lean["messages"][context_idx]
        ctx_text = ctx_msg.get("content", "") if isinstance(ctx_msg, dict) else ""
        while ctx_text and tokens > max_tokens:
            new_len = len(ctx_text) // 2
            if new_len == 0:
                # Remove the entire context message
                lean["messages"] = (
                    lean["messages"][:context_idx] + lean["messages"][context_idx + 1:]
                )
                logger.info("Token budget: removed retrieved context entirely")
                break
            ctx_text = ctx_text[:new_len]
            lean["messages"][context_idx] = {"role": "system", "content": ctx_text}
            logger.info(
                "Token budget: truncated retrieved context to %d chars", new_len,
            )
            tokens = _count_lean_tokens(lean)

        if tokens <= max_tokens:
            return lean

    # --- Step 3: still over budget after both trim steps ---
    tokens = _count_lean_tokens(lean)
    if tokens > max_tokens:
        logger.warning(
            "Token budget still exceeded after trimming: %d > %d (over by %d) — "
            "current user message and tools array are preserved as-is",
            tokens, max_tokens, tokens - max_tokens,
        )
    return lean


def compose_lean_payload(
    original_payload: dict,
    decomposed: DecomposedPayload,
    config: PipelineConfig,
    retrieved_context: str = "",
    profile_owner_pin: str = "",
    pure_general: bool = False,
    progression: PhaseDecision | None = None,
) -> dict:
    """Compose a lean payload from the decomposed original.

    Strips: system prompt bloat, tool schemas, workspace files, old history.
    Keeps: model, stream flag, options, last N turns, user message.
    Adds: lean system prompt, recall tool, and optionally pre-populated context.

    Args:
        retrieved_context: Pre-retrieved context block to inject (empty = skip).
                           Injected as a second system message after the lean prompt.
        profile_owner_pin: Explicit identity sentence to preserve across the
                           lean-prompt substitution. If empty, the composer
                           scans the agent's original system prompt for a
                           matching pin and reuses that.
        pure_general: When True, swap LEAN_SYSTEM_PROMPT for
                      GENERAL_LEAN_SYSTEM_PROMPT (neutral "helpful,
                      knowledgeable assistant" framing). Used by the
                      caller when the L0 classifier is confident the
                      query needs no personal context.
        progression: Optional progressive-activation decision. When
                      supplied, ``progression.turns`` overrides
                      ``config.conversation_turns`` for the history
                      trim. When omitted, the pipeline's static
                      ``conversation_turns`` applies — this preserves
                      backward compatibility for callers that haven't
                      wired in phase detection yet.

    Returns the new payload dict ready to forward to the LLM.
    """
    messages: list[dict] = []

    # 1. Lean system prompt (replaces the bloated one).
    # For pure general-knowledge queries, use the neutral framing so the
    # model answers from its own knowledge without any memory bias. Any
    # detected owner pin is intentionally dropped for pure-G queries
    # because the pin exists to ground personal context, which by
    # definition is irrelevant here.
    base_prompt = GENERAL_LEAN_SYSTEM_PROMPT if pure_general else LEAN_SYSTEM_PROMPT
    lean_system = base_prompt
    if not pure_general:
        pin = profile_owner_pin.strip()
        if not pin:
            sys_section = decomposed.section_by_name("system_prompt")
            if sys_section and sys_section.content:
                pin = _extract_owner_pin(sys_section.content)
        if pin:
            lean_system = f"{base_prompt}\n\n{pin}"
    messages.append({"role": "system", "content": lean_system})

    # 1b. Pre-populated context block (injected between system prompt and history)
    if retrieved_context:
        messages.append({"role": "system", "content": retrieved_context})

    # 2. Last N conversation turns from history
    hist_section = decomposed.section_by_name("conversation_history")
    if hist_section and hist_section.content:
        try:
            history_msgs = json.loads(hist_section.content)
        except (json.JSONDecodeError, TypeError):
            history_msgs = []
        turns_to_keep = progression.turns if progression is not None else config.conversation_turns
        trimmed = _last_n_turns(history_msgs, turns_to_keep)
        messages.extend(trimmed)

    # 3. User message (always included)
    user_section = decomposed.section_by_name("user_message")
    if user_section:
        messages.append({"role": "user", "content": user_section.content})

    # 4. Build lean payload — preserve model, stream, options
    inbound_tools = original_payload.get("tools") or []
    # Drop any agent tool named "recall" — ours takes precedence.
    # Non-dict items are passed through untouched so we don't crash on malformed payloads.
    deduped = []
    for t in inbound_tools:
        if isinstance(t, dict) and _tool_name(t) == "recall":
            logger.warning(
                "Agent sent a tool named 'recall' — dropping it; Recall's own tool wins"
            )
            continue
        deduped.append(t)

    lean: dict[str, Any] = {
        "model": original_payload.get("model", ""),
        "messages": messages,
        "tools": [RECALL_TOOL] + deduped,
    }

    # Preserve stream flag
    if "stream" in original_payload:
        lean["stream"] = original_payload["stream"]

    # Preserve options (temperature, etc.) but not tools/messages
    if "options" in original_payload:
        lean["options"] = original_payload["options"]

    # Intentionally do NOT inject think:false.
    # On qwen3:30b-a3b + Ollama 0.20.2, top-level think:false disables
    # the structured `thinking` field without suppressing reasoning
    # generation, so reasoning text leaks into `message.content`
    # (terminated by </think>, followed by the answer only if the
    # response isn't truncated by num_predict). With `think` unset,
    # Ollama's template correctly separates reasoning → `thinking`
    # and answer → `content`, and downstream readers see clean content.

    # 5. Apply token budget (progressive trim if over max_outbound_tokens)
    upstream_ctx = (
        (original_payload.get("options") or {}).get("num_ctx")
        or config.upstream_ctx_default
    )
    _apply_token_budget(lean, config.resolve_budget(upstream_ctx))

    # Log the reduction
    input_tokens = decomposed.total_tokens
    output_tokens = _estimate_tokens(json.dumps(lean))

    if input_tokens > 0:
        reduction = (1 - output_tokens / input_tokens) * 100
    else:
        reduction = 0.0

    # Log key outbound flags for debugging
    logger.info(
        "Compose: model=%s stream=%s think=%s tools=%d",
        lean.get("model", "?"),
        lean.get("stream", "unset"),
        lean.get("think", "unset"),
        len(lean.get("tools", [])),
    )

    logger.info(
        "Strip: %s tokens → %s tokens (%.0f%% reduction)",
        f"{input_tokens:,}", f"{output_tokens:,}", reduction,
    )

    return lean


async def compose_with_tool_selection(
    original_payload: dict,
    decomposed: DecomposedPayload,
    config: PipelineConfig,
    tool_classifier: Any,
    user_query: str,
    retrieved_context: str = "",
    profile_owner_pin: str = "",
    pure_general: bool = False,
    progression: PhaseDecision | None = None,
) -> dict:
    """Layer 2 wrapper: compose the lean payload, then filter tools via the classifier.

    Workflow:
        1. Call the existing sync `compose_lean_payload` to get a Layer-1
           passthrough payload (all agent tools preserved alongside recall).
        2. Call `tool_classifier.select(user_query)` to get the selected
           subset of tools.
        3. Rebuild `lean["tools"] = [RECALL_TOOL] + classifier_tools`,
           dropping any stray "recall" collision.

    If the classifier raises, we log the exception and fall back to the
    Layer-1 passthrough payload (all agent tools included).
    """
    lean = compose_lean_payload(
        original_payload, decomposed, config,
        retrieved_context=retrieved_context,
        profile_owner_pin=profile_owner_pin,
        pure_general=pure_general,
        progression=progression,
    )

    try:
        selection = await tool_classifier.select(user_query)
    except Exception as exc:
        logger.warning(
            "Tool classifier failed, falling back to Layer 1 passthrough: %s", exc
        )
        return lean

    selected = selection.tools or []

    # Defensive dedupe — the classifier should never return "recall" itself,
    # but in case a future refactor allows it, strip collisions again.
    deduped: list[dict] = []
    for t in selected:
        if isinstance(t, dict) and _tool_name(t) == "recall":
            continue
        deduped.append(t)

    lean["tools"] = [RECALL_TOOL] + deduped

    logger.info(
        "Tool selection applied: level=%s n=%d reason=%r",
        selection.level, len(deduped), selection.reason,
    )

    # Re-apply the token budget — the tools array shape has changed, so the
    # budget check from compose_lean_payload may no longer be accurate. This
    # is a no-op when the selected tools are a subset of the passthrough set
    # (the common case), but it catches scenarios where a classifier for some
    # reason returns more tokens than the raw agent tools (shouldn't happen,
    # but cheap to guard).
    upstream_ctx = (
        (original_payload.get("options") or {}).get("num_ctx")
        or config.upstream_ctx_default
    )
    _apply_token_budget(lean, config.resolve_budget(upstream_ctx))
    return lean

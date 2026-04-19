"""History-preamble adapter.

Some agent frameworks embed the entire conversation history INSIDE the
`user` content field (not inside `messages[].history`), using a fixed
marker pair:

    [Chat messages since your last reply - for context]
    User: ...
    Assistant: ...
    User: ...
    Assistant: ...

    [Current message - respond to this]
    User: <the actual new question>

Sieve's `_apply_token_budget` refuses to truncate the current user
message (doing so silently would break semantics), so every such
request grows linearly. This adapter lifts the history block out of
the user content and rewrites it as proper message-level turns, so
the existing `conversation_history` strip / fingerprint / last-N-turns
logic can apply. The `user` field is left holding only the real new
question.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("sieve.history_preamble_adapter")


# Fixed marker pair that the agent framework always emits. The `-`
# variants cover the historical and current spellings we've seen in
# captured payloads.
_HISTORY_MARKER_RX = re.compile(
    r"\[Chat\s+messages\s+since\s+your\s+last\s+reply\s*[-–—]\s*"
    r"for\s+context\]",
    re.IGNORECASE,
)
_CURRENT_MARKER_RX = re.compile(
    r"\[Current\s+message\s*[-–—]\s*respond\s+to\s+this\]",
    re.IGNORECASE,
)


# Turn header: "User: ..." / "Assistant: ..." at the start of a line.
# Case-insensitive; captures the role and lets us slice at the header
# boundaries.
_TURN_HEADER_RX = re.compile(
    r"^(?P<role>User|Assistant)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _split_turns(block: str) -> list[dict]:
    """Parse a `User: ...\nAssistant: ...` block into message dicts.

    Returns a list of {"role": "user"|"assistant", "content": text}
    preserving order. Empty segments are dropped. Non-matching leading
    text is attached to the first user turn (rare; guards against
    malformed preambles).
    """
    headers = list(_TURN_HEADER_RX.finditer(block))
    if not headers:
        return []
    turns: list[dict] = []
    for i, m in enumerate(headers):
        content_start = m.end()
        content_end = headers[i + 1].start() if i + 1 < len(headers) else len(block)
        content = block[content_start:content_end].strip()
        if not content:
            continue
        role = m.group("role").lower()
        turns.append({"role": role, "content": content})
    return turns


def has_history_preamble(user_text: str) -> bool:
    """Cheap probe used by the intercept handlers to decide whether to
    invoke the adapter. Matches on the history marker only — the
    current-message marker is not always present in follow-up payloads."""
    if not user_text:
        return False
    return bool(_HISTORY_MARKER_RX.search(user_text))


def adapt_history_preamble_payload(payload: dict[str, Any]) -> bool:
    """Rewrite a history-preamble-shaped payload in place.

    When the final `user` message content contains the marker pair,
    this function:
      1. extracts the history block between the two markers and parses
         it into message dicts (`{role, content}`)
      2. inserts those turns into `payload["messages"]` just before the
         final user message, so they flow through the standard
         conversation_history section (stripped, fingerprinted, fed to
         the writer, trimmed by last-N-turns).
      3. replaces the final user message content with only the real new
         question (everything after `[Current message - respond to this]`).

    Returns True when the payload was modified, False otherwise. Safe
    to call on any payload; malformed content falls through unchanged.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    # Find the last user message — the one the agent framework packages
    # with history.
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return False

    user_content = messages[user_idx].get("content")
    if not isinstance(user_content, str) or not _HISTORY_MARKER_RX.search(user_content):
        return False

    h_match = _HISTORY_MARKER_RX.search(user_content)
    c_match = _CURRENT_MARKER_RX.search(user_content)
    if h_match is None:
        return False

    history_start = h_match.end()
    if c_match is not None:
        history_block = user_content[history_start:c_match.start()].strip()
        current_block = user_content[c_match.end():].strip()
    else:
        # No explicit "[Current message...]" marker — treat the last
        # "User: ..." as the current question and everything before it
        # as history.
        tail_headers = list(_TURN_HEADER_RX.finditer(user_content[history_start:]))
        if not tail_headers:
            return False
        last_user = None
        for m in tail_headers:
            if m.group("role").lower() == "user":
                last_user = m
        if last_user is None:
            return False
        history_block = user_content[history_start:history_start + last_user.start()].strip()
        current_block = user_content[history_start + last_user.end():].strip()

    turns = _split_turns(history_block)
    # If the current block is itself "User: ..." (because the
    # [Current...] marker framed the header), strip it.
    cb_match = _TURN_HEADER_RX.match(current_block)
    if cb_match and cb_match.group("role").lower() == "user":
        current_block = current_block[cb_match.end():].strip()

    if not current_block:
        # Nothing to ask — leave payload alone rather than emit an
        # empty user turn.
        return False

    # Splice: keep everything BEFORE the last user msg, insert history
    # turns, then the freshly extracted current user question.
    before = messages[:user_idx]
    after = messages[user_idx + 1:]
    new_user = {"role": "user", "content": current_block}
    payload["messages"] = before + turns + [new_user] + after

    logger.info(
        "history-preamble adapter: lifted %d history turns, shrank user msg "
        "%d→%d chars", len(turns), len(user_content), len(current_block),
    )
    return True

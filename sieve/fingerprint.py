"""Payload decomposition and xxHash fingerprinting.

Parses incoming chat payloads into logical sections, hashes each with xxHash,
and detects which sections changed since the last request. This phase is
observation-only — the payload is forwarded unchanged.

Supports both Ollama (/api/chat) and OpenAI (/v1/chat/completions) formats.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import xxhash

logger = logging.getLogger("recall.fingerprint")

# Rough token estimate: ~4 chars per token for English text
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate. Errs on the side of overestimating."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _hash_content(content: str) -> str:
    """Hash a string with xxHash (xxh64). Returns hex digest."""
    return xxhash.xxh64(content).hexdigest()


@dataclass
class Section:
    """A decomposed section of a chat payload."""
    name: str
    content: str
    token_estimate: int
    hash: str
    changed: bool = True

    def log_line(self) -> str:
        status = "new" if self.changed else "unchanged"
        return f"{self.name}: ~{self.token_estimate:,} tokens ({status})"


@dataclass
class DecomposedPayload:
    """Result of decomposing a chat payload into sections."""
    sections: list[Section] = field(default_factory=list)
    format: str = "unknown"  # "ollama" or "openai"
    decompose_time_us: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(s.token_estimate for s in self.sections)

    @property
    def changed_tokens(self) -> int:
        return sum(s.token_estimate for s in self.sections if s.changed)

    @property
    def unchanged_tokens(self) -> int:
        return sum(s.token_estimate for s in self.sections if not s.changed)

    def section_by_name(self, name: str) -> Section | None:
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def log_breakdown(self) -> None:
        """Log the full section breakdown."""
        lines = [s.log_line() for s in self.sections]
        logger.info(
            "Payload breakdown (%s, ~%s total tokens, ~%s unchanged):\n  %s",
            self.format,
            f"{self.total_tokens:,}",
            f"{self.unchanged_tokens:,}",
            "\n  ".join(lines),
        )


# --- Stored hashes for change detection ---

class FingerprintCache:
    """In-memory cache of section hashes, backed by the fingerprints table.

    When a MemoryStore is available, hashes are persisted. Otherwise
    operates purely in-memory (useful for tests or when store isn't init'd).
    """

    def __init__(self, store: Any | None = None):
        self._store = store
        self._cache: dict[str, str] = {}
        if store is not None and store._conn is not None:
            self._load_from_store()

    def _load_from_store(self) -> None:
        """Load existing fingerprints from the DB into memory."""
        try:
            rows = self._store.conn.execute(
                "SELECT section_key, hash FROM fingerprints"
            ).fetchall()
            self._cache = {row[0]: row[1] for row in rows}
        except Exception:
            pass

    def check_and_update(self, section_key: str, new_hash: str) -> bool:
        """Check if a section changed. Updates the stored hash. Returns True if changed."""
        old_hash = self._cache.get(section_key)
        changed = old_hash != new_hash
        if changed:
            self._cache[section_key] = new_hash
            if self._store is not None and self._store._conn is not None:
                self._store.upsert_fingerprint(section_key, new_hash)
        return changed


# --- Decomposition ---

def _extract_workspace_content(messages: list[dict]) -> tuple[str, list[dict]]:
    """Detect and extract workspace file content from system messages.

    Some agents embed workspace files (AGENTS.md, SOUL.md, etc.) as separate
    system messages or within the main system prompt. This extracts them.

    Returns (workspace_text, remaining_messages).
    """
    workspace_parts = []
    remaining = []

    # Markers that indicate workspace/file content in system messages
    file_markers = {
        "AGENTS.md", "SOUL.md", "README.md", "CLAUDE.md",
        "```", "# File:", "## File:", "---\nfile:",
    }

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)

        if role == "system":
            # Check if this looks like an embedded file
            is_file_content = any(marker in content for marker in file_markers)
            # But the first/main system prompt typically isn't a file
            if is_file_content and len(content) > 500 and remaining:
                workspace_parts.append(content)
            else:
                remaining.append(msg)
        else:
            remaining.append(msg)

    return "\n---\n".join(workspace_parts), remaining


def _extract_tools_content(payload: dict) -> str:
    """Extract tool definitions from the payload.

    Ollama: payload["tools"] — list of tool objects
    OpenAI: payload["tools"] — list of tool objects
    """
    tools = payload.get("tools")
    if not tools:
        return ""
    return json.dumps(tools, sort_keys=True)


def _split_conversation(
    messages: list[dict],
) -> tuple[str, list[dict], str]:
    """Split messages into system_prompt, conversation_history, and user_message.

    Returns (system_prompt, history_messages, user_message).
    """
    system_prompt = ""
    user_message = ""
    history: list[dict] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)

        if role == "system" and not system_prompt:
            system_prompt = content
        elif i == len(messages) - 1 and role == "user":
            user_message = content
        else:
            history.append(msg)

    return system_prompt, history, user_message


def decompose(
    payload: dict,
    fingerprint_cache: FingerprintCache,
    *,
    api_format: str = "ollama",
) -> DecomposedPayload:
    """Decompose a chat payload into fingerprinted sections.

    Identifies: system_prompt, tools, workspace_files, conversation_history, user_message.
    Hashes each section and checks for changes against the cache.
    The payload is NOT modified — this is observation only.
    """
    start = time.perf_counter_ns()
    sections: list[Section] = []

    messages = payload.get("messages", [])

    # 1. Extract workspace file content from messages
    workspace_text, messages_clean = _extract_workspace_content(messages)

    # 2. Split remaining messages
    system_prompt, history, user_message = _split_conversation(messages_clean)

    # 3. Extract tools
    tools_text = _extract_tools_content(payload)

    # 4. Build sections
    if system_prompt:
        h = _hash_content(system_prompt)
        sections.append(Section(
            name="system_prompt",
            content=system_prompt,
            token_estimate=_estimate_tokens(system_prompt),
            hash=h,
            changed=fingerprint_cache.check_and_update("system_prompt", h),
        ))

    if tools_text:
        h = _hash_content(tools_text)
        sections.append(Section(
            name="tools",
            content=tools_text,
            token_estimate=_estimate_tokens(tools_text),
            hash=h,
            changed=fingerprint_cache.check_and_update("tools", h),
        ))

    if workspace_text:
        h = _hash_content(workspace_text)
        sections.append(Section(
            name="workspace_files",
            content=workspace_text,
            token_estimate=_estimate_tokens(workspace_text),
            hash=h,
            changed=fingerprint_cache.check_and_update("workspace_files", h),
        ))

    if history:
        history_text = json.dumps(history, sort_keys=True)
        h = _hash_content(history_text)
        sections.append(Section(
            name="conversation_history",
            content=history_text,
            token_estimate=_estimate_tokens(history_text),
            hash=h,
            changed=fingerprint_cache.check_and_update("conversation_history", h),
        ))

    if user_message:
        # User message is always "changed" — it's the new input
        h = _hash_content(user_message)
        sections.append(Section(
            name="user_message",
            content=user_message,
            token_estimate=_estimate_tokens(user_message),
            hash=h,
            changed=True,  # always new
        ))

    # 5. Capture any other top-level keys as "options"
    other_keys = {k: v for k, v in payload.items() if k not in ("messages", "tools")}
    if other_keys:
        other_text = json.dumps(other_keys, sort_keys=True)
        h = _hash_content(other_text)
        sections.append(Section(
            name="options",
            content=other_text,
            token_estimate=_estimate_tokens(other_text),
            hash=h,
            changed=fingerprint_cache.check_and_update("options", h),
        ))

    elapsed_us = (time.perf_counter_ns() - start) // 1000
    result = DecomposedPayload(
        sections=sections,
        format=api_format,
        decompose_time_us=elapsed_us,
    )

    result.log_breakdown()
    logger.debug("Decomposition took %dµs", elapsed_us)

    return result

"""Cycle 27 T10+T11: v2 context formatter.

Renders the 5-section template validated by the Step 1 ceiling probe:

    <cardinal rules header — pinned to profile_owner_name>

    CONTEXT
    [CURRENT SLOTS]
      - predicate: value    (category)
      ...
    [TIMELINE] — temporal queries only
      [T1 valid_from] content
      [T2 valid_from] content
      ...
    [RELATIONSHIPS] — multi-hop queries (or when any relationships present)
      - relationship_type → target_name (status)
      ...
    [NOT PRESENT]
      - <slot_key>
      ...

Unused sections are omitted. Section ordering is fixed. Token budgets
per bloat tier come from spec §9.2; callers pass their cap in via the
`max_tokens` kwarg.
"""
from __future__ import annotations

import logging
from typing import Iterable

from sieve.slot_retriever import SlotRetrievalResult

logger = logging.getLogger("recall.context_format_v2")


_CARDINAL_TEMPLATE = """\
You are answering questions about {owner_name}. All the facts below are \
about them — treat "{owner_name}" and "the user" as the same person.

Rules:
1. Answer only from the facts in the CONTEXT section below. If the answer \
is not there, say "I don't know" — do NOT guess.
2. When two facts conflict, trust the one marked [CURRENT] over any other. \
[PAST] facts describe previous states that no longer hold.
3. Do NOT invent people, places, jobs, or relationships that are not in \
the context.
4. If the question asks about someone or something marked [NOT PRESENT], \
say so directly — do not confabulate.
"""


def _rough_tokens(text: str) -> int:
    """Cheap token estimate: 1 token per 4 chars, rounded up."""
    return max(1, (len(text) + 3) // 4)


def _fmt_slot_line(row: dict) -> str:
    """Render a single current-slots row as one line."""
    pred = row.get("predicate") or "fact"
    cat = row.get("category") or ""
    content = row.get("content") or row.get("object_literal") or ""
    if content and pred != "fact":
        return f"  - [CURRENT] {pred}: {content.strip()}" + (f"  ({cat})" if cat else "")
    return f"  - [CURRENT] {content.strip()}"


def _fmt_timeline_row(row: dict, idx: int) -> str:
    vf = row.get("valid_from") or ""
    vt = row.get("valid_to")
    tag = f"[T{idx} {vf}]" if vf else f"[T{idx}]"
    if vt:
        tag = tag[:-1] + f" → {vt}]"
    content = (row.get("content") or row.get("object_literal") or "").strip()
    marker = "[PAST]" if vt else "[CURRENT]"
    return f"  {tag} {marker} {content}"


def _fmt_relationship_row(row: dict) -> str:
    rel = row.get("relationship") or "related_to"
    name = row.get("target_name") or row.get("target_entity") or "?"
    status = row.get("status") or "current"
    return f"  - {rel} → {name}  ({status})"


def _cardinal_header(owner_name: str) -> str:
    """Render the cardinal rules header pinned to the profile owner.

    When owner_name is empty, falls back to "the user".
    """
    name = owner_name or "the user"
    return _CARDINAL_TEMPLATE.format(owner_name=name)


def format_context_v2(
    result: SlotRetrievalResult,
    *,
    profile_owner_name: str,
    extra_facts: Iterable[dict] | None = None,
    max_tokens: int = 800,
) -> tuple[str, int]:
    """Render the v2 context block.

    Args:
        result: SlotRetrievalResult from SlotRetriever.
        profile_owner_name: canonical name for the cardinal header.
        extra_facts: optional legacy facts to supplement the [CURRENT SLOTS]
            section when the slot paths returned thin data. Each row should
            have 'content' at minimum.
        max_tokens: cap on the total rendered block (rough estimate at 4
            chars/token). Sections are truncated in reverse order of
            priority (not-present last, then relationships, then timeline,
            then extra_facts, then current_slots).

    Returns:
        (text_block, token_estimate).
    """
    header = _cardinal_header(profile_owner_name)

    # ── Build sections ────────────────────────────────────────────────
    current_lines: list[str] = []
    seen_contents: set[str] = set()
    for row in result.current_slots:
        content = (row.get("content") or "").strip().lower()
        if content in seen_contents:
            continue
        seen_contents.add(content)
        current_lines.append(_fmt_slot_line(row))

    extra_lines: list[str] = []
    if extra_facts:
        for row in extra_facts:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            key = content.lower()
            if key in seen_contents:
                continue
            seen_contents.add(key)
            extra_lines.append(f"  - {content}")

    timeline_lines: list[str] = []
    for i, row in enumerate(result.timeline, 1):
        timeline_lines.append(_fmt_timeline_row(row, i))

    relationship_lines: list[str] = []
    for row in result.relationships:
        relationship_lines.append(_fmt_relationship_row(row))

    not_present_lines: list[str] = []
    for slot in result.known_unknowns:
        not_present_lines.append(f"  - {slot}")

    # ── Assemble with truncation budget ───────────────────────────────
    parts: list[str] = [header.rstrip(), "", "CONTEXT"]

    def _add_section(title: str, lines: list[str]) -> None:
        if not lines:
            return
        parts.append(title)
        parts.extend(lines)

    _add_section("[CURRENT SLOTS]", current_lines)
    _add_section("[SUPPORTING FACTS]", extra_lines)
    _add_section("[TIMELINE]", timeline_lines)
    _add_section("[RELATIONSHIPS]", relationship_lines)
    _add_section("[NOT PRESENT]", not_present_lines)

    text = "\n".join(parts)
    tokens = _rough_tokens(text)

    # Budget truncation: trim sections bottom-up.
    if tokens > max_tokens:
        drop_order = [
            "[NOT PRESENT]",
            "[RELATIONSHIPS]",
            "[TIMELINE]",
            "[SUPPORTING FACTS]",
        ]
        for marker in drop_order:
            if tokens <= max_tokens:
                break
            if marker in parts:
                idx = parts.index(marker)
                # Drop the section header and everything until the next
                # section marker (or end of list).
                end = idx + 1
                while end < len(parts) and not parts[end].startswith("["):
                    end += 1
                del parts[idx:end]
                text = "\n".join(parts)
                tokens = _rough_tokens(text)

    return text, tokens

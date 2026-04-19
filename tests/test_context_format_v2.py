"""Cycle 27 T10+T11: context format v2 tests."""
from __future__ import annotations

import pytest

from sieve.context_format_v2 import format_context_v2
from sieve.slot_retriever import SlotRetrievalResult


def _current_slot(**kwargs) -> dict:
    row = {
        "id": "x",
        "content": "",
        "slot_key": "",
        "predicate": "",
        "object_literal": "",
        "valid_from": "",
        "category": "",
        "confidence": 0.9,
        "status": "current",
        "source_turn_id": None,
    }
    row.update(kwargs)
    return row


def test_cardinal_header_pins_profile_owner_name():
    result = SlotRetrievalResult(query="x")
    text, _ = format_context_v2(
        result, profile_owner_name="Mary Chen",
    )
    assert "Mary Chen" in text
    assert "the user" in text
    assert "Rules" in text
    assert "Answer only from the facts" in text


def test_cardinal_header_fallback_without_owner():
    result = SlotRetrievalResult(query="x")
    text, _ = format_context_v2(result, profile_owner_name="")
    assert "the user" in text


def test_current_slots_section_renders():
    result = SlotRetrievalResult(
        query="what's Mary's job",
        query_class="slot_lookup",
        slot_predicate="role",
        current_slots=[
            _current_slot(
                predicate="role",
                content="Mary Chen's role is VP of Product",
                category="employment",
            ),
            _current_slot(
                predicate="employer",
                content="Mary Chen works at Meridian Health",
                category="employment",
            ),
        ],
    )
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen")
    assert "[CURRENT SLOTS]" in text
    assert "VP of Product" in text
    assert "Meridian Health" in text
    assert "[CURRENT]" in text


def test_timeline_section_renders_with_past_markers():
    result = SlotRetrievalResult(
        query="how has Mary's career changed over time",
        query_class="temporal_sequence",
        timeline=[
            _current_slot(
                content="Mary Chen works at Nexus Health",
                valid_from="2024-01-01", valid_to="2026-04-01",
                object_literal="Nexus Health",
            ),
            _current_slot(
                content="Mary Chen works at Meridian Health",
                valid_from="2026-04-01", valid_to=None,
                object_literal="Meridian Health",
            ),
        ],
    )
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen")
    assert "[TIMELINE]" in text
    assert "[T1" in text
    assert "[T2" in text
    assert "[PAST]" in text    # first row has valid_to
    assert "[CURRENT]" in text  # second row has no valid_to
    assert "Nexus Health" in text
    assert "Meridian Health" in text


def test_relationships_section_renders():
    result = SlotRetrievalResult(
        query="who in her network",
        query_class="multi_hop",
        relationships=[
            {"relationship": "mentor", "target_name": "Derek Liu", "status": "current"},
            {"relationship": "reports_to", "target_name": "Eva Gupta", "status": "current"},
        ],
    )
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen")
    assert "[RELATIONSHIPS]" in text
    assert "mentor → Derek Liu" in text
    assert "reports_to → Eva Gupta" in text


def test_not_present_section_renders():
    result = SlotRetrievalResult(
        query="mother's name",
        query_class="generic",
        known_unknowns=["mary_chen:mother_first_name"],
    )
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen")
    assert "[NOT PRESENT]" in text
    assert "mary_chen:mother_first_name" in text


def test_empty_result_still_renders_header_only():
    result = SlotRetrievalResult(query="x")
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen")
    assert "Mary Chen" in text
    assert "CONTEXT" in text
    assert "[CURRENT SLOTS]" not in text
    assert "[TIMELINE]" not in text


def test_extra_facts_supplement_current_slots():
    result = SlotRetrievalResult(query="x", query_class="generic")
    extra = [
        {"content": "Mary Chen has twin sons Jake and Ethan"},
        {"content": "Mary Chen's housing budget is $3,200/month"},
    ]
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen", extra_facts=extra)
    assert "[SUPPORTING FACTS]" in text
    assert "Jake and Ethan" in text
    assert "$3,200/month" in text


def test_supporting_facts_dedup_against_current_slots():
    result = SlotRetrievalResult(
        query="x", query_class="slot_lookup",
        current_slots=[_current_slot(content="Mary Chen works at Meridian Health")],
    )
    extra = [{"content": "Mary Chen works at Meridian Health"}]
    text, _ = format_context_v2(result, profile_owner_name="Mary Chen", extra_facts=extra)
    # Should appear exactly once
    assert text.count("Mary Chen works at Meridian Health") == 1


def test_token_budget_truncates_bottom_up():
    # Build a huge result and cap at a small budget.
    result = SlotRetrievalResult(
        query="x",
        current_slots=[
            _current_slot(content=f"important current fact number {i}") for i in range(6)
        ],
        timeline=[
            _current_slot(
                content=f"huge padding timeline row {i} " * 20,
                valid_from="2023-01-01", valid_to=None,
            ) for i in range(10)
        ],
        relationships=[{"relationship": "friend", "target_name": f"Person{i}", "status": "current"} for i in range(10)],
        known_unknowns=[f"mary_chen:slot_{i}" for i in range(10)],
    )
    text, tokens = format_context_v2(result, profile_owner_name="Mary Chen", max_tokens=300)
    # Budget enforced modulo the non-truncatable cardinal header (~110 tok)
    # and current slots (highest priority).
    assert tokens <= 320
    # Cardinal header always present
    assert "Mary Chen" in text
    # CURRENT SLOTS preserved (highest priority)
    assert "[CURRENT SLOTS]" in text
    assert "important current fact number 0" in text
    # Bottom sections dropped (check for the section marker on its own
    # line — the phrase "[NOT PRESENT]" is also present in the header).
    assert "\n[NOT PRESENT]" not in text
    assert "\n[RELATIONSHIPS]" not in text
    assert "\n[TIMELINE]" not in text


def test_format_returns_token_estimate():
    result = SlotRetrievalResult(
        query="x",
        current_slots=[_current_slot(content="test fact")],
    )
    text, tokens = format_context_v2(result, profile_owner_name="Mary Chen")
    assert tokens > 0
    assert tokens == max(1, (len(text) + 3) // 4)

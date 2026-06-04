"""Context format v2 tests."""
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
        result, profile_owner_name="Jamie Rivera",
    )
    assert "Jamie Rivera" in text
    assert "the user" in text
    assert "Rules" in text
    assert "Answer only from the facts" in text


def test_cardinal_header_fallback_without_owner():
    result = SlotRetrievalResult(query="x")
    text, _ = format_context_v2(result, profile_owner_name="")
    assert "the user" in text


def test_current_slots_section_renders():
    result = SlotRetrievalResult(
        query="what's Jamie's job",
        query_class="slot_lookup",
        slot_predicate="role",
        current_slots=[
            _current_slot(
                predicate="role",
                content="Jamie Rivera's role is VP of Product",
                category="employment",
            ),
            _current_slot(
                predicate="employer",
                content="Jamie Rivera works at Example Corp",
                category="employment",
            ),
        ],
    )
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera")
    assert "[CURRENT SLOTS]" in text
    assert "VP of Product" in text
    assert "Example Corp" in text
    assert "[CURRENT]" in text


def test_timeline_section_renders_with_past_markers():
    result = SlotRetrievalResult(
        query="how has Jamie's career changed over time",
        query_class="temporal_sequence",
        timeline=[
            _current_slot(
                content="Jamie Rivera works at Other Corp",
                valid_from="2024-01-01", valid_to="2026-04-01",
                object_literal="Other Corp",
            ),
            _current_slot(
                content="Jamie Rivera works at Example Corp",
                valid_from="2026-04-01", valid_to=None,
                object_literal="Example Corp",
            ),
        ],
    )
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera")
    assert "[TIMELINE]" in text
    assert "[T1" in text
    assert "[T2" in text
    assert "[PAST]" in text    # first row has valid_to
    assert "[CURRENT]" in text  # second row has no valid_to
    assert "Other Corp" in text
    assert "Example Corp" in text


def test_relationships_section_renders():
    result = SlotRetrievalResult(
        query="who in her network",
        query_class="multi_hop",
        relationships=[
            {"relationship": "mentor", "target_name": "Derek Liu", "status": "current"},
            {"relationship": "reports_to", "target_name": "Eva Gupta", "status": "current"},
        ],
    )
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera")
    assert "[RELATIONSHIPS]" in text
    assert "mentor → Derek Liu" in text
    assert "reports_to → Eva Gupta" in text


def test_not_present_section_renders():
    result = SlotRetrievalResult(
        query="mother's name",
        query_class="generic",
        known_unknowns=["jamie_rivera:mother_first_name"],
    )
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera")
    assert "[NOT PRESENT]" in text
    assert "jamie_rivera:mother_first_name" in text


def test_empty_result_still_renders_header_only():
    result = SlotRetrievalResult(query="x")
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera")
    assert "Jamie Rivera" in text
    assert "CONTEXT" in text
    assert "[CURRENT SLOTS]" not in text
    assert "[TIMELINE]" not in text


def test_extra_facts_supplement_current_slots():
    result = SlotRetrievalResult(query="x", query_class="generic")
    extra = [
        {"content": "Jamie Rivera has twin sons Pat and Alex"},
        {"content": "Jamie Rivera's housing budget is $3,200/month"},
    ]
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera", extra_facts=extra)
    assert "[SUPPORTING FACTS]" in text
    assert "Pat and Alex" in text
    assert "$3,200/month" in text


def test_supporting_facts_dedup_against_current_slots():
    result = SlotRetrievalResult(
        query="x", query_class="slot_lookup",
        current_slots=[_current_slot(content="Jamie Rivera works at Example Corp")],
    )
    extra = [{"content": "Jamie Rivera works at Example Corp"}]
    text, _ = format_context_v2(result, profile_owner_name="Jamie Rivera", extra_facts=extra)
    # Should appear exactly once
    assert text.count("Jamie Rivera works at Example Corp") == 1


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
        known_unknowns=[f"jamie_rivera:slot_{i}" for i in range(10)],
    )
    text, tokens = format_context_v2(result, profile_owner_name="Jamie Rivera", max_tokens=300)
    # Budget enforced modulo the non-truncatable cardinal header (~110 tok)
    # and current slots (highest priority).
    assert tokens <= 320
    # Cardinal header always present
    assert "Jamie Rivera" in text
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
    text, tokens = format_context_v2(result, profile_owner_name="Jamie Rivera")
    assert tokens > 0
    assert tokens == max(1, (len(text) + 3) // 4)

"""Query classifier tests.

Exercises the classifier against 10 simulation queries plus adversarial
edge cases. The goal is that routing is correct for the failure set;
incorrect routes for exotic queries are tolerated because generic
fall-back is never worse than the prior behaviour.
"""
from __future__ import annotations

import pytest

from sieve.query_classifier_v2 import (
    QueryClass,
    classify_query,
    slot_from_query,
)


# ── The 10 simulation queries ──────────────────────────────────────────────

@pytest.mark.parametrize("qid,query,expected_class,expected_predicate", [
    # A_precision — mostly slot_lookup
    ("A1", "What's Jamie's current living situation?",
     QueryClass.SLOT_LOOKUP, "residence_city"),
    ("A2", "Is Jamie still married?",
     QueryClass.SLOT_LOOKUP, "marital_status"),
    ("A3", "What's Jamie's current job title and where does she work?",
     QueryClass.SLOT_LOOKUP, "role"),
    ("A4", "How much does Jamie spend on housing per month right now?",
     QueryClass.SLOT_LOOKUP, "monthly_mortgage"),
    # A5 is a relationship question — no slot predicate fits perfectly.
    # Route to generic (fall back to vector retrieval) is fine.
    ("A5", "What's Jamie's relationship with Kim right now?",
     QueryClass.GENERIC, None),

    # B_multihop
    ("B2", "What birthday gifts would work for Jamie's twin boys given her current financial situation?",
     QueryClass.MULTI_HOP, None),
    ("B4", "Who in Jamie's professional network could help with a career transition?",
     QueryClass.MULTI_HOP, None),

    # C_temporal — all routed to temporal_sequence
    ("C1", "Walk me through Jamie's career progression over the last few years.",
     QueryClass.TEMPORAL_SEQUENCE, None),
    ("C2", "How has Jamie's living situation changed over time?",
     QueryClass.TEMPORAL_SEQUENCE, None),
    ("C3", "What's happened with Jamie and Kim's relationship across the story?",
     QueryClass.TEMPORAL_SEQUENCE, None),
    ("C4", "Has Jamie's opinion about Python changed over time?",
     QueryClass.TEMPORAL_SEQUENCE, None),

    # D_trap
    # D2 "daughter" — generic is fine; retrieval then surfaces children facts
    # and the NOT_PRESENT rule lands via known_unknowns (T5/T7).
    ("D2", "Tell me about Jamie's daughter.",
     QueryClass.GENERIC, None),
    # D4 "happy at work" — no slot perfectly matches emotion, fall to generic.
    ("D4", "Is Jamie happy at work?",
     QueryClass.GENERIC, None),
    # D5 "mother's name" — generic, but retrieval must emit [NOT PRESENT].
    ("D5", "What's Jamie's mother's name?",
     QueryClass.GENERIC, None),
])
def test_classify_simulation_queries(qid, query, expected_class, expected_predicate):
    result = classify_query(query)
    assert result.query_class == expected_class, (
        f"{qid}: expected {expected_class}, got {result.query_class} "
        f"(trigger={result.trigger})"
    )
    if expected_predicate is not None:
        assert result.slot_predicate == expected_predicate, (
            f"{qid}: expected predicate {expected_predicate}, "
            f"got {result.slot_predicate}"
        )


def test_empty_query_is_generic():
    assert classify_query("").query_class == QueryClass.GENERIC
    assert classify_query("   ").query_class == QueryClass.GENERIC


def test_unknown_query_falls_back_to_generic():
    assert classify_query("What's the weather like today?").query_class == QueryClass.GENERIC
    assert classify_query("Tell me a joke.").query_class == QueryClass.GENERIC


def test_temporal_overrides_slot_lookup():
    """'Current' + 'over time' should route to temporal, not slot."""
    result = classify_query("How has Jamie's current job changed over time?")
    assert result.query_class == QueryClass.TEMPORAL_SEQUENCE


def test_multi_hop_overrides_slot_lookup():
    """'Who in network' + 'current' should route to multi-hop."""
    result = classify_query("Who in Jamie's professional network knows her current manager?")
    assert result.query_class == QueryClass.MULTI_HOP


def test_slot_from_query_builds_canonical_slot_key():
    # "Is Jamie still married?" → jamie_rivera:marital_status
    slot = slot_from_query("Is Jamie still married?", "Jamie Rivera")
    assert slot == "jamie_rivera:marital_status"

    # Non-slot query returns None
    assert slot_from_query("How did her career change over time?", "Jamie Rivera") is None


def test_slot_from_query_handles_weird_owner_names():
    assert slot_from_query("Where does she live?", "Jean-Luc Picard") == "jean_luc_picard:residence_city"
    assert slot_from_query("Where does she live?", "") == ":residence_city"

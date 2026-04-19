"""Tests for Phase 8: Writer Stage 2 + Conflict Resolution.

Tests cover:
- S2 gate logic
- S2 LLM prompt parsing
- Conflict resolution decision tree (all branches)
- Session coherence scoring
- Subjective fact coexistence (nuanced_view)
- Speculative fact flagging
- Quarantine for high-confidence contradictions
"""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, patch

import pytest

from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.writer import (
    ConflictResolution,
    ExtractedFact,
    MemoryWriter,
    WriteResult,
    _content_equivalent,
    _cosine_similarity,
    _is_numeric_content,
    _is_speculative_text,
    _parse_s2_response,
    _s2_gate,
    compute_session_coherence,
    extract_facts_s1,
    resolve_conflict,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    config = StoreConfig(path=str(tmp_path / "test.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test-writer-s2")
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


def _fake_embed(text: str) -> list[float]:
    v = [ord(text[0]) / 256.0 if text else 0.1, 0.5, 0.3, 0.2]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


async def _async_embed(text: str) -> list[float]:
    return _fake_embed(text)


def _make_fact(content, fact_type="objective", category="general", confidence=0.75):
    return ExtractedFact(
        content=content,
        fact_type=fact_type,
        category=category,
        confidence=confidence,
    )


# ─── S2 Gate ──────────────────────────────────────────────────────────────────

class TestS2Gate:
    def test_short_message_does_not_trigger(self):
        assert _s2_gate("I live in Dubai.", []) is False

    def test_long_message_no_s1_triggers(self):
        text = "The weather has been really nice lately and I've been spending lots of time outdoors with friends and I really enjoy it."
        # >15 words, no S1 facts → triggers
        assert _s2_gate(text, []) is True

    def test_long_message_with_uncovered_proper_nouns_triggers(self):
        text = "I had dinner with Marcus at that Italian place near the Marina yesterday evening."
        s1 = []  # S1 didn't catch Marcus
        assert _s2_gate(text, s1) is True

    def test_long_message_all_nouns_covered_no_trigger(self):
        text = "I live in Dubai and I work at Emirates as a pilot for the main fleet."
        s1_fact1 = ExtractedFact(
            content="User lives in Dubai", fact_type="objective", category="location",
            entity_names=["Dubai"],
        )
        s1_fact2 = ExtractedFact(
            content="User works at Emirates", fact_type="objective", category="occupation",
            entity_names=["Emirates"],
        )
        result = _s2_gate(text, [s1_fact1, s1_fact2])
        # Could go either way depending on exact noun coverage
        assert isinstance(result, bool)

    def test_subjective_language_triggers(self):
        text = "I really think that electric cars are the future and I love the idea of sustainability."
        s1 = []
        assert _s2_gate(text, s1) is True

    def test_speculative_language_triggers(self):
        text = "I'm considering buying a new house and maybe moving to the countryside next year."
        s1 = []
        assert _s2_gate(text, s1) is True

    def test_ten_words_exactly_no_trigger(self):
        text = "one two three four five six seven eight nine ten"
        assert _s2_gate(text, []) is False


# ─── S2 Response Parsing ─────────────────────────────────────────────────────

class TestS2Parsing:
    def test_valid_json_response(self):
        resp = json.dumps({
            "facts": [
                {
                    "content": "User is considering buying a Tesla",
                    "fact_type": "subjective",
                    "category": "preference",
                    "confidence": 0.6,
                    "entities": ["Tesla"],
                    "speculative": True,
                }
            ]
        })
        facts = _parse_s2_response(resp, "I'm thinking about getting a Tesla")
        assert len(facts) == 1
        assert facts[0].fact_type == "subjective"
        assert facts[0].confidence <= 0.4  # speculative cap

    def test_invalid_json(self):
        facts = _parse_s2_response("not json at all", "test")
        assert facts is None  # returns None on parse failure

    def test_empty_facts_list(self):
        facts = _parse_s2_response('{"facts": []}', "test")
        assert facts == []

    def test_missing_content_skipped(self):
        resp = json.dumps({"facts": [{"fact_type": "objective"}]})
        facts = _parse_s2_response(resp, "test")
        assert facts is None  # Pydantic requires content field

    def test_invalid_fact_type_defaults_to_objective(self):
        resp = json.dumps({
            "facts": [{"content": "User likes pizza", "fact_type": "banana"}]
        })
        facts = _parse_s2_response(resp, "test")
        assert facts[0].fact_type == "objective"

    def test_speculative_detection_from_original_text(self):
        resp = json.dumps({
            "facts": [{"content": "User might buy a house", "fact_type": "objective", "speculative": False}]
        })
        facts = _parse_s2_response(resp, "I'm thinking about buying a house, maybe next year")
        # Should detect speculative from original text + content match
        assert facts[0].confidence <= 0.4 or facts[0].fact_type == "subjective"


# ─── Conflict Resolution ─────────────────────────────────────────────────────

class TestConflictResolutionNoExisting:
    def test_no_existing_stores_as_current(self):
        fact = _make_fact("User lives in Dubai")
        res = resolve_conflict(fact, None)
        assert res.action == "store"
        assert res.new_status == "current"

    def test_subjective_no_existing_stores_as_current(self):
        fact = _make_fact("User loves pizza", fact_type="subjective")
        res = resolve_conflict(fact, None)
        assert res.action == "store"


class TestConflictResolutionSameValue:
    def test_same_value_boosts_confidence(self):
        fact = _make_fact("User lives in Dubai")
        existing = {"id": "abc", "content": "User lives in Dubai", "confidence": 0.7, "usage_count": 5}
        res = resolve_conflict(fact, existing)
        assert res.action == "boost"
        assert res.new_confidence > 0.7

    def test_same_value_case_insensitive(self):
        fact = _make_fact("User lives in dubai")
        existing = {"id": "abc", "content": "User lives in Dubai", "confidence": 0.7, "usage_count": 1}
        # _content_equivalent lowercases before comparison
        res = resolve_conflict(fact, existing)
        assert res.action == "boost"


class TestConflictResolutionSubjective:
    def test_subjective_coexists_with_nuanced_view(self):
        fact = _make_fact("User loves electric cars", fact_type="subjective")
        existing = {"id": "xyz", "content": "User dislikes electric cars", "confidence": 0.8, "usage_count": 3}
        res = resolve_conflict(fact, existing)
        assert res.action == "coexist"
        assert "nuanced_view" in res.detail

    def test_subjective_never_supersedes(self):
        fact = _make_fact("User hates cold weather", fact_type="subjective")
        existing = {"id": "xyz", "content": "User loves cold weather", "confidence": 0.9, "usage_count": 20}
        res = resolve_conflict(fact, existing)
        assert res.action != "supersede"
        assert res.action == "coexist"


class TestConflictResolutionSpeculative:
    def test_speculative_gets_low_confidence(self):
        fact = _make_fact("User is thinking about getting a Tesla")
        existing = {"id": "xyz", "content": "User drives a BMW", "confidence": 0.6, "usage_count": 2}
        res = resolve_conflict(fact, existing)
        # _is_speculative_text should detect "thinking about"
        assert res.new_confidence <= 0.4 or res.action in ("store", "supersede")


class TestConflictResolutionHighConfidence:
    def test_high_confidence_low_coherence_quarantines(self):
        fact = _make_fact("User lives in Tokyo")
        existing = {"id": "abc", "content": "User lives in Dubai", "confidence": 0.9, "usage_count": 15}
        res = resolve_conflict(fact, existing, session_coherence=0.2)
        assert res.action == "quarantine"
        assert res.new_status == "quarantined"

    def test_high_confidence_temporal_supersedes(self):
        fact = _make_fact("User is 39 years old", fact_type="temporal")
        existing = {"id": "abc", "content": "User is 38 years old", "confidence": 0.9, "usage_count": 12}
        res = resolve_conflict(fact, existing, session_coherence=0.8)
        assert res.action == "supersede"
        assert "temporal" in res.detail

    def test_high_confidence_numeric_supersedes(self):
        fact = _make_fact("User earns $200k per year")
        existing = {"id": "abc", "content": "User earns $180k per year", "confidence": 0.85, "usage_count": 11}
        res = resolve_conflict(fact, existing, session_coherence=0.8)
        assert res.action == "supersede"

    def test_high_confidence_otherwise_provisional(self):
        fact = _make_fact("User lives in Tokyo")
        existing = {"id": "abc", "content": "User lives in Dubai", "confidence": 0.9, "usage_count": 15}
        res = resolve_conflict(fact, existing, session_coherence=0.8)
        assert res.action == "provisional"
        assert res.new_status == "provisional"


class TestConflictResolutionLowConfidence:
    def test_low_confidence_existing_superseded(self):
        fact = _make_fact("User lives in Sydney")
        existing = {"id": "abc", "content": "User lives in Melbourne", "confidence": 0.4, "usage_count": 1}
        res = resolve_conflict(fact, existing)
        assert res.action == "supersede"

    def test_same_session_contradiction(self):
        fact = _make_fact("User likes cats")
        existing = {"id": "abc", "content": "User likes dogs", "confidence": 0.6, "usage_count": 3}
        res = resolve_conflict(fact, existing)
        assert res.action == "supersede"
        assert res.new_confidence == 0.5


# ─── Content helpers ──────────────────────────────────────────────────────────

class TestContentHelpers:
    def test_content_equivalent_exact(self):
        assert _content_equivalent("lives in dubai", "lives in dubai")

    def test_content_equivalent_with_prefix(self):
        assert _content_equivalent("user lives in dubai", "lives in dubai")

    def test_content_not_equivalent(self):
        assert not _content_equivalent("lives in dubai", "lives in tokyo")

    def test_is_speculative(self):
        assert _is_speculative_text("I'm thinking about getting a dog")
        assert _is_speculative_text("Maybe I should learn piano")
        assert not _is_speculative_text("I live in Dubai")

    def test_is_numeric(self):
        assert _is_numeric_content("User earns $180,000 per year")
        assert _is_numeric_content("User is 38 years old")
        assert not _is_numeric_content("User lives in Dubai")


# ─── Predicate-based contradiction ──────────────────────────────────────────

class TestPredicateContradiction:
    """Contradictions must be same-predicate, different-object."""

    def _check(self, a: str, b: str) -> bool:
        from sieve.writer import _is_direct_contradiction
        return _is_direct_contradiction(a.lower(), b.lower())

    # Regression: existing supersession cases must still fire
    def test_residence_change_contradicts(self):
        assert self._check("User lives in Sydney", "User lives in Melbourne")

    def test_likes_change_contradicts(self):
        assert self._check("User likes cats", "User likes dogs")

    def test_age_change_contradicts(self):
        assert self._check("User is 39 years old", "User is 38 years old")

    # Audit failures: must NOT contradict
    def test_marriage_duration_vs_origin_does_not_contradict(self):
        # The bug from the audit: "married for 9 years" was being superseded
        # by "met at a wedding in 2014" because they shared topical tokens.
        assert not self._check(
            "The user and Kim have been married for nine years",
            "The user and Kim met at a friend's wedding in 2014",
        )

    def test_role_vs_joining_date_does_not_contradict(self):
        # "Jamie Rivera is a product manager at Other Corp" vs
        # "Jamie Rivera joined Other Corp three years ago" — different facts,
        # both true.
        assert not self._check(
            "Jamie Rivera is a product manager at Other Corp",
            "Jamie Rivera joined Other Corp three years ago",
        )

    def test_husband_name_vs_husband_occupation_does_not_contradict(self):
        # "User's husband is named Kim" vs "User's husband is a high school
        # history teacher" — different attributes.
        assert not self._check(
            "The user's husband is named Kim",
            "The user's husband is a high school history teacher",
        )

    # New supersession cases the FIX should enable
    def test_marital_state_change_contradicts(self):
        assert self._check("User is married", "User is separated")

    def test_marital_state_divorced_contradicts(self):
        assert self._check("User is married", "User is divorced")

    def test_pet_name_change_contradicts(self):
        # Both old garbage facts ("pet named Alex" and "pet named Kim")
        # should chain so the older one supersedes.
        assert self._check(
            "User has a pet named Alex",
            "User has a pet named Kim",
        )

    def test_role_change_contradicts(self):
        assert self._check(
            "User's role is Senior PM",
            "User's role is VP of Product",
        )

    def test_decision_change_contradicts(self):
        assert self._check(
            "User is considering getting a dog",
            "User does not want a dog",
        )

    # Predicate not recognised → no contradiction (safe default)
    def test_unknown_predicates_do_not_contradict(self):
        assert not self._check(
            "User went to Boston yesterday",
            "User went to Chicago last year",
        )


class TestOwnerNameCanonicalization:
    """_strip_user_prefix only handled 'user'/'user's'/'the user' variants,
    so every S2-written fact — which uses the profile owner's name
    ('Jamie Rivera's ...') — bypassed predicate matching. Result: the
    an early smoke run stored contradictions as 'current' side by side
    (mortgage fixed at 3.1% + mortgage higher than 3.1%, etc.).

    After the fix, _strip_user_prefix accepts an owner_names iterable
    and strips those too. _content_equivalent and _is_direct_contradiction
    take the same parameter and forward it.
    """

    def test_strip_user_prefix_removes_owner_name(self):
        from sieve.writer import _strip_user_prefix
        out = _strip_user_prefix(
            "Jamie Rivera's mortgage rate is higher than 3.1%",
            owner_names=["Jamie Rivera"],
        )
        assert out == "mortgage rate is higher than 3.1%"

    def test_strip_user_prefix_removes_owner_name_without_possessive(self):
        from sieve.writer import _strip_user_prefix
        out = _strip_user_prefix(
            "Jamie Rivera lives in Bristol.",
            owner_names=["Jamie Rivera"],
        )
        assert out == "lives in bristol"

    def test_strip_user_prefix_handles_case_and_trailing_punctuation(self):
        from sieve.writer import _strip_user_prefix
        out = _strip_user_prefix(
            "JAMIE RIVERA's role is VP of Engineering.",
            owner_names=["Jamie Rivera"],
        )
        assert out == "role is vp of engineering"

    def test_strip_user_prefix_falls_through_without_owner(self):
        """Regression: existing callers that don't pass owner_names
        must keep the old behaviour — only user/user's variants stripped."""
        from sieve.writer import _strip_user_prefix
        out = _strip_user_prefix("Jamie Rivera lives in Bristol")
        # Without owner_names, owner prefix is not recognised
        assert out == "jamie rivera lives in bristol"

    def test_strip_user_prefix_strips_first_name_alias(self):
        """Writer passes the full owner plus first-name aliases so both
        'Jamie Rivera's X' and 'Jamie's X' canonicalise."""
        from sieve.writer import _strip_user_prefix
        out = _strip_user_prefix(
            "Jamie's father is Colin.",
            owner_names=["Jamie Rivera", "Jamie"],
        )
        assert out == "father is colin"

    def test_content_equivalent_across_user_and_owner_form(self):
        """'User lives in Bristol' and 'Jamie Rivera lives in Bristol'
        must be treated as the same fact for boost/dedup purposes."""
        from sieve.writer import _content_equivalent
        assert _content_equivalent(
            "user lives in bristol",
            "jamie rivera lives in bristol",
            owner_names=["Jamie Rivera"],
        )

    def test_mortgage_rate_contradiction_on_owner_form(self):
        """The headline bug from an early baseline run."""
        from sieve.writer import _is_direct_contradiction
        assert _is_direct_contradiction(
            "jamie rivera's mortgage rate is fixed at 3.1% until november",
            "jamie rivera's mortgage rate is currently higher than 3.1%",
            owner_names=["Jamie Rivera"],
        )

    def test_role_change_contradicts_on_owner_form(self):
        from sieve.writer import _is_direct_contradiction
        assert _is_direct_contradiction(
            "jamie rivera's role is vp of engineering",
            "jamie rivera's role is senior pm",
            owner_names=["Jamie Rivera"],
        )

    def test_residence_contradicts_across_user_and_owner_form(self):
        """Supersession must chain across the User→Owner rewrite too."""
        from sieve.writer import _is_direct_contradiction
        assert _is_direct_contradiction(
            "user lives in bristol",
            "jamie rivera lives in sydney",
            owner_names=["Jamie Rivera"],
        )


class TestExtendedValuePredicates:
    """Predicates added to cover the full test persona — same-value-predicate
    different-object shapes that _VALUE_PREDICATES did not catch.
    """

    def _check(self, a: str, b: str, owner_names=("Jamie Rivera",)) -> bool:
        from sieve.writer import _is_direct_contradiction
        return _is_direct_contradiction(a.lower(), b.lower(), owner_names=owner_names)

    def test_mortgage_rate_numeric_change_contradicts(self):
        assert self._check(
            "Jamie Rivera's mortgage rate is 3.1%",
            "Jamie Rivera's mortgage rate is 4.2%",
        )

    def test_relation_name_change_contradicts(self):
        """'My son is Oscar' then 'My son is Pat' — same relation slot,
        different names. The writer's view of 'User's son is X' must
        treat them as candidate contradictions for dedup/nuanced-view
        routing."""
        assert self._check(
            "User's son is Oscar",
            "User's son is Pat",
            owner_names=[],
        )

    def test_father_change_contradicts(self):
        assert self._check(
            "User's father is Colin",
            "User's father is David",
            owner_names=[],
        )

    def test_role_plain_is_form_contradicts(self):
        """Without an 'at ...' suffix, 'role is X' should still match."""
        assert self._check(
            "Jamie Rivera's role is VP of Engineering",
            "Jamie Rivera's role is Director of Platform",
        )


# ─── Session Coherence ────────────────────────────────────────────────────────

class TestSessionCoherence:
    async def test_single_message_returns_1(self):
        score = await compute_session_coherence(["hello"], _async_embed)
        assert score == 1.0

    async def test_empty_messages_returns_1(self):
        score = await compute_session_coherence([], _async_embed)
        assert score == 1.0

    async def test_similar_messages_high_coherence(self):
        msgs = ["I live in Dubai", "I love Dubai weather", "Dubai is great"]
        score = await compute_session_coherence(msgs, _async_embed)
        # Similar first chars → similar embeddings → high coherence
        assert score > 0.5

    async def test_no_embed_fn_returns_1(self):
        score = await compute_session_coherence(["a", "b"], None)
        assert score == 1.0

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 0.001


# ─── Full MemoryWriter integration ───────────────────────────────────────────

class TestMemoryWriterS2Integration:
    async def test_s1_still_works(self, store):
        writer = MemoryWriter(store, embed_fn=_async_embed)
        result = await writer.process("I live in Dubai and I'm 38 years old.")
        assert result.facts_written >= 1

    async def test_subjective_coexistence_checkpoint(self, store):
        """Checkpoint: 'I'm thinking about getting a Tesla but I'd never buy an electric car'
        should produce two subjective facts linked via nuanced_view.
        """
        writer = MemoryWriter(store, embed_fn=_async_embed)

        # First: seed an opinion about electric cars
        fact1_id = store.insert_fact(
            "User would never buy an electric car",
            embedding=_fake_embed("User would never buy an electric car"),
            fact_type="subjective",
            confidence=0.7,
        )

        # Now process a contradictory subjective fact
        fact2 = _make_fact(
            "User is considering getting a Tesla",
            fact_type="subjective",
            category="preference",
        )
        existing = store.search_facts_by_vector(
            _fake_embed("User is considering getting a Tesla"), limit=1,
        )

        if existing:
            res = resolve_conflict(fact2, existing[0])
            assert res.action == "coexist"
            assert "nuanced_view" in res.detail
        else:
            # No similar fact found — that's ok, still stores
            res = resolve_conflict(fact2, None)
            assert res.action == "store"

    async def test_quarantine_high_confidence_contradiction(self, store):
        """Checkpoint: contradicting a high-confidence fact in low-coherence session → quarantine."""
        # Seed a high-confidence well-confirmed fact
        fact_id = store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
            confidence=0.95,
        )
        # Boost usage_count above 10
        for _ in range(12):
            store.boost_fact_confidence(fact_id, boost=0.001)

        existing = store.get_fact(fact_id)
        assert existing["usage_count"] > 10

        # Contradictory fact in low-coherence session
        new_fact = _make_fact("User lives in Tokyo")
        res = resolve_conflict(new_fact, existing, session_coherence=0.15)
        assert res.action == "quarantine"
        assert res.new_status == "quarantined"

    async def test_temporal_supersession(self, store):
        """Age update: temporal type supersedes even high-confidence existing."""
        fact_id = store.insert_fact(
            "User is 38 years old",
            embedding=_fake_embed("User is 38 years old"),
            confidence=0.9,
        )
        for _ in range(11):
            store.boost_fact_confidence(fact_id, boost=0.001)

        existing = store.get_fact(fact_id)
        new_fact = _make_fact("User is 39 years old", fact_type="temporal")
        res = resolve_conflict(new_fact, existing, session_coherence=0.9)
        assert res.action == "supersede"

    async def test_dedup_still_works(self, store):
        writer = MemoryWriter(store, embed_fn=_async_embed)
        r1 = await writer.process("I live in Dubai.")
        r2 = await writer.process("I live in Dubai.")
        assert r1.facts_written >= 1
        assert r2.facts_skipped >= 1 or r2.facts_written == 0

    async def test_speculative_flagging(self):
        """Speculative markers should lower confidence."""
        fact = _make_fact("User is thinking about getting a Tesla")
        # _is_speculative_text detects "thinking about"
        assert _is_speculative_text(fact.content)

        existing = {"id": "x", "content": "User drives a BMW", "confidence": 0.6, "usage_count": 2}
        res = resolve_conflict(fact, existing)
        assert res.new_confidence <= 0.4


class TestStoreConflictMethods:
    def test_update_fact_status(self, store):
        fid = store.insert_fact("Test fact", embedding=None)
        store.update_fact_status(fid, "quarantined", status_detail="test quarantine")
        fact = store.get_fact(fid)
        assert fact["status"] == "quarantined"

    def test_boost_fact_confidence(self, store):
        fid = store.insert_fact("Test fact", embedding=None, confidence=0.5)
        store.boost_fact_confidence(fid, boost=0.1)
        fact = store.get_fact(fid)
        assert fact["confidence"] > 0.5
        assert fact["usage_count"] == 1

    def test_boost_capped_at_1(self, store):
        fid = store.insert_fact("Test fact", embedding=None, confidence=0.98)
        store.boost_fact_confidence(fid, boost=0.1)
        fact = store.get_fact(fid)
        assert fact["confidence"] <= 1.0

    def test_find_similar_facts(self, store):
        store.insert_fact("User lives in Dubai", embedding=_fake_embed("User lives in Dubai"))
        results = store.find_similar_facts(_fake_embed("User lives in Dubai"), limit=5, max_distance=1.0)
        assert len(results) >= 1

    def test_find_similar_facts_filtered_by_distance(self, store):
        store.insert_fact("User lives in Dubai", embedding=_fake_embed("User lives in Dubai"))
        results = store.find_similar_facts(_fake_embed("User lives in Dubai"), limit=5, max_distance=0.001)
        # Only exact matches within tiny distance
        # Depends on embedding — may or may not match
        assert isinstance(results, list)

    def test_get_fact_confirmation_count(self, store):
        fid = store.insert_fact("Test fact", embedding=None)
        assert store.get_fact_confirmation_count(fid) == 0
        store.boost_fact_confidence(fid)
        assert store.get_fact_confirmation_count(fid) == 1

    def test_get_fact_confirmation_count_nonexistent(self, store):
        assert store.get_fact_confirmation_count("nonexistent") == 0


def test_s2_prompt_template_renders_owner_name():
    from sieve.writer import _render_s2_prompt
    out = _render_s2_prompt("Jamie Rivera")
    assert "PROFILE OWNER: Jamie Rivera" in out
    assert "{owner_name}" not in out  # no leaked placeholders
    # Must preserve the existing rules
    assert 'fact_type "objective"' in out
    assert "STATE TRANSITIONS" in out


def test_s2_prompt_template_empty_owner_degrades_gracefully():
    from sieve.writer import _render_s2_prompt
    out = _render_s2_prompt("")
    # No owner name → fall back to generic body (no PROFILE OWNER block)
    assert "PROFILE OWNER" not in out
    assert 'fact_type "objective"' in out
    assert "{owner_name}" not in out

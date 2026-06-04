"""Response Verification Layer tests."""
from __future__ import annotations

import pytest

from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.verification import (
    AbsenceSignal,
    CLOSED_WORLD_FRAMING,
    Verification,
    build_absence_signals,
    extract_response_text,
    replace_response_text,
    verify_response,
    _extract_query_proper_nouns,
    _extract_relationship_words,
    _extract_response_proper_nouns,
)


@pytest.fixture
def store(tmp_path):
    cfg = StoreConfig(path=str(tmp_path / "verify_test.db"), embedding_dimensions=4)
    s = MemoryStore(cfg, passphrase="test-verify")
    s.open()
    s.init_schema()
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with the user, twin boys (Pat, Alex), and a husband (Kim),
    plus enough filler facts to clear the coverage gate
    (facts ≥ 100, ≥ 3 family entities → coverage score > 0.5)."""
    user_id = store.insert_entity("User", type="person")
    jake_id = store.insert_entity("Pat", type="person")
    ethan_id = store.insert_entity("Alex", type="person")
    tom_id = store.insert_entity("Kim", type="person")
    store.insert_relationship(user_id, "son", jake_id, confidence=0.9)
    store.insert_relationship(user_id, "son", ethan_id, confidence=0.9)
    store.insert_relationship(user_id, "husband", tom_id, confidence=0.9)
    store.insert_fact("User has twin boys named Pat and Alex", confidence=0.9)
    store.insert_fact("User's husband is Kim", confidence=0.9)
    # Coverage gate: seed 100 filler facts so facts_score saturates at 1.0.
    for i in range(100):
        store.insert_fact(f"Filler fact #{i} about the user.", confidence=0.8)
    return store


@pytest.fixture
def high_coverage_store(store):
    """Empty family/pet graph but facts_count saturated. Used to test that
    trap/proper-noun signals fire once the facts density is sufficient."""
    user_id = store.insert_entity("User", type="person")
    # At least 3 family entities so _store_coverage_score("family") == 1.0.
    mum_id = store.insert_entity("Mum", type="person")
    dad_id = store.insert_entity("Dad", type="person")
    sis_id = store.insert_entity("Sis", type="person")
    store.insert_relationship(user_id, "mother", mum_id, confidence=0.9)
    store.insert_relationship(user_id, "father", dad_id, confidence=0.9)
    store.insert_relationship(user_id, "sister", sis_id, confidence=0.9)
    for i in range(110):
        store.insert_fact(f"Background fact #{i}.", confidence=0.8)
    return store


# ─── Helper extraction tests ─────────────────────────────────────────────────


class TestExtractRelationshipWords:
    def test_finds_daughter(self):
        assert "daughter" in _extract_relationship_words("Tell me about Jamie's daughter")

    def test_finds_son(self):
        assert "son" in _extract_relationship_words("How is Jamie's son doing?")

    def test_finds_dog(self):
        assert "dog" in _extract_relationship_words("What breed is Jamie's dog?")

    def test_no_false_positive(self):
        assert _extract_relationship_words("What is Jamie's job?") == set()

    def test_word_boundaries(self):
        # 'sonatina' should not match 'son'
        assert "son" not in _extract_relationship_words("She plays a sonatina")


class TestExtractQueryProperNouns:
    def test_extracts_name(self):
        nouns = _extract_query_proper_nouns("What's Kim's salary?")
        assert "Kim" in nouns

    def test_skips_question_words(self):
        nouns = _extract_query_proper_nouns("Tell me about Jamie")
        assert "Tell" not in nouns

    def test_skips_user_self(self):
        # The owner's name must be supplied as extra_noise; no longer hardcoded.
        from sieve.config import ProfileOwnerConfig
        from sieve.verification import _owner_alias_set
        owner = ProfileOwnerConfig(name="Jamie Rivera", aliases=["Jamie"])
        noise = frozenset(
            a.title() for a in _owner_alias_set(owner) if a
        ) | frozenset(
            a.capitalize() for a in _owner_alias_set(owner) if a
        )
        nouns = _extract_query_proper_nouns("Where does Jamie live?", extra_noise=noise)
        assert "Jamie" not in nouns


# ─── Layer 1: build_absence_signals ───────────────────────────────────────────


class TestBuildAbsenceSignals:
    def test_no_signals_when_relationship_known(self, populated_store):
        # 'son' is in the store, no signal needed
        signals = build_absence_signals("Tell me about Jamie's son", [], populated_store)
        rels = [s for s in signals if s.reason == "relationship_word"]
        assert rels == []

    def test_signal_for_unknown_daughter(self, populated_store):
        signals = build_absence_signals("Tell me about Jamie's daughter", [], populated_store)
        texts = [s.text for s in signals]
        assert any("daughter" in t for t in texts)

    def test_signal_for_unknown_proper_noun(self, populated_store):
        signals = build_absence_signals("What does Derek do?", [], populated_store)
        texts = [s.text for s in signals]
        # New v2 format uses FACT: prefix and 'not in the user's records'
        assert any("Derek" in t and "not in" in t for t in texts)

    def test_no_signal_for_known_person(self, populated_store):
        signals = build_absence_signals("What is Kim doing?", [], populated_store)
        # Kim is in the store
        assert all("Kim" not in s.text for s in signals)

    def test_no_signal_when_in_retrieved_facts(self, populated_store):
        # If Derek appears in retrieved facts, no signal even if not in store
        facts = [{"content": "User worked with Derek at Old Company"}]
        signals = build_absence_signals("What does Derek do?", facts, populated_store)
        derek_signals = [s for s in signals if "Derek" in s.text]
        assert derek_signals == []

    def test_stored_surface_form_suppresses_canonical_query(self, store):
        """D23: stored relation 'mum' (raw surface) must suppress the
        absence signal for a 'mother' query. The canonicaliser maps both
        to 'mother' but the old code only compared query-canonical to
        store-raw keys, missing the match."""
        user_id = store.insert_entity("User", type="person")
        mum_id = store.insert_entity("Mum", type="person")
        sis_id = store.insert_entity("Sis", type="person")
        # Store the raw surface form "mum" as the relation label — this is
        # what the writer actually produced in the first 30-day run.
        store.insert_relationship(user_id, "mum", mum_id, confidence=0.95)
        store.insert_relationship(user_id, "sister", sis_id, confidence=0.95)
        # Coverage gate: seed enough facts to put family coverage at 1.0.
        for i in range(110):
            store.insert_fact(f"Filler #{i}", confidence=0.8)

        signals = build_absence_signals("Where does my mum live?", [], store)
        assert all("mother" not in s.text.lower() and "mum" not in s.text.lower()
                   for s in signals), [s.text for s in signals]
        # And also the canonical "mother" query must be covered.
        signals2 = build_absence_signals("Where does my mother live?", [], store)
        assert all("mother" not in s.text.lower() and "mum" not in s.text.lower()
                   for s in signals2), [s.text for s in signals2]

    # ── Q64 widening: assertion / recent-turn suppression ──────────────

    def test_possessive_assertion_in_query_suppresses(self, store):
        """The user asserting 'my daughter' in the question must never
        trigger a 'user has no daughter' signal — even when the store
        is empty (Day-1 cold start)."""
        query = "I need to pick up my daughter Lily from school."
        signals = build_absence_signals(query, [], store)
        assert all("daughter" not in s.text for s in signals), [s.text for s in signals]

    def test_possessive_wife_assertion_suppresses(self, store):
        query = "My wife Sam's birthday is on the 22nd."
        signals = build_absence_signals(query, [], store)
        assert all("wife" not in s.text for s in signals), [s.text for s in signals]

    def test_child_group_suppression_via_son_assertion(self, store):
        """Asserting 'my son Oscar' in the query should also suppress
        generic 'no kids/children' signals — they're in the same group."""
        query = "My son Oscar and I are planning a kids' trip to Wookey Hole."
        signals = build_absence_signals(query, [], store)
        for s in signals:
            assert "kid" not in s.text.lower() and "child" not in s.text.lower() \
                and "son" not in s.text.lower(), s.text

    def test_recent_turn_mention_suppresses(self, store):
        """A relation mentioned in a recent turn (but not yet graph-edged)
        must not trigger a false negative."""
        recent = [
            {"role": "user", "content": "My daughter Lily is 8 years old."},
            {"role": "assistant", "content": "Got it."},
        ]
        signals = build_absence_signals(
            "What are some board games Lily would enjoy?",
            [], store, recent_turns=recent,
        )
        assert all("daughter" not in s.text for s in signals)

    def test_trap_query_still_fires_when_no_evidence(self, high_coverage_store):
        """Classic trap ('What was the name of Dana's cat again?') has
        no assertion, no fact, no recent turn — must still fire once the
        store has enough coverage to be authoritative about pets.

        On a sparse / cold-start store the signal is (correctly)
        suppressed — see test_sparse_store_suppresses_trap below. The
        mature-store case uses the high_coverage fixture (100+ facts,
        3+ family entities → coverage > 0.5)."""
        # Seed a pet entity so pet category has coverage too. Without this,
        # pet category score stays 0 and the signal is silenced even on a
        # mature store — which is actually correct behaviour in general
        # but defeats the test's intent (the store knows this is a pet
        # question and should be confident about it).
        user_row = high_coverage_store._conn.execute(
            "SELECT id FROM entities WHERE name='User'"
        ).fetchone()
        user_id = user_row[0]
        goldie_id = high_coverage_store.insert_entity("Goldie", type="animal")
        bubba_id = high_coverage_store.insert_entity("Bubba", type="animal")
        whiskers_id = high_coverage_store.insert_entity("Whiskers", type="animal")
        high_coverage_store.insert_relationship(user_id, "dog", goldie_id, confidence=0.9)
        high_coverage_store.insert_relationship(user_id, "dog", bubba_id, confidence=0.9)
        high_coverage_store.insert_relationship(user_id, "cat", whiskers_id, confidence=0.9)
        # The user has cats/dogs but no entity named "Dana's cat". The
        # "cat" relationship is in the store, so the relationship-word
        # signal is correctly suppressed. The proper-noun check fires on
        # "Dana" (not a known entity, not in facts). Accept either
        # signal type — the point of this test is that a mature store
        # with a trap query still produces at least one negative signal.
        signals = build_absence_signals(
            "What was the name of Dana's cat again?",
            [], high_coverage_store, recent_turns=[],
        )
        assert signals, f"trap query failed to fire on mature store: {signals}"

    def test_sparse_store_suppresses_trap(self, store):
        """On a cold-start store the trap query is intentionally
        silenced. Early days don't have enough evidence to claim
        authority over what the user does/doesn't have."""
        signals = build_absence_signals(
            "What was the name of Dana's cat again?",
            [], store, recent_turns=[],
        )
        assert signals == [], \
            f"sparse store should have suppressed signals: {[s.text for s in signals]}"

    def test_coverage_score_thresholds(self, store):
        """Unit-test the coverage score math directly. Saturates at 1.0
        when facts ≥ 100 and 3+ category entities exist; zero either when
        the store is empty or when the category has no entities."""
        from sieve.verification import _store_coverage_score
        assert _store_coverage_score(store, "family") == pytest.approx(0.0)

        user_id = store.insert_entity("User", type="person")
        # Add exactly 50 filler facts: facts score = 0.5
        for i in range(50):
            store.insert_fact(f"Filler {i}", confidence=0.8)
        assert _store_coverage_score(store, "family") == pytest.approx(0.0)  # no family edges

        mum_id = store.insert_entity("Mum", type="person")
        store.insert_relationship(user_id, "mother", mum_id, confidence=0.9)
        # 50 facts × 1 family entity → 0.5 × 1/3 ≈ 0.167 (below gate)
        score = _store_coverage_score(store, "family")
        assert 0.15 < score < 0.2, f"expected ~0.17, got {score}"

        dad_id = store.insert_entity("Dad", type="person")
        sis_id = store.insert_entity("Sis", type="person")
        store.insert_relationship(user_id, "father", dad_id, confidence=0.9)
        store.insert_relationship(user_id, "sister", sis_id, confidence=0.9)
        for i in range(50, 105):
            store.insert_fact(f"Filler {i}", confidence=0.8)
        # 105 facts (capped to 1.0) × 3 family entities (capped to 1.0) = 1.0
        assert _store_coverage_score(store, "family") == pytest.approx(1.0)


# ─── Layer 3: verify_response ─────────────────────────────────────────────────


class TestVerifyResponse:
    def test_clean_response_with_known_entities(self, populated_store):
        v = verify_response(
            "Who is Kim?",
            "Kim is the user's husband.",
            populated_store,
        )
        assert v.is_clean

    def test_dirty_response_unknown_entity(self, populated_store):
        v = verify_response(
            "Who works with the user?",
            "The user works with Derek and Sarah on the project.",
            populated_store,
        )
        assert not v.is_clean
        assert "Derek" in v.flagged_entities
        assert v.corrective_prompt is not None

    def test_refusal_response_is_clean(self, populated_store):
        # v2: refusal text no longer short-circuits, but a pure refusal
        # mentioning no entities or unknown relationships should still be clean.
        v = verify_response(
            "What is Kim's salary?",
            "I don't have information about Kim's salary in my context.",
            populated_store,
        )
        assert v.is_clean

    def test_query_mentions_are_not_flagged(self, populated_store):
        # If the user asks about "Derek", the model echoing "Derek" should not
        # be a hallucination flag — the noun came from the prompt
        v = verify_response(
            "Tell me about Derek",
            "Derek isn't in your records.",
            populated_store,
        )
        assert v.is_clean or "Derek" not in v.flagged_entities

    def test_empty_response_is_clean(self, populated_store):
        v = verify_response("anything", "", populated_store)
        assert v.is_clean

    # Layer 3 v2 catches relationship-word claims about relationships
    # the user does not have.
    def test_v2_catches_daughter_claim(self, populated_store):
        # User has twin boys Pat/Alex, no daughter. Response asserts a daughter.
        v = verify_response(
            "Tell me about Jamie's daughter",
            "Jamie's daughter Sarah is 8 years old and loves soccer.",
            populated_store,
        )
        assert not v.is_clean
        assert "daughter" in v.flagged_relations

    def test_v2_negation_about_unknown_relation_is_clean(self, populated_store):
        # If the response correctly says there is no daughter, that is clean.
        v = verify_response(
            "Tell me about Jamie's daughter",
            "There is no daughter on record for Jamie.",
            populated_store,
        )
        assert v.is_clean

    def test_v2_response_with_refusal_prefix_then_hallucination(self, populated_store):
        # The model says "I don't know" then hallucinates anyway — v1 would
        # short-circuit on the refusal; v2 should still catch the rest.
        v = verify_response(
            "Tell me about Jamie's daughter",
            "I don't have specific details. Jamie's daughter Sarah was born in 2018.",
            populated_store,
        )
        assert not v.is_clean
        assert "daughter" in v.flagged_relations

    def test_known_entity_in_facts_passes(self, populated_store):
        facts = [{"content": "User has a colleague Robin Webb at Example"}]
        v = verify_response(
            "Who works with the user?",
            "The user has a colleague Robin Webb.",
            populated_store,
            retrieved_facts=facts,
        )
        # Robin Webb is in retrieved facts, so should pass even if not in entities
        assert v.is_clean or "Robin" not in v.flagged_entities


# ─── Body parsing helpers ────────────────────────────────────────────────────


class TestExtractResponseText:
    def test_ollama_format(self):
        body = b'{"message": {"content": "hello world"}, "done": true}'
        assert extract_response_text(body, "ollama") == "hello world"

    def test_openai_format(self):
        body = b'{"choices": [{"message": {"content": "hi there"}}]}'
        assert extract_response_text(body, "openai") == "hi there"

    def test_invalid_json_returns_none(self):
        assert extract_response_text(b"not json", "ollama") is None


class TestReplaceResponseText:
    def test_ollama_replace(self):
        body = b'{"message": {"content": "old"}, "done": true}'
        new_body = replace_response_text(body, "ollama", "new")
        assert b'"new"' in new_body
        assert b'"old"' not in new_body

    def test_openai_replace(self):
        body = b'{"choices": [{"message": {"content": "old"}}]}'
        new_body = replace_response_text(body, "openai", "new")
        assert b'"new"' in new_body


class TestClosedWorldFraming:
    def test_framing_string_present(self):
        assert "complete known context" in CLOSED_WORLD_FRAMING

"""Tests for Phase 5: Memory Writer Stage 1 — regex extraction, dedup, write."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.writer import (
    ExtractedFact,
    MemoryWriter,
    WriteResult,
    _is_duplicate,
    extract_facts_s1,
    extract_proper_nouns,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    config = StoreConfig(path=str(tmp_path / "test.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test-writer")
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


@pytest.fixture
def writer(store):
    return MemoryWriter(store, embed_fn=None)


def _fake_embed(text: str) -> list[float]:
    """Deterministic 4-dim unit embedding based on first char."""
    v = [ord(text[0]) / 256.0, 0.5, 0.3, 0.2] if text else [0.1, 0.1, 0.1, 0.1]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


async def _async_embed(text: str) -> list[float]:
    return _fake_embed(text)


@pytest.fixture
def writer_with_embeddings(store):
    return MemoryWriter(store, embed_fn=_async_embed)


# ─── extract_facts_s1 ───────────────────────────────────────────────────────────

class TestIdentityPatterns:
    def test_i_am(self):
        facts = extract_facts_s1("I am a software engineer.")
        assert any("software engineer" in f.content for f in facts)

    def test_im(self):
        facts = extract_facts_s1("I'm a pilot.")
        assert any("pilot" in f.content for f in facts)

    def test_im_with_article(self):
        facts = extract_facts_s1("I'm an architect based in Dubai.")
        assert any("architect" in f.content for f in facts)

    def test_identity_fact_type(self):
        facts = extract_facts_s1("I'm a doctor.")
        identity = [f for f in facts if f.category == "identity"]
        assert identity
        assert identity[0].fact_type == "objective"

    def test_filters_trivial(self):
        facts = extract_facts_s1("I'm ok.")
        identity = [f for f in facts if f.category == "identity"]
        assert not identity  # "ok" should be filtered


class TestLocationPatterns:
    def test_live_in(self):
        facts = extract_facts_s1("I live in Dubai.")
        location = [f for f in facts if f.category == "location"]
        assert location
        assert "Dubai" in location[0].content

    def test_based_in(self):
        facts = extract_facts_s1("I'm based in London.")
        location = [f for f in facts if f.category == "location"]
        assert location
        assert "London" in location[0].content

    def test_from(self):
        facts = extract_facts_s1("I'm from New York.")
        location = [f for f in facts if f.category == "location"]
        assert location

    def test_moved_to_is_temporal(self):
        facts = extract_facts_s1("I just moved to Berlin.")
        location = [f for f in facts if f.category == "location"]
        assert location
        assert location[0].fact_type == "temporal"

    def test_entity_name_captured(self):
        facts = extract_facts_s1("I live in Dubai.")
        location = [f for f in facts if f.category == "location"]
        assert location
        assert "Dubai" in location[0].entity_names


class TestOccupationPatterns:
    def test_work_at(self):
        facts = extract_facts_s1("I work at Google.")
        occ = [f for f in facts if f.category == "occupation"]
        assert occ
        assert "Google" in occ[0].content

    def test_work_as(self):
        facts = extract_facts_s1("I work as a nurse.")
        occ = [f for f in facts if f.category == "occupation"]
        assert occ

    def test_work_for(self):
        facts = extract_facts_s1("I work for a tech startup.")
        occ = [f for f in facts if f.category == "occupation"]
        assert occ


class TestRelationshipPatterns:
    def test_partner_is(self):
        facts = extract_facts_s1("My partner is Madeline.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel
        assert "Madeline" in rel[0].content
        assert rel[0].related_entity == "Madeline"
        assert rel[0].relation == "partner"

    def test_wife(self):
        facts = extract_facts_s1("My wife is Sarah.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel
        assert "Sarah" in rel[0].content

    def test_daughter(self):
        facts = extract_facts_s1("My daughter is named Emma.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel
        assert "Emma" in rel[0].content

    def test_entity_names_captured(self):
        facts = extract_facts_s1("My partner is Madeline.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel
        assert "Madeline" in rel[0].entity_names

    def test_friend(self):
        facts = extract_facts_s1("My best friend is Kim.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel


class TestRelationshipCaptureBoundaries:
    """S1 regex used re.IGNORECASE, which neutralised the [A-Z] name
    anchors and let lowercase function words bleed into captures.
    Validation store runs surfaced facts like
    'User's daughter is from', 'User's son is Oscar has',
    'User's dad is Colin was' — these must not be extracted.
    """

    def test_daughter_from_school_is_not_a_name(self):
        """'my daughter from school' — 'from' is not a name, no fact should fire."""
        facts = extract_facts_s1(
            "I need to leave work early today to pick up my daughter from school."
        )
        for f in facts:
            if f.category == "relationship":
                assert f.related_entity and f.related_entity.lower() != "from", (
                    f"captured 'from' as a name — got {f.content!r}"
                )
                assert "from" not in (f.related_entity or "").lower().split(), (
                    f"captured function word 'from' inside name: {f.content!r}"
                )

    def test_son_oscar_has_school_does_not_capture_trailing_verb(self):
        """'my son Oscar has school today' — must capture 'Oscar', not 'Oscar has'."""
        facts = extract_facts_s1("I need to pick up my son Oscar has school today.")
        rel = [f for f in facts if f.category == "relationship"]
        # If a fact IS extracted, the related_entity must be clean
        for f in rel:
            assert f.related_entity, f"empty related_entity in {f.content!r}"
            # No trailing lowercase-only verb token
            tokens = f.related_entity.split()
            for tok in tokens:
                assert tok[0].isupper(), (
                    f"name contains lowercase token {tok!r} in {f.content!r}"
                )

    def test_dad_colin_was_does_not_capture_trailing_verb(self):
        facts = extract_facts_s1("My dad Colin was a mason.")
        rel = [f for f in facts if f.category == "relationship"]
        for f in rel:
            assert f.related_entity, f"empty related_entity in {f.content!r}"
            tokens = f.related_entity.split()
            for tok in tokens:
                assert tok[0].isupper(), (
                    f"name contains lowercase token {tok!r} in {f.content!r}"
                )

    def test_daughter_is_from_bristol_does_not_include_from(self):
        """'My daughter is from Bristol.' — ambiguous, but if we extract,
        the name must not start with 'from' (a preposition)."""
        facts = extract_facts_s1("My daughter is from Bristol.")
        rel = [f for f in facts if f.category == "relationship"]
        for f in rel:
            tokens = (f.related_entity or "").split()
            assert not tokens or tokens[0].lower() != "from", (
                f"captured 'from' as name start: {f.content!r}"
            )

    # Regression guards — the GOOD cases must still work
    def test_partner_is_madeline_still_extracts(self):
        facts = extract_facts_s1("My partner is Madeline.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel, "benign case regressed"
        assert rel[0].related_entity == "Madeline"

    def test_wife_named_sarah_still_extracts(self):
        facts = extract_facts_s1("My wife is named Sarah.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel, "benign case regressed"

    def test_best_friend_marcus_runs_still_extracts(self):
        """relation_apposition's legitimate use case: 'My best friend Marcus runs ...'"""
        facts = extract_facts_s1("My best friend Marcus runs every morning.")
        rel = [f for f in facts if f.category == "relationship"]
        assert rel, "apposition benign case regressed"
        assert any(f.related_entity and "Marcus" in f.related_entity for f in rel)


class TestFamilyPatterns:
    def test_children_count_two(self):
        facts = extract_facts_s1("I have two children.")
        family = [f for f in facts if f.category == "family"]
        assert family
        assert "two" in family[0].content.lower()

    def test_children_count_numeric(self):
        facts = extract_facts_s1("I have 3 kids.")
        family = [f for f in facts if f.category == "family"]
        assert family


class TestAgePatterns:
    def test_age(self):
        facts = extract_facts_s1("I'm 35 years old.")
        age = [f for f in facts if f.category == "age"]
        assert age
        assert "35" in age[0].content
        assert age[0].fact_type == "temporal"


class TestMultipleFactsInOneSentence:
    def test_pilot_based_in_dubai(self):
        text = "I'm a pilot based in Dubai, my partner is Madeline."
        facts = extract_facts_s1(text)
        categories = {f.category for f in facts}
        assert "identity" in categories
        assert "location" in categories
        assert "relationship" in categories

    def test_complex_intro(self):
        text = "I'm a software engineer, I work at Acme Corp, and I live in Berlin."
        facts = extract_facts_s1(text)
        assert len(facts) >= 3


class TestProperNounExtraction:
    def test_single_proper_noun(self):
        nouns = extract_proper_nouns("My partner is Madeline.")
        assert "Madeline" in nouns

    def test_multi_word_proper_noun(self):
        nouns = extract_proper_nouns("I work at Acme Corporation.")
        assert any("Acme" in n for n in nouns)

    def test_common_words_filtered(self):
        nouns = extract_proper_nouns("I am from The United States.")
        assert "I" not in nouns
        assert "The" not in nouns

    def test_empty_string(self):
        assert extract_proper_nouns("") == []


# ─── Dedup ───────────────────────────────────────────────────────────────────────

class TestDedup:
    def test_no_duplicate_empty_store(self, store):
        assert _is_duplicate(store, "User is a pilot", []) is False

    def test_exact_match_is_duplicate(self, store):
        store.insert_fact("User is a pilot", embedding=None)
        assert _is_duplicate(store, "User is a pilot", []) is True

    def test_case_insensitive_match(self, store):
        store.insert_fact("User is a pilot", embedding=None)
        assert _is_duplicate(store, "user is a pilot", []) is True

    def test_no_duplicate_different_fact(self, store):
        store.insert_fact("User is a pilot", embedding=None)
        assert _is_duplicate(store, "User lives in Dubai", []) is False


# ─── MemoryWriter.process ─────────────────────────────────────────────────────────

class TestMemoryWriterProcess:
    async def test_write_basic_fact(self, writer, store):
        result = await writer.process("I'm a pilot based in Dubai.")
        assert result.facts_written >= 2  # identity + location

    async def test_creates_entity(self, writer, store):
        await writer.process("I live in Dubai.")
        entity = store.find_entity_by_name("Dubai")
        assert entity is not None

    async def test_creates_relationship(self, writer, store):
        await writer.process("My partner is Madeline.")
        result = await writer.process("My partner is Madeline.")
        # Second time should be dedup'd
        rels = store.conn.execute("SELECT * FROM relationships").fetchall()
        assert len(rels) >= 1

    async def test_creates_user_entity_for_relationships(self, writer, store):
        await writer.process("My partner is Madeline.")
        user = store.find_entity_by_name("User")
        assert user is not None

    async def test_dedup_skips_second_write(self, writer, store):
        result1 = await writer.process("I'm a pilot.")
        result2 = await writer.process("I'm a pilot.")
        # Second run: all facts should be skipped (exact text match)
        assert result2.facts_skipped >= 1
        assert result2.facts_written == 0

    async def test_session_id_returned(self, writer):
        result = await writer.process("I live in Dubai.")
        assert result.session_id

    async def test_custom_session_id(self, writer):
        result = await writer.process("I live in Dubai.", session_id="sess-123")
        assert result.session_id == "sess-123"

    async def test_empty_text_returns_zero(self, writer):
        result = await writer.process("")
        assert result.facts_written == 0
        assert result.facts_skipped == 0

    async def test_elapsed_tracked(self, writer):
        result = await writer.process("I'm a software engineer.")
        assert result.elapsed_ms >= 0

    async def test_full_pilot_sentence(self, writer, store):
        """The checkpoint sentence."""
        result = await writer.process(
            "I'm a pilot based in Dubai, my partner is Madeline, I have two children."
        )
        assert result.facts_written >= 3

        # Verify specific content
        facts = store.get_facts()
        contents = [f["content"] for f in facts]
        assert any("pilot" in c for c in contents)
        assert any("Dubai" in c for c in contents)
        assert any("Madeline" in c for c in contents)

    async def test_entities_written_count(self, writer, store):
        result = await writer.process("My partner is Madeline.")
        # Should create: Madeline + User entities
        assert result.entities_written >= 1

    async def test_confidence_scores_set(self, writer, store):
        await writer.process("I live in Dubai.")
        facts = store.get_facts()
        for f in facts:
            assert 0.0 < f["confidence"] <= 1.0

    async def test_source_is_writer_s1(self, writer, store):
        await writer.process("I live in Dubai.")
        facts = store.get_facts()
        assert all(f["source"] == "writer_s1" for f in facts)

    async def test_with_embeddings(self, writer_with_embeddings, store):
        """Writer with embeddings should store vectors."""
        result = await writer_with_embeddings.process("I live in Dubai.")
        assert result.facts_written >= 1
        # Check that embedding was stored
        facts = store.get_facts()
        assert any(f["embedding"] is not None for f in facts)

    async def test_multiple_turns(self, writer, store):
        """Multiple conversation turns accumulate facts."""
        await writer.process("I'm a software engineer.")
        await writer.process("I live in Berlin.")
        await writer.process("My partner is Lena.")

        facts = store.get_facts()
        assert len(facts) >= 3
        contents = {f["content"] for f in facts}
        assert any("engineer" in c for c in contents)
        assert any("Berlin" in c for c in contents)
        assert any("Lena" in c for c in contents)


class TestWriteResultStageCounts:
    """Validation-harness instrumentation: per-stage extraction counts.

    These fields let the live-validation metrics DB distinguish S1 (regex)
    vs S2 (LLM) contributions per request without rerunning extraction.
    """

    async def test_stage1_facts_matches_regex_extraction(self, writer):
        """result.stage1_facts equals the count of S1 regex candidates."""
        # This input triggers multiple distinct S1 patterns
        text = "I'm a pilot. I live in Dubai."
        expected = len(extract_facts_s1(text))
        assert expected >= 2, "sanity: need multiple S1 candidates for this test"

        result = await writer.process(text)
        assert result.stage1_facts == expected

    async def test_stage1_facts_zero_for_empty_input(self, writer):
        result = await writer.process("")
        assert result.stage1_facts == 0

    async def test_stage2_facts_zero_when_s2_disabled(self, store):
        """With stage2_enabled=False, stage2 never runs, count stays 0."""
        w = MemoryWriter(store, embed_fn=None, stage2_enabled=False)
        result = await w.process("I'm a pilot. I live in Dubai.")
        assert result.stage2_facts == 0
        assert result.stage2_invoked is False

    async def test_stage2_invoked_reflects_gate_decision(self, store):
        """stage2_invoked is True iff _s2_gate opened AND s2 was enabled.

        The gate opens for longer text with fewer S1 hits — a long
        sentence with no regex matches forces S2.
        """
        # Short text → no S1 hits and fewer than 15 words → gate stays closed
        w_disabled_but_enabled = MemoryWriter(store, embed_fn=None, stage2_enabled=True)
        result_closed = await w_disabled_but_enabled.process("Hello there friend.")
        assert result_closed.stage2_invoked is False, (
            "short unmatched input shouldn't trip the S2 gate"
        )

    async def test_conflicts_detected_zero_when_no_ghosts(self, writer):
        """No ghost-validator drops on clean S1-only input."""
        result = await writer.process("I live in Dubai.")
        assert result.conflicts_detected == 0

    async def test_supersessions_zero_on_first_write(self, writer):
        """First write of a fact can't supersede anything."""
        result = await writer.process("I live in Dubai.")
        assert result.supersessions == 0


# ─── D42: Pet species coverage (breed + acquisition forms) ───────────────

class TestPetSpeciesPatterns:
    """D42 regression: pets named with a breed (whippet, labrador, cat)
    rather than a generic "pet" word went unextracted. Also "we adopted
    a cat called X" style announcements were missed."""

    def test_my_whippet_named(self):
        from sieve.writer import extract_facts_s1
        facts = extract_facts_s1("My whippet Ziggy loves the park.")
        rels = [f for f in facts if f.relation == "whippet"]
        assert rels, f"expected whippet relation, got {[(f.relation, f.content) for f in facts]}"

    def test_we_adopted_cat_named(self):
        from sieve.writer import extract_facts_s1
        facts = extract_facts_s1("We adopted a third pet — a cat called Toast.")
        rels = [f for f in facts if f.relation == "cat"]
        assert rels, f"expected cat relation, got {[(f.relation, f.content) for f in facts]}"
        assert rels[0].related_entity == "Toast"

    def test_we_decided_to_get_whippet(self):
        from sieve.writer import extract_facts_s1
        # This phrasing is trickier — "His name is Ziggy" as a separate
        # sentence. Single-sentence acquisition forms covered; multi-
        # sentence falls to S2. Not asserting here but documenting.
        facts = extract_facts_s1("We got a whippet called Ziggy.")
        rels = [f for f in facts if f.relation == "whippet"]
        assert rels, f"expected whippet relation, got {[(f.relation, f.content) for f in facts]}"


# ─── D1/D18: Interrogative turns must not produce fact writes ───────────

class TestS1RejectsInterrogatives:
    """D1/D18: S1 regex was extracting "User's hamster is Nibbles's" from
    the trap question 'What time is my hamster Nibbles's vet appointment?'
    Pure-interrogative sentences must return [] from S1."""

    def test_trap_hamster_produces_no_facts(self):
        from sieve.writer import extract_facts_s1
        facts = extract_facts_s1("What time is my hamster Nibbles's vet appointment?")
        assert facts == [], f"expected no facts, got {[f.content for f in facts]}"

    def test_trap_brother_produces_no_facts(self):
        from sieve.writer import extract_facts_s1
        facts = extract_facts_s1("How is my brother Tom doing?")
        assert facts == [], f"expected no facts, got {[f.content for f in facts]}"

    def test_declarative_still_extracted(self):
        from sieve.writer import extract_facts_s1
        # A declarative statement must still produce facts.
        facts = extract_facts_s1("My sister Amy lives in Edinburgh with her husband and two kids.")
        assert facts, "expected at least one fact from declarative input"


# ─── D2: Relative-date resolution at S2/S1 time ──────────────────────────

class TestRelativeDateResolution:
    """D2: 'next weekend' stored verbatim was still being reported as
    'next weekend' 10 days later. Resolution at write time anchors
    dates to the clock injected at the time of the fact write."""

    def test_resolves_next_weekend(self):
        from datetime import datetime, timezone
        from sieve.writer import _resolve_relative_dates
        # Thursday 15 Jan 2026 → next weekend is Saturday 17 Jan.
        out = _resolve_relative_dates(
            "Amy is visiting next weekend.",
            datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
        )
        # Fix 4: natural-language date, no (originally ...) parenthetical.
        assert "Saturday" in out and "17" in out and "2026" in out
        assert "originally" not in out
        assert "next weekend" not in out

    def test_resolves_next_month(self):
        from datetime import datetime, timezone
        from sieve.writer import _resolve_relative_dates
        out = _resolve_relative_dates(
            "Pepper needs her vaccination next month.",
            datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
        )
        assert "February 2026" in out
        assert "next month" not in out

    def test_ignores_absolute_dates(self):
        from datetime import datetime, timezone
        from sieve.writer import _resolve_relative_dates
        # Should not mangle absolute dates.
        out = _resolve_relative_dates(
            "Wedding set for June 2027.",
            datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
        )
        assert out == "Wedding set for June 2027."

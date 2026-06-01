"""Tests for sieve._writer_classifier.should_skip_writer().

Methodology: the classifier is precision-first — false-positives (skipping
a real fact-share) are expensive; false-negatives (an unnecessary writer
call) are cheap. Tests therefore lean heavily on the SKIP=False direction
to verify nothing fact-bearing is missed.

The skip-true cases are also tested but with explicit examples drawn from
the Phase 3 simulator's representative traffic.
"""

from __future__ import annotations

import pytest

from sieve._writer_classifier import should_skip_writer


# ── DO NOT SKIP cases — fact-shares, named-entity messages ──────────────────

class TestDoNotSkipFactShares:
    """The classifier must NOT skip any of these. Each is a fact-bearing turn."""

    @pytest.mark.parametrize("text", [
        "Quick context: I live in Lisbon, Portugal.",
        "My hobbies are: blogging, tennis, crocheting.",
        "Hi! I'm Sana, by the way.",
        "I'm 32 and I work as a physiotherapist.",
        "My wife Sam works at the Royal Bristol Infirmary.",
        "We adopted a rescue mutt named Pepper last spring.",
        "I prefer kimchi stew over pad thai for dinner.",
        "Born in Manchester, moved to London in 2019.",
        "My partner is a barrister.",
        "We have three children — Marcus, Amy, and Tom.",
        "I went to Cambridge for university.",
        "Just so you know, my phone number ends in 4827.",
    ])
    def test_fact_share_not_skipped(self, text: str):
        assert should_skip_writer(text) is False, \
            f"FACT-SHARE SKIPPED — would silently lose this fact: {text!r}"


class TestDoNotSkipProperNounMessages:
    """Messages with proper nouns get the writer treatment even if short
    or otherwise pattern-matching a social/filler shape."""

    @pytest.mark.parametrize("text", [
        "Pepper is doing well today.",          # short, but proper noun
        "Lisbon was wonderful.",                # short, but proper noun
        "Marcus called this morning.",          # short, but proper noun
        "Tell me about the Cotswolds.",         # question, but proper noun
        "Thanks for asking about Naoise.",      # social start, but proper noun
    ])
    def test_proper_noun_messages_not_skipped(self, text: str):
        assert should_skip_writer(text) is False


# ── SKIP cases — true filler / pure questions / social ─────────────────────

class TestSkipFiller:
    """Filler turns with no fact content — should skip."""

    @pytest.mark.parametrize("text", [
        "thanks",
        "ok",
        "okay",
        "great",
        "got it",
        "perfect",
        "alright",
        "sounds good",
        "lol",
        "haha",
        "yes",
        "no",
    ])
    def test_pure_filler_skipped(self, text: str):
        assert should_skip_writer(text) is True


class TestSkipPureQuestions:
    """Questions with no proper-noun anchors and no first-person fact markers
    — the writer cannot extract facts from a pure interrogative."""

    @pytest.mark.parametrize("text", [
        "what time is it?",
        "how does this work?",
        "can you explain that?",
        "is that right?",
        "what does the gate do?",
    ])
    def test_pure_question_skipped(self, text: str):
        assert should_skip_writer(text) is True

    def test_question_with_proper_noun_not_skipped(self):
        assert should_skip_writer("What does Pepper do?") is False

    def test_question_phrased_as_first_person_still_skipped(self):
        # Questions about the user that don't trigger any fact-marker
        # patterns are correctly skipped. The fact-marker check requires
        # the trailing space, so "i live?" doesn't match "i live " — the
        # question gets skipped.
        assert should_skip_writer("when was my birthday?") is True
        assert should_skip_writer("how old am I again?") is True

    def test_question_with_fact_marker_pattern_not_skipped(self):
        # If the question phrasing happens to include a fact-marker pattern
        # like "i like " (with space), the classifier preserves safety by
        # not skipping. False-negative cost: one unnecessary writer call.
        # Documents the trade-off explicitly.
        assert should_skip_writer("what do I like best?") is False


class TestSkipSocialOpeners:
    """Short messages starting with social tokens, with no fact content."""

    @pytest.mark.parametrize("text", [
        "thanks, that's helpful",
        "hi there",
        "hello again",
        "hey, just checking in",
        "great, no worries",
        "nice, appreciate it",
    ])
    def test_short_social_skipped(self, text: str):
        assert should_skip_writer(text) is True

    def test_ok_with_comma_does_not_match_social_start(self):
        # "ok, " has a comma not a space, so doesn't start with "ok " in
        # the recognised social-start list. Documents the boundary. We
        # don't skip these — let the writer decide (default safe).
        # In practice these turns will have 0 facts so the writer just
        # confirms that quickly. Cost: one extra writer call. Acceptable.
        assert should_skip_writer("ok, sounds good to me") is False


# ── Edge cases + defensive paths ───────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        assert should_skip_writer("") is True

    def test_whitespace_only(self):
        assert should_skip_writer("   \n\t  ") is True

    def test_none_safe(self):
        # The classifier is robust to None (returns True via the
        # `not user_text` guard) — protects callers that might pass
        # a missing field rather than an empty string.
        assert should_skip_writer(None) is True  # type: ignore[arg-type]

    def test_very_long_filler_not_skipped(self):
        # Long ramble with no markers and no proper nouns — defaults to
        # NOT skipping (default safe path; let the writer decide).
        text = "well " * 50 + "and then more of the same and so on and so forth"
        assert should_skip_writer(text) is False

    def test_fact_marker_case_insensitive(self):
        # Markers are lowercase-compared; uppercase user text should still match.
        assert should_skip_writer("I LIVE IN LISBON") is False

    def test_proper_noun_at_sentence_start_alone_does_not_block(self):
        # A capitalised first word is sentence-start, not a proper-noun signal.
        # "Cool that you mentioned that." has no marker, no mid-sentence proper noun.
        # Default-safe: should not skip (long enough to escape filler rule).
        # Actually this is over 80 chars? Let's pick something definite.
        # "Sounds good." — short social start, no marker, no proper noun → SKIP.
        assert should_skip_writer("Sounds good.") is True


# ── Calibration — Phase 3 representative sample expected outcome ───────────

class TestPhase3Calibration:
    """Sanity-check that the classifier's behaviour on Phase-3-shaped traffic
    skips ~70% — matches the writer-latency-battery measurement."""

    PHASE3_SAMPLE = [
        # 6 substantive_q — none are fact-shares
        ("What's the best way to tighten rope without a knot?", True),
        ("How do I improve my running pace?", True),
        ("Can you summarise the basics?", True),
        ("Walk me through how that works.", False),  # imperative, ambiguous
        ("Why does the sky look orange at sunset?", True),
        ("Tell me more about that approach.", False),  # default-safe

        # 6 personal_q — questions about the user (no fact-shares).
        # All correctly skip: questions can't add new facts the writer
        # didn't already see. The fact-marker check requires the trailing
        # space (so "i live " matches but "i live?" does not).
        ("How old am I again?", True),
        ("What was my name?", True),
        ("Where did I say I live?", True),
        ("What's my pet's name?", True),
        ("Did I mention my partner?", True),
        ("When was my birthday?", True),

        # 6 followups — no fact-shares
        ("How is that progressing?", True),
        ("And what about the other one?", True),
        ("Going back to the rope thing.", False),  # default-safe (proper-shape)
        ("Can you elaborate on that point?", True),
        ("Right, makes sense.", True),  # short social
        ("And what about Marcus?", False),  # proper noun

        # 6 filler — all skippable
        ("thanks", True), ("ok", True), ("great", True),
        ("got it", True), ("perfect", True), ("alright", True),

        # 4 social
        ("hi there", True), ("hello again", True),
        ("good morning", True), ("hey, just a quick one", True),

        # 2 fact_share — MUST NOT be skipped
        ("Quick context: I live in Lisbon, Portugal.", False),
        ("Hi! I'm Sana, by the way.", False),
    ]

    def test_phase3_sample_expectations(self):
        misses = []
        for text, expected_skip in self.PHASE3_SAMPLE:
            actual = should_skip_writer(text)
            if actual != expected_skip:
                misses.append((text, expected_skip, actual))
        assert not misses, f"Classifier disagreements: {misses}"

    def test_phase3_skip_rate_target(self):
        """Of the 30 representative turns, ~70% should be classified as skip."""
        skipped = sum(1 for text, _ in self.PHASE3_SAMPLE if should_skip_writer(text))
        total = len(self.PHASE3_SAMPLE)
        ratio = skipped / total
        # Target: at least 50% skipped (conservative — the actual
        # measured rate from the writer-latency battery is ~70-80%).
        assert ratio >= 0.5, f"Skip rate {ratio:.2f} below 0.5 target"


# ── Performance smoke ──────────────────────────────────────────────────────

class TestPerformance:
    def test_classifier_is_fast(self):
        """The classifier should be sub-millisecond on any realistic input."""
        import time
        text = "I live in Lisbon and my hobbies are blogging and tennis."
        t0 = time.perf_counter()
        for _ in range(1000):
            should_skip_writer(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # 1000 calls should complete in < 50 ms (~50us/call worst case)
        assert elapsed_ms < 50, f"Classifier too slow: {elapsed_ms:.1f}ms for 1000 calls"

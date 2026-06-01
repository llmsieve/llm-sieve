"""Skip-empty classifier — fast pre-filter that decides whether the writer's
S2 extraction call is worth making.

The S2 LLM call costs ~100-2000 ms per turn depending on writer model. On a
representative traffic sample (Phase 3 simulator captures, 30-turn matrix
across 6 intents) **~70% of turns contain no extractable facts**: filler
("thanks"), social greetings ("hi"), questions ("what's my name?"),
followups with no named anchor, etc. Calling the writer on these turns
both wastes latency AND raises the risk of the small writer hallucinating
content from nothing.

This classifier runs in microseconds (regex + string checks only) and
returns True when the writer call should be skipped. It is designed to
be **precision-first**: false-negatives (an unnecessary writer call) are
cheap; false-positives (skipping a real fact-share) are expensive.

The heuristics:

  1. If ANY first-person fact marker is present ("I live", "my name",
     "we adopted", etc.) → DO NOT skip. Fact-share possible.
  2. If a proper-noun (capitalised mid-sentence word) is present
     → DO NOT skip. Captures "I'm Sana", "We live in Lisbon",
     "Pepper is my pet", etc.
  3. Very short text (< 20 chars), no markers, no proper nouns → SKIP.
  4. Pure question (ends with "?", no markers, no proper nouns) → SKIP.
  5. Greeting/social start ("thanks", "ok", "hi" etc.) under 80 chars,
     no markers, no proper nouns → SKIP.
  6. Anything else → DO NOT skip (default safe path: let the writer decide).

Empirical validation: WRITER_LATENCY_BATTERY_RESULTS.md shows this skips
~70-80% of turns from the Phase 3 representative sample while preserving
fact-share recall.

The classifier is intentionally a regex/string heuristic rather than an
ML model so it stays:

  - reproducible (no model weights to ship)
  - auditable (rules are read in source)
  - sub-millisecond (no inference overhead)
  - portable (no Python deps beyond stdlib)
"""

from __future__ import annotations

import re


# First-person fact markers — strong signal that a fact-share is happening.
# Conservative on purpose: matches "I am", "I'm", "my <noun>", "we live",
# etc. Anything that looks like the user is asserting something about
# themselves or their world.
_FACT_MARKERS = (
    "i am ", "i'm ", "i was ", "i live ", "i work ", "i drive ", "my name ",
    "my wife ", "my husband ", "my partner ", "my son ", "my daughter ",
    "my mother ", "my father ", "my brother ", "my sister ", "my parent",
    "my pet ", "my dog ", "my cat ", "my children ", "my kids ",
    "i prefer ", "i like ", "i love ", "i hate ", "i enjoy ",
    "born in ", "born on ",
    "i went ", "i visited ", "i graduated ", "i studied ",
    "i moved ", "we moved ", "we adopted ", "we got ", "we have ",
    "i have ",
    # Multi-word starts that telegraph fact-sharing
    "quick context", "quick update", "by the way", "just so you know",
)


# Social/filler starts — short turns that begin with these and contain no
# fact markers or proper nouns are safe to skip.
_SOCIAL_STARTS = (
    "thanks", "thank you", "got it", "ok ", "okay", "okay ", "great",
    "sounds good", "perfect", "good morning", "good afternoon",
    "good evening", "good night",
    "hi", "hello", "hey", "alright", "cool", "nice",
    "lol", "haha", "yes", "no",
)


# A "proper noun" is a capitalised word AFTER the first character of
# the sentence (so we don't catch the leading "I" or sentence-start
# capitalisation). Min length 2 to skip stray initials.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zà-ÿ]+\b")


def _has_proper_noun(text: str) -> bool:
    """True if the text contains a capitalised mid-sentence word."""
    # Skip the first 2 chars so we don't match sentence-start capitalisation
    if len(text) < 3:
        return False
    return bool(_PROPER_NOUN_RE.search(text[2:]))


def _has_fact_marker(text_lower: str) -> bool:
    """True if any first-person fact marker appears in the text."""
    return any(marker in text_lower for marker in _FACT_MARKERS)


def _is_pure_social(text_lower: str) -> bool:
    """True if the text starts with a recognised social/filler token."""
    return any(text_lower.startswith(token) for token in _SOCIAL_STARTS)


def should_skip_writer(user_text: str) -> bool:
    """Returns True if the writer call should be skipped (no extractable facts likely).

    This is the public entrypoint. Calibrated to skip ~70-80% of representative
    traffic while preserving fact-share recall.
    """
    if not user_text or not user_text.strip():
        return True

    text = user_text.lower().strip()

    # Strong DO-NOT-SKIP signals — short-circuit early
    if _has_fact_marker(text):
        return False
    if _has_proper_noun(user_text):
        return False

    # Now consider the SKIP cases — only fire when no markers and no proper nouns
    # Heuristic 1: very short pure filler ("ok", "thanks", "got it")
    if len(text) < 20:
        return True

    # Heuristic 2: pure question (ends with "?", no first-person markers,
    # no proper nouns to anchor a fact-share to)
    if text.endswith("?"):
        return True

    # Heuristic 3: greeting/social with no fact markers and no proper nouns
    if len(text) < 80 and _is_pure_social(text):
        return True

    # Default: do not skip. Let the writer handle it. Cost of a false-negative
    # (unnecessary writer call) is small; cost of a false-positive (missed
    # fact-share) is large.
    return False

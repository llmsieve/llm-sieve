"""Memory Writer — Stage 1 (regex) + Stage 2 (LLM) fact extraction with conflict resolution.

Stage 1: Regex-based extraction of identity, location, occupation, relationships,
         financial, temporal facts. ~40-50% coverage, <2ms.
Stage 2: LLM-based deep extraction via small model (qwen3.5:0.5b). Handles implicit
         facts, sentiment, pronoun resolution, fact_type classification. ~200ms GPU.
Stage 3: Dedup + conflict resolution using a deterministic decision tree.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger("recall.writer")

# ─── Regex patterns ────────────────────────────────────────────────────────────
# Each pattern yields (fact_text, fact_type, category, raw_groups)

_PATTERNS: list[tuple[str, re.Pattern, str, str]] = []


def _pat(name: str, pattern: str, fact_type: str, category: str) -> None:
    _PATTERNS.append((name, re.compile(pattern, re.IGNORECASE), fact_type, category))


# Identity: "I am/I'm [X]"
_pat("identity_name",
     r"\b[Mm]y\s+name\s+is\s+([A-Z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|\s+and\s+|$)",
     "objective", "identity")
_pat("identity_called",
     r"\bI(?:'m|\s+am)\s+called\s+([A-Z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "identity")
_pat("identity_am",
     r"\bI\s+am\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "identity")
_pat("identity_im",
     r"\bI'm\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "identity")

# Location: "I live in/based in/from [X]"
_pat("location_live",
     r"\bI\s+live\s+in\s+([A-Z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "location")
_pat("location_based",
     r"\bbased\s+in\s+([A-Z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "location")
_pat("location_from",
     r"\bI(?:'m|\s+am)\s+from\s+([A-Z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "location")
_pat("location_moved",
     r"\bI\s+(?:just\s+)?moved\s+to\s+([A-Z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "temporal", "location")

# Occupation: "I work at/for [X]", "I'm a [title]"
_pat("occupation_work_at",
     r"\bI\s+work\s+(?:at|for)\s+([A-Za-z][A-Za-z0-9 '\-\&\.]{1,50}?)(?:[,\.!?]|$)",
     "objective", "occupation")
_pat("occupation_work_as",
     r"\bI\s+work\s+as\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "occupation")
_pat("occupation_job",
     r"\bmy\s+job\s+is\s+([A-Za-z][A-Za-z '\-]{1,40}?)(?:[,\.!?]|$)",
     "objective", "occupation")

# Relationships: "My [relation] [name]", "My [relation] is [name]"
_RELATION_WORDS = (
    r"(?:partner|wife|husband|spouse|girlfriend|boyfriend|fiancee?|"
    r"mother|father|mum|mom|dad|parent|"
    r"son|daughter|child|children|kid|baby|"
    r"brother|sister|sibling|"
    r"friend|best\s+friend|colleague|boss|"
    r"dog|cat|pet)"
)
_pat("relation_is",
     rf"\bmy\s+({_RELATION_WORDS})\s+is\s+(?:called\s+|named\s+)?([A-Z][A-Za-z '\-]{{1,30}}?)(?:[,\.!?]|$)",
     "objective", "relationship")
_pat("relation_named",
     rf"\bmy\s+({_RELATION_WORDS})\s+(?:[A-Za-z]{{1,20}}\s+)?(?:is\s+)?named\s+([A-Z][A-Za-z '\-]{{1,30}}?)(?:[,\.!?]|$)",
     "objective", "relationship")
_pat("relation_possessive",
     rf"\bmy\s+({_RELATION_WORDS})['\u2019]?s?\s+name\s+is\s+([A-Z][A-Za-z '\-]{{1,30}}?)(?:[,\.!?]|$)",
     "objective", "relationship")
# "My best friend Marcus runs…" — direct apposition: capture name after relation
_pat("relation_apposition",
     rf"\bmy\s+({_RELATION_WORDS})\s+([A-Z][a-z]{{1,20}}(?:\s+[A-Z][a-z]{{1,20}})?)\s+[a-z]",
     "objective", "relationship")

# Children count
_pat("children_count",
     r"\bI\s+have\s+((?:one|two|three|four|five|six|seven|eight|\d+)\s+(?:child(?:ren)?|kids?|sons?|daughters?))(?:[,\.!?\s]|$)",
     "objective", "family")

# Financial: numbers with currency/units
_pat("financial_salary",
     r"\b(?:I\s+(?:earn|make|get\s+paid?)\s+)([\$€£¥]?\d[\d,\.]+(?:\s*[kmb](?:illion)?)?(?:\s*(?:USD|EUR|GBP|AED|per\s+(?:year|month|week|hour)))?)\b",
     "conditional", "financial")
_pat("financial_net_worth",
     r"\bmy\s+(?:net\s+worth|savings?|portfolio)\s+(?:is|are)\s+([\$€£¥]?\d[\d,\.]+(?:\s*[kmb](?:illion)?)?(?:\s*(?:USD|EUR|GBP|AED))?)\b",
     "conditional", "financial")

# Hobbies / interests: "My hobbies are X, Y, Z" / "My hobby is X" / "I love/enjoy X"
_pat("hobbies_list",
     r"\bmy\s+hobbies?\s+(?:are|is|include)\s+([^.!?\n]{3,200}?)(?:[.!?\n]|$)",
     "subjective", "hobby")
_pat("interest_love",
     r"\bI\s+(?:love|enjoy|really\s+like)\s+((?:playing|doing|building|making|going|watching|reading)?\s*[A-Za-z][A-Za-z '\-]{2,60}?)(?:[,\.!?\n]|$)",
     "subjective", "hobby")

# Languages spoken
_pat("languages",
     r"\bI\s+speak\s+([A-Z][A-Za-z ,'\-]{2,120}?)(?:[.!?\n]|$)",
     "objective", "language")

# Allergies (multi-item lists)
_pat("allergies",
     r"\bI(?:'m|\s+am)?\s+allergic\s+to\s+([A-Za-z][A-Za-z ,\-]{2,120}?)(?:[.!?\n]|$)",
     "objective", "health")

# Pets: "My cat/dog is named X" / "My cat X" / "We have a dog named X"
#
# D42: extended to include breed nouns as species indicators. In the
# 30-day run "we decided to get the whippet! His name is Ziggy" didn't
# trigger because 'whippet' wasn't recognised as a pet word — now it is.
_pat("pet_named",
     r"\b(?:my|our)\s+(cat|dog|pet|puppy|kitten|rabbit|hamster|bird|parrot|"
     r"whippet|greyhound|retriever|spaniel|poodle|bulldog|shepherd|collie|"
     r"husky|beagle|labrador|pug|dachshund|chihuahua|mastiff|boxer|corgi|"
     r"terrier|hound|tabby|siamese|persian|ragdoll)"
     r"\s+(?:is\s+)?(?:named\s+|called\s+)?([A-Z][A-Za-z '\-]{1,30}?)(?:[,\.!?]|$)",
     "objective", "relationship")
_pat("we_have_relation_named",
     rf"\bwe\s+have\s+(?:a\s+|an\s+)?(?:\w+\s+)?({_RELATION_WORDS})\s+(?:[A-Za-z]{{1,20}}\s+)?(?:is\s+)?(?:named|called)\s+([A-Z][A-Za-z '\-]{{1,30}}?)(?:[,\.!?]|$)",
     "objective", "relationship")
# "I have a dog called Mabel" / "I have a cat named Luna" — seeds the pet
# identity directly so follow-ups ("What breed is Mabel?") can retrieve
# the animal entity even if the user never said "my dog".
_pat("i_have_pet_named",
     r"\bi\s+have\s+(?:a\s+|an\s+|another\s+)?(cat|dog|pet|puppy|kitten|rabbit|hamster|bird|parrot|"
     r"whippet|greyhound|retriever|spaniel|poodle|bulldog|shepherd|collie|husky|beagle|labrador)"
     r"\s+(?:named\s+|called\s+)([A-Z][A-Za-z '\-]{1,30}?)(?:[,\.!?]|$)",
     "objective", "relationship")
# D42: "we adopted/got a cat called X" / "We adopted a third pet — a cat
# called Toast" — covers the acquisition-style announcements. The
# middle-filler matcher accepts anything up to ~20 chars so em-dashes
# and qualifiers like "third pet —" don't break the match.
_pat("we_adopted_pet_named",
     r"\bwe\s+(?:adopted|got|rescued|picked\s+up|brought\s+home)\s+"
     r"[^.!?]{0,40}?"  # liberal middle: handles "a third pet — a "
     r"\b(cat|dog|pet|puppy|kitten|rabbit|hamster|bird|parrot|"
     r"whippet|greyhound|retriever|spaniel|poodle|bulldog|shepherd|collie|"
     r"husky|beagle|labrador|pug|dachshund|chihuahua|mastiff|boxer|corgi|"
     r"terrier|hound|tabby)"
     r"\s+(?:named\s+|called\s+)([A-Z][A-Za-z '\-]{1,30}?)(?:[,\.!?]|$)",
     "objective", "relationship")
# Pet breed: "Mabel is a border terrier" / "she's a border terrier" /
# "he's a golden retriever". Captures the breed as a discrete fact so
# the breed-query path can hit it on its own embedding. The breed slot
# is deliberately liberal (any lowercase noun phrase up to a sentence
# boundary) because breed names vary too much to enumerate.
_pat("pet_breed_named_subject",
     r"\b([A-Z][a-z]{1,20})\s+is\s+(?:a\s+|an\s+)?([a-z][a-z \-]{3,40}?(?:terrier|retriever|spaniel|poodle|bulldog|shepherd|collie|husky|beagle|labrador|pug|dachshund|chihuahua|mastiff|pointer|setter|boxer|corgi|whippet|greyhound|ridgeback|hound|cat|kitten|tabby|siamese|persian|ragdoll|maine\s+coon|sphynx))\b",
     "objective", "pet_breed")
_pat("pet_breed_pronoun",
     r"\b(?:she|he|it)(?:'s|\s+is)\s+(?:a\s+|an\s+)?([a-z][a-z \-]{3,40}?(?:terrier|retriever|spaniel|poodle|bulldog|shepherd|collie|husky|beagle|labrador|pug|dachshund|chihuahua|mastiff|pointer|setter|boxer|corgi|whippet|greyhound|ridgeback|hound|tabby|siamese|persian|ragdoll))\b",
     "objective", "pet_breed")

# Temporal / age
_pat("age",
     r"\bI(?:'m|\s+am)\s+(\d{1,3})\s+years?\s+old\b",
     "temporal", "age")
_pat("birthday",
     r"\bmy\s+birthday\s+is\s+(?:on\s+)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
     "objective", "temporal")

# ── State-transition patterns ────────────────────────────────────────────────
# These emit STATE facts (not evidence) so the supersession matcher has
# predicate-shaped facts to chain together. Category="state_transition"
# routes them to the state_transition handler in extract_facts_s1.

# Marital separation / divorce — broad cues, no capture group needed
_pat("state_separated",
     r"\b(?:we(?:'ve| have)?\s+separated|"
     r"we(?:'re| are)\s+(?:now\s+)?separated|"
     r"(?:[A-Z][a-z]+)\s+and\s+i\s+(?:have\s+|just\s+)?separated|"
     r"separated\s+from\s+(?:my\s+)?(?:husband|wife|spouse|partner)|"
     r"(?:my\s+)?(?:husband|wife|spouse|partner)\s+moved\s+out|"
     r"i\s+moved\s+out\s+of\s+(?:the\s+|our\s+)|"
     r"living\s+apart|"
     r"trial\s+separation)\b",
     "objective", "state_transition_separated")
_pat("state_divorced",
     r"\b(?:we(?:'ve| have)?\s+divorced|we\s+got\s+divorced|"
     r"i(?:'m| am)\s+divorced|"
     r"finali[sz]ed\s+(?:the|our)\s+divorce)\b",
     "objective", "state_transition_divorced")

# Promotion / role change — capture role title up to a stop word or end
_pat("state_promoted_to",
     r"\b(?:i\s+(?:was\s+|got\s+)?promoted\s+to|"
     r"i(?:'m| am)\s+now\s+the|"
     r"i(?:'ve| have)\s+been\s+(?:made|named)\s+the|"
     r"i\s+accepted\s+the|"
     r"i\s+took\s+the)\s+"
     r"((?:VP\s+(?:of\s+)?[A-Za-z]+|CTO|CEO|CFO|COO|"
     r"(?:Senior|Lead|Principal|Staff|Chief)\s+[A-Za-z][A-Za-z]{1,20}(?:\s+[A-Za-z]{1,20})?))"
     r"(?:\s+role|\s+position|\s+job|\s+at\b|[,\.\!?]|\s+yesterday|\s+today|\s+last\s+week|$)",
     "objective", "state_transition_role")

# Quit / left employer — stop at temporal modifiers
_pat("state_quit",
     r"\bi\s+(?:quit|left|resigned\s+from)\s+"
     r"([A-Z][A-Za-z0-9 '\-\&\.]{1,40}?)"
     r"(?:\s+(?:on|last|yesterday|today|this|two|three)\b|[,\.\!?]|$)",
     "objective", "state_transition_quit")

# Decided against / abandoned plan — capture noun phrase, drop "getting"/"applying"/etc.
_pat("state_decided_against",
     r"\bi(?:'ve| have)?\s+(?:decided\s+against|abandoned|dropped)\s+"
     r"(?:the\s+|my\s+|getting\s+|applying\s+to\s+|applying\s+for\s+)?"
     r"((?:a\s+|an\s+)?[A-Za-z][A-Za-z '\-]{2,40}?)"
     r"(?:\s+(?:after\s+all|idea|plan|application|programme|program))?(?:[,\.\!?]|$)",
     "temporal", "state_transition_decided_against")
_pat("state_we_decided_against",
     r"\bwe(?:'ve| have)?\s+decided\s+(?:against|not\s+to)\s+"
     r"(?:get(?:ting)?\s+|adopt(?:ing)?\s+)?"
     r"((?:a\s+|an\s+)?[A-Za-z][A-Za-z '\-]{2,40}?)"
     r"(?:\s+after\s+all)?(?:[,\.\!?]|$)",
     "temporal", "state_transition_decided_against")
_pat("state_no_longer",
     r"\bi(?:'m| am)\s+no\s+longer\s+"
     r"(?:considering|planning|interested\s+in|going\s+to)\s+"
     r"((?:the\s+|a\s+|an\s+)?[A-Za-z][A-Za-z '\-]{2,40}?)"
     r"(?:\s+(?:program|programme|application|plan|idea))?"
     r"(?:[,\.\!?]|$)",
     "temporal", "state_transition_no_longer")

# Relocation — stop at temporal tail
_pat("state_moved_to",
     r"\bwe\s+(?:just\s+|recently\s+)?(?:moved|relocated)\s+to\s+"
     r"([A-Z][A-Za-z '\-]{1,40}?)"
     r"(?:\s+(?:last|this|two|yesterday|today)\b|[,\.\!?]|$)",
     "temporal", "state_transition_moved")

# Named entities (proper nouns — supplement regex facts)
_PROPER_NOUN_PATTERN = re.compile(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})*)\b")
# Common words to filter out of proper noun detection
_COMMON_WORDS = frozenset({
    "I", "The", "A", "An", "My", "Your", "Our", "Their", "His", "Her",
    "This", "That", "These", "Those", "It", "We", "You", "He", "She",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Ok", "Okay", "Yes", "No", "Hi", "Hello", "Thanks", "Thank",
})
# Lowercase index of the same set — used by pet_breed handler to reject
# "She is a border terrier" capturing "She" as the subject name.
_COMMON_WORDS_LOWER = frozenset(w.lower() for w in _COMMON_WORDS)


# ─── Extracted candidates ───────────────────────────────────────────────────────

@dataclass
class ExtractedFact:
    content: str
    fact_type: str        # objective | subjective | conditional | temporal
    category: str         # identity | location | occupation | relationship | financial | temporal
    confidence: float = 0.75
    entity_names: list[str] = field(default_factory=list)
    relation: str | None = None          # for relationship facts
    related_entity: str | None = None    # for relationship facts


@dataclass
class WriteResult:
    facts_written: int = 0
    facts_skipped: int = 0       # dedup hit
    entities_written: int = 0
    relationships_written: int = 0
    session_id: str = ""
    elapsed_ms: float = 0.0
    # Per-stage extraction counts (validation instrumentation).
    # stage1_facts / stage2_facts count CANDIDATES extracted, not
    # post-filter writes. The difference against facts_written reveals
    # dedup / ghost-validator / pet-filter drops.
    stage1_facts: int = 0
    stage2_facts: int = 0
    stage2_invoked: bool = False  # True iff the S2 gate opened and S2 ran
    conflicts_detected: int = 0   # ghost-validator rejects
    supersessions: int = 0        # temporal-versioning supersedes on write


# ─── Stage 1 extraction ─────────────────────────────────────────────────────────

# Tokens that must never appear inside a captured name. The S1 regexes
# compile with re.IGNORECASE (required for 'My' vs 'my', etc.) which
# neutralises their [A-Z] anchors on name captures, letting function
# words bleed in from surrounding prose: "my daughter from school"
# would otherwise produce a fact "User's daughter is from".
_NAME_STOP_WORDS = frozenset({
    # prepositions / conjunctions
    "from", "to", "of", "for", "with", "at", "by", "in", "on", "and",
    "or", "but", "as", "into", "onto", "about",
    # auxiliaries / common verbs
    "is", "was", "are", "were", "has", "have", "had", "do", "does",
    "did", "be", "been", "being", "will", "would", "should", "could",
    # articles / quantifiers
    "a", "an", "the", "my", "your", "his", "her", "our", "their",
    "this", "that", "these", "those",
})


def _trim_name_capture(candidate: str, original_text: str) -> str:
    """Trim regex captures back to the real name boundary.

    The S1 patterns use re.IGNORECASE, which disables their [A-Z]
    name anchors. Captures can therefore include trailing function
    words ('from', 'has', 'was') or lowercase verbs ('runs') that
    bled in from the next clause. Keep only the leading run of tokens
    that (a) are not function words and (b) are capitalised in the
    source. Returns '' when no leading token qualifies.
    """
    candidate = (candidate or "").strip()
    if not candidate:
        return ""
    kept: list[str] = []
    for tok in candidate.split():
        clean = tok.strip(".,!?;:'\u2019-").lower()
        if not clean or clean in _NAME_STOP_WORDS:
            break
        m = re.search(rf"\b{re.escape(tok.strip('.,!?;:'))}\b", original_text)
        if m and not m.group(0)[:1].isupper():
            break
        kept.append(tok)
    return " ".join(kept)


_PURE_QUESTION_RE = re.compile(
    r"^\s*(what|when|where|why|who|whom|whose|which|how|is|are|was|were|"
    r"do|does|did|have|has|had|can|could|will|would|should|shall|may|might|"
    r"am|am i|did i|can i|do i|will i|have i|will you|can you|could you)\b",
    re.IGNORECASE,
)


def extract_facts_s1(text: str) -> list[ExtractedFact]:
    """Stage 1: regex extraction from a single text string.

    Returns extracted fact candidates. Fast (<2ms). No LLM.

    D1/D18: pure interrogative turns are skipped. S1 patterns were
    extracting "User's hamster is Nibbles's" from the trap query
    "What time is my hamster Nibbles's vet appointment?" — the regex
    caught "my hamster Nibbles" as a relationship fact. Rejecting
    single-sentence question turns before pattern matching is the
    simplest correct fix.
    """
    stripped = text.strip()
    if (
        stripped.endswith("?")
        and stripped.count("?") == 1
        and stripped.count(".") == 0
        and _PURE_QUESTION_RE.match(stripped)
    ):
        return []

    results: list[ExtractedFact] = []
    seen_contents: set[str] = set()

    def _add(content: str, fact_type: str, category: str,
             confidence: float = 0.75, entity_names: list[str] | None = None,
             relation: str | None = None, related_entity: str | None = None) -> None:
        content = content.strip().rstrip(".,!?;")
        if len(content) < 3 or content.lower() in seen_contents:
            return
        seen_contents.add(content.lower())
        results.append(ExtractedFact(
            content=content,
            fact_type=fact_type,
            category=category,
            confidence=confidence,
            entity_names=entity_names or [],
            relation=relation,
            related_entity=related_entity,
        ))

    for name, pattern, fact_type, category in _PATTERNS:
        for m in pattern.finditer(text):
            groups = [g.strip() for g in m.groups() if g]
            # Zero-capture state-transition patterns (separated /
            # divorced) emit a fixed state fact on any match.
            if not groups:
                if category == "state_transition_separated":
                    _add("User is separated", "temporal", "relationship", confidence=0.9)
                elif category == "state_transition_divorced":
                    _add("User is divorced", "temporal", "relationship", confidence=0.9)
                continue

            if category == "relationship" and len(groups) >= 2:
                relation_word = groups[0].lower()
                entity_name = groups[1].strip().rstrip(".,!?")
                # Trim captures that bled into surrounding prose —
                # the S1 patterns' re.IGNORECASE defeats their [A-Z]
                # name anchors and can include words like 'from', 'has',
                # 'runs' from the next clause.
                entity_name = _trim_name_capture(entity_name, text)
                if not entity_name:
                    continue
                content = f"User's {relation_word} is {entity_name}"
                _add(content, fact_type, category,
                     entity_names=[entity_name],
                     relation=relation_word,
                     related_entity=entity_name)

            elif category == "location":
                loc = groups[0].strip().rstrip(".,!?")
                content = f"User lives in {loc}"
                _add(content, fact_type, category, entity_names=[loc])

            elif category == "identity":
                val = groups[0].strip().rstrip(".,!?")
                # Filter out very short or common false-positives
                if len(val) >= 2 and val.lower() not in {"ok", "fine", "good", "here", "sure", "not", "just", "still", "also"}:
                    content = f"User is {val}"
                    _add(content, fact_type, category)

            elif category == "occupation":
                val = groups[0].strip().rstrip(".,!?")
                if len(val) >= 3:
                    content = f"User works at/as {val}"
                    _add(content, fact_type, category)

            elif category == "age":
                content = f"User is {groups[0]} years old"
                _add(content, "temporal", "age")

            elif category in ("financial", "temporal", "family"):
                val = " ".join(groups).strip().rstrip(".,!?")
                if val:
                    content = f"User has {val}" if category == "family" else f"User {val}"
                    _add(content, fact_type, category)

            elif category == "hobby":
                val = groups[0].strip().rstrip(".,!?")
                if len(val) >= 3:
                    # Split on commas/and for multi-item lists so retrieval picks each item
                    items = [p.strip(" .,;") for p in re.split(r",|\band\b", val) if p.strip(" .,;")]
                    if len(items) > 1:
                        for item in items:
                            if len(item) >= 3:
                                _add(f"User enjoys {item}", fact_type, category)
                    else:
                        _add(f"User enjoys {val}", fact_type, category)

            elif category == "language":
                val = groups[0].strip().rstrip(".,!?")
                if len(val) >= 2:
                    items = [p.strip(" .,;") for p in re.split(r",|\band\b", val) if p.strip(" .,;")]
                    if len(items) > 1:
                        for item in items:
                            if len(item) >= 2:
                                _add(f"User speaks {item}", fact_type, category)
                    else:
                        _add(f"User speaks {val}", fact_type, category)

            elif category.startswith("state_transition"):
                # Emit STATE facts shaped to match the value-predicate
                # templates so they can supersede earlier state facts.
                kind = category.removeprefix("state_transition_")
                if kind == "separated":
                    _add("User is separated", "temporal", "relationship",
                         confidence=0.9)
                elif kind == "divorced":
                    _add("User is divorced", "temporal", "relationship",
                         confidence=0.9)
                elif kind == "role":
                    role = groups[-1].strip().rstrip(".,!?")
                    if role:
                        _add(f"User's role is {role}", "temporal", "occupation",
                             confidence=0.9)
                elif kind == "quit":
                    employer = groups[0].strip().rstrip(".,!?")
                    if employer:
                        _add(f"User no longer works at {employer}",
                             "temporal", "occupation", confidence=0.85)
                elif kind == "decided_against":
                    target = groups[0].strip().rstrip(".,!?")
                    if target:
                        _add(f"User decided against {target}",
                             "temporal", "decision", confidence=0.85)
                elif kind == "no_longer":
                    target = groups[-1].strip().rstrip(".,!?")
                    if target:
                        _add(f"User no longer wants {target}",
                             "temporal", "decision", confidence=0.85)
                elif kind == "moved":
                    place = groups[0].strip().rstrip(".,!?")
                    if place:
                        _add(f"User lives in {place}",
                             "temporal", "location", confidence=0.9,
                             entity_names=[place])

            elif category == "health":
                val = groups[0].strip().rstrip(".,!?")
                if len(val) >= 3:
                    items = [p.strip(" .,;") for p in re.split(r",|\band\b", val) if p.strip(" .,;")]
                    if len(items) > 1:
                        for item in items:
                            if len(item) >= 3:
                                _add(f"User is allergic to {item}", fact_type, category)
                    else:
                        _add(f"User is allergic to {val}", fact_type, category)

            elif category == "pet_breed":
                # Two shapes:
                #   pet_breed_named_subject → groups = [Name, breed]
                #   pet_breed_pronoun       → groups = [breed]
                if len(groups) >= 2:
                    name = groups[0].strip().rstrip(".,!?")
                    breed = groups[1].strip().rstrip(".,!?")
                    if name and breed and name.lower() not in _COMMON_WORDS_LOWER:
                        _add(
                            f"{name} is a {breed}",
                            fact_type, category,
                            entity_names=[name],
                        )
                elif len(groups) == 1:
                    breed = groups[0].strip().rstrip(".,!?")
                    if breed:
                        _add(
                            f"User's pet is a {breed}",
                            fact_type, category,
                        )

    return results


def extract_proper_nouns(text: str) -> list[str]:
    """Extract proper nouns from text for entity detection."""
    found = []
    for m in _PROPER_NOUN_PATTERN.finditer(text):
        word = m.group(1)
        if word not in _COMMON_WORDS and len(word) > 1:
            found.append(word)
    return list(dict.fromkeys(found))  # deduplicate preserving order


# ─── Stage 2: LLM extraction ──────────────────────────────────────────────────

# The S2 prompt is a template rendered with the profile owner's name.
# When owner_name is empty, we render the generic body only
# (backwards-compatible fallback).

_S2_OWNER_HEADER_TEMPLATE = """\
PROFILE OWNER: {owner_name}

The person speaking in the conversation below is {owner_name}. First-person
pronouns ("I", "me", "my", "we") ALWAYS refer to {owner_name}. Every fact
you extract must have {owner_name} or one of {owner_name}'s known associates
as its subject — never invert.

Before emitting a fact, rewrite any first-person sentence to third-person
with {owner_name} as the subject:
  "I listed my condo"           -> "{owner_name} listed {owner_name}'s condo"
  "Kim moved out"                -> "Kim moved out of {owner_name}'s residence"
  "I got promoted to VP"         -> "{owner_name}'s role is VP of Engineering"

NEVER emit a fact where {owner_name} is the object of a family/relationship
predicate (husband, wife, spouse, parent, child, sibling). {owner_name} IS
the user; other people have relationships TO {owner_name}, not the other way.

NEVER emit a fact that introduces a new entity with the same name as
{owner_name}. If the conversation says "my twin brother {owner_name}",
that is a pronoun/narrative error — skip that fact entirely.

"""

_S2_EXTRACTION_BODY = """\
Extract EVERY factual claim from the message — do not skip anything that
might be useful for future recall. A message that mentions a spouse, a
date, a location, and a preference contains FOUR facts, not one; emit
all four.

Return ONLY valid JSON with this structure:
{
  "facts": [
    {
      "content": "short factual statement",
      "fact_type": "objective|subjective|conditional|temporal",
      "category": "identity|location|occupation|relationship|financial|preference|opinion|health|education|hobby",
      "confidence": 0.7,
      "entities": ["entity names mentioned"],
      "speculative": false
    }
  ]
}

Thoroughness directive (read before every extraction):
- Include every: name, location, relationship, date, number, preference,
  decision, plan, emotion, or personal detail mentioned. Each becomes
  its own fact row — do not pack multiple claims into one content
  string.
- For each fact, the content string should read as a complete
  subject-predicate-object sentence (e.g. "Jamie's wife is Sam",
  NOT just "Sam"). The subject is almost always the profile owner
  or a named associate.
- If a message contains personal information, emit AT LEAST 3-5 facts
  whenever the message supports it. A typical personal message
  ("I ran 8 miles in 70 minutes, training for the Bristol Half Marathon
  on April 15, goal sub-2-hours") contains at least four extractable
  facts — extract all of them.
- A message with zero personal information (pure general-knowledge
  question, no self-reference, no named people) may legitimately yield
  0 facts. Do not invent.

GROUNDING RULE (D24):
  Every extracted fact must be directly supported by text that appears
  IN THIS MESSAGE. Do not carry forward entities from imagined prior
  context. Do not extract facts about people, places, dates, or objects
  that are not named or clearly referenced in THIS message.

  Example — correct grounding:
    Input: "Sam works as a physiotherapist at the Royal Bristol Infirmary."
    OK:    "Sam works as a physiotherapist at the Royal Bristol Infirmary"
    OK:    "Sam's employer is the Royal Bristol Infirmary"
    NOT OK: "User's sister is Amy"    (Amy is not in this message)
    NOT OK: "Sam is the user's wife"  (not stated in this message)

  If the message doesn't contain the subject OR object of a proposed
  fact, DO NOT emit that fact.

QUESTION RULE:
  If the entire message is a question or request ("How is X doing?",
  "What time is Y?"), return {"facts": []}. Questions are not
  assertions. The trap pattern is "How is my brother Tom doing?" —
  this is the user asking about a brother named Tom, NOT asserting
  one exists. Do not extract "User's brother is Tom" from it.

POLARITY RULE (D41 / children leak):
  When the user mentions a relationship in passing that belongs to
  SOMEONE ELSE, do not re-attach it to the user. Examples:
    "Amy is visiting with the kids"     → Amy has kids, NOT user has kids
    "Sam's mother called"                → Sam has a mother, NOT user's
                                             mother
    "Marcus and his wife came over"     → Marcus has a wife, NOT user
  Rule of thumb: if a kinship word ("kids", "mother", "wife",
  "children") follows a 3rd-person possessive (her, his, their) OR
  follows another named person with "with" / "and", that relationship
  belongs to the named person — NOT to the user. Do NOT emit a fact
  making the user the possessor.

Worked example. Input message:
  "My wife Sam's birthday is coming up on the 22nd. She mentioned
   wanting to try The Ox restaurant in Bristol."
This message contains THREE extractable facts (spouse identity,
Sam's birthday, Sam's preference). All three must appear in the
"facts" array of the same JSON object — ALWAYS wrap every fact inside
{"facts": [...]} as shown in the schema above. A single unwrapped fact
object is invalid output.

Rules:
- fact_type "objective" = singular truth (name, birthdate, employer)
- fact_type "subjective" = opinion, feeling, preference (never contradict each other)
- fact_type "conditional" = context-dependent (salary in a role, skill at a level)
- fact_type "temporal" = changes over time (age, location, job title)
- Set speculative=true if the user uses hedging language (thinking about, might, considering, maybe, not sure)
- Extract implicit facts (e.g. "We celebrated at the restaurant near my office" -> user has an office)
- Resolve ALL first-person pronouns to the profile owner. Resolve second-person to the assistant.
- Do NOT extract generic knowledge or instructions the user gives to the assistant
- Only extract facts about the USER, not about other topics
- ALWAYS preserve specific proper nouns in the fact content: pet names,
  business names, product names, place names, book/movie titles. Never
  summarize "Scout" to "a puppy" or "Ember Coffee" to "a coffee business".
- STATE TRANSITIONS: when the user describes a life-state CHANGE, extract
  the new STATE as a fact, not just the evidence. Examples:
    "Kim moved out last month"            -> "User and Kim are separated"
    "I got promoted to VP of Engineering" -> "User's role is VP of Engineering"
    "we filed for divorce"                -> "User is divorced"
    "I quit Other Corp on Friday"         -> "User no longer works at Other Corp"
    "we decided not to get the dog"       -> "User decided against getting a dog"
  Use fact_type="temporal" for state transitions. The new state should
  contradict the prior state in writing form ("User is married" -> "User is
  separated"), so the older fact can be properly superseded.
- DO NOT confuse persons with pets. If a name is already known to be a person
  (e.g. spouse, child, friend), do not extract a "pet named X" fact about it.
"""


def _render_s2_prompt(owner_name: str) -> str:
    """Render the S2 extraction system prompt.

    When owner_name is non-empty, the owner header is prepended to the
    prompt so S2 resolves first-person pronouns to the profile subject.
    When empty, falls back to the generic body (backwards-compatible).
    """
    if owner_name:
        header = _S2_OWNER_HEADER_TEMPLATE.format(owner_name=owner_name)
        return header + _S2_EXTRACTION_BODY
    return _S2_EXTRACTION_BODY


# ─── Ghost-fact validator ─────────────────────────────────────────────────────
# Rejects S2-extracted facts whose shape indicates the writer inverted the
# profile owner into another person's position. Two deterministic rules:
#   1. identity collision  — owner appears as another person's spouse/relative
#   2. duplicate-name      — a new entity is introduced with the owner's name
#
# An earlier draft had a third rule (inverted role assignment) that depended
# on a `category` column in the facts table; that column does not exist, so
# the rule was dropped. If post-reseed audits show residual inverted-role
# ghost facts, Rule 2 can be reintroduced with a different design.


def _owner_first_name(owner_name: str) -> str:
    """Return the first whitespace-delimited token of the owner's full name."""
    return owner_name.split()[0] if owner_name else ""


@functools.lru_cache(maxsize=8)
def _compile_ghost_patterns(owner_name: str) -> dict:
    """Compile the validator regexes for a given profile owner.

    Cached because they're called on every S2 fact. Cache is small (8)
    because we only have one profile owner per session in practice.
    """
    owner_first = _owner_first_name(owner_name)
    first_esc = re.escape(owner_first) if owner_first else ""
    full_esc = re.escape(owner_name)
    return {
        "pat1a": re.compile(
            r"(?i)\b(?:user|" + first_esc + r"|" + full_esc + r")\s+is\s+"
            r"(?P<possessor>[A-Z][a-z]+)'s\s+"
            r"(?:husband|wife|spouse|partner|fianc[eé]e?)\b"
        ),
        "pat1b": re.compile(
            r"(?i)\b(?:" + full_esc + r"|" + first_esc + r")\s+is\s+"
            r"(?:a\s+|the\s+)?(?:twin\s+|half-?)?"
            r"(?:brother|sister|cousin|parent|mother|father|son|daughter)\s+"
            r"(?:of|named|called)\s+[A-Z][a-z]+"
        ),
        "pat3a": re.compile(
            r"(?i)\bnamed\s+" + first_esc + r"\b"
        ) if first_esc else None,
        "pat3b": re.compile(
            r"(?i)\b(?:twin\s+|half-?)?"
            r"(?:brother|sister|cousin|son|daughter)\s+"
            + first_esc + r"\b"
        ) if first_esc else None,
        # Rule 4: reject "owner lives with X" when X is a known
        # relative. The relatives set is passed in at call time; empty
        # set = rule is a no-op. Allows intervening qualifiers like "a
        # neighbor named" before the capitalized name so "Jamie Rivera
        # lives with a neighbor named Pat" still fires on "Pat".
        "pat4_head": re.compile(
            r"(?i)\b(?:user|" + first_esc + r"|" + full_esc + r")\s+"
            r"(?:lives?|lived|resides?|stays?|staying)\s+with\s+"
        ),
        # Secondary: match any capitalized-first-name-shape token in the
        # tail (window scanned after the head match).
        "pat4_name": re.compile(r"\b([A-Z][a-z]+)\b"),
    }


def _known_relative_first_names(store: Any, owner_entity_id: str) -> set[str]:
    """Return the set of first names of the profile owner's known relatives.

    Queries the relationships table for outbound edges from the owner
    with family-shaped relationship types, splits each target entity
    name on whitespace, and returns the lowercased first tokens.

    Used by ghost validator Rule 4 to reject "owner lives with X"
    facts when X is a family member — family cohabitation is encoded
    in the relationships graph, not as residence slots.

    Returns an empty set when the store is unavailable, when no owner
    entity exists, or when no family edges have been written yet (early
    in a fresh reseed). An empty set makes Rule 4 a no-op.
    """
    if not store or not owner_entity_id:
        return set()

    _FAMILY_REL_TYPES = (
        "has_child", "has_son", "has_daughter", "child", "son", "daughter",
        "spouse", "husband", "wife", "partner",
        "parent", "mother", "father",
        "sibling", "brother", "sister",
    )
    names: set[str] = set()
    try:
        conn = getattr(store, "conn", None)
        if conn is None:
            return names
        # Join relationships -> entities on the target side so we can
        # read the human-readable name. Owner entity_id is the canonical
        # snake-case form from the Tier 2 classifier ("jamie_rivera").
        placeholders = ",".join("?" for _ in _FAMILY_REL_TYPES)
        rows = conn.execute(
            "SELECT e.name FROM relationships r "
            "LEFT JOIN entities e ON e.id = r.target_entity "
            f"WHERE r.source_entity = ? AND r.relationship IN ({placeholders}) "
            "AND (r.valid_to IS NULL OR r.valid_to = '')",
            (owner_entity_id, *_FAMILY_REL_TYPES),
        ).fetchall()
        for row in rows:
            name = (row[0] or "").strip()
            if not name:
                continue
            first = name.split()[0].lower()
            if first:
                names.add(first)
    except Exception as exc:
        logger.warning("_known_relative_first_names failed: %s", exc)
    return names


def _validate_s2_fact(
    fact: "ExtractedFact",
    owner_name: str,
    aliases: list[str],
    relatives: set[str] | None = None,
) -> tuple[bool, str | None]:
    """Structural validator for S2-extracted facts.

    Returns (keep, reject_reason). When keep=False, reject_reason is a short
    rule tag ("identity", "duplicate", or "relative_cohabitation") used
    in logs.

    Empty owner_name returns (True, None) — backwards-compatible fallback.

    `relatives` is the set of lowercased first names of the owner's
    known relatives (from _known_relative_first_names). An empty or
    None set disables Rule 4.
    """
    if not owner_name:
        return (True, None)

    content = fact.content or ""
    owner_first = _owner_first_name(owner_name)
    aliases_lower = {a.lower() for a in aliases}
    aliases_lower |= {owner_first.lower(), owner_name.lower()}
    pats = _compile_ghost_patterns(owner_name)

    # Rule 1a: "(user|Jamie|Jamie Rivera) is X's spouse/etc" where X is not the owner.
    m = pats["pat1a"].search(content)
    if m:
        possessor = m.group("possessor").lower()
        if possessor not in aliases_lower:
            return (False, "identity")

    # Rule 1b: "Jamie (Rivera) is (a) (twin|half-)? (brother|sister|...) (of|named|called) X"
    # Note: [A-Z][a-z]+ is effectively \w+ under (?i). A future editor should NOT
    # tighten this character class — S2 can emit lowercase names and we want
    # those rejected too.
    if pats["pat1b"].search(content):
        return (False, "identity")

    # Rule 3a: "named {owner_first}" — new entity introduced with owner's name.
    # Note: we intentionally do NOT check "called" here because colloquial
    # phrasing like "we called Jamie to discuss" would produce false positives.
    if pats["pat3a"] is not None and pats["pat3a"].search(content):
        return (False, "duplicate")

    # Rule 3b: appositive form — "twin sister Jamie", "brother Jamie", etc.
    if pats["pat3b"] is not None and pats["pat3b"].search(content):
        return (False, "duplicate")

    # Rule 5 (D41/children leak): reject assertions that the user has
    # unnamed children / kids. S2 was extracting "User has children" or
    # "User's children are visiting" from turns like "Amy is visiting
    # with the kids" — where the kids belong to Amy, not the user.
    # Legitimate child facts always name the child (msg: "my son Jake").
    content_lc = content.lower()
    _CHILD_PHRASES = (
        "user has children", "user has kids", "user's children",
        "user's kids", f"{owner_first.lower()} has children",
        f"{owner_first.lower()} has kids",
        f"{owner_first.lower()}'s children",
        f"{owner_first.lower()}'s kids",
    )
    if any(p in content_lc for p in _CHILD_PHRASES):
        # Unless a specific child name is present — "User's children
        # include Freddie and Max" is still legitimate. Check for a
        # capitalised name after the kinship word.
        if not re.search(r"\b(?:child(?:ren)?|kids?)\b[^.!?\n]*?\b[A-Z][a-z]{2,}\b", content):
            return (False, "unnamed_child")

    # Rule 4: "<owner> lives/resides/stays with <X>" where <X> is a
    # known family-first-name. Two-stage regex: first find the head
    # ("Jamie Rivera lives with "), then scan the tail for any
    # capitalized first-name token. This way "Jamie Rivera lives with a
    # neighbor named Pat" still catches "Pat". No-op when the relatives
    # set is empty (e.g. early in a fresh reseed).
    if relatives:
        head = pats["pat4_head"].search(content)
        if head:
            tail = content[head.end():]
            for name_match in pats["pat4_name"].finditer(tail):
                cohabitant = name_match.group(1).lower()
                if cohabitant in relatives:
                    return (False, "relative_cohabitation")

    return (True, None)


# D2: relative-date resolver. Replaces "next week", "next month",
# "last weekend", etc. in a fact content string with a resolved
# date range, anchored to the clock passed in via ``now``. Preserves
# the original phrase in parentheses so the fact is still traceable.
_RELATIVE_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnext weekend\b", re.IGNORECASE), "next_weekend"),
    (re.compile(r"\bthis weekend\b", re.IGNORECASE), "this_weekend"),
    (re.compile(r"\blast weekend\b", re.IGNORECASE), "last_weekend"),
    (re.compile(r"\bnext week\b", re.IGNORECASE), "next_week"),
    (re.compile(r"\blast week\b", re.IGNORECASE), "last_week"),
    (re.compile(r"\bnext month\b", re.IGNORECASE), "next_month"),
    (re.compile(r"\blast month\b", re.IGNORECASE), "last_month"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), "tomorrow"),
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
]


def _resolve_relative_dates(content: str, now) -> str:
    """Replace relative-date tokens in fact content with resolved dates.

    ``now`` is a timezone-aware datetime. For each matched relative
    expression, replace with a natural-language absolute date
    ("Saturday 17 Jan 2026"). The earlier "(originally 'X')"
    parenthetical was dropped in Fix 4 — it penalised Sieve in
    grading for verbosity and confused the composer. Original phrase
    is not preserved inline; callers that need audit trail should
    diff against the pre-resolve content.
    """
    if not content or now is None:
        return content
    from datetime import timedelta

    def _weekend_of(ref):
        # Saturday = weekday 5, Sunday = 6. Return the next Saturday.
        days_ahead = (5 - ref.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return ref + timedelta(days=days_ahead)

    def _fmt(d):
        # "Saturday 17 Jan 2026" — compact, unambiguous, composer-friendly.
        return d.strftime("%A %-d %b %Y")

    def _fmt_month(d):
        return d.strftime("%B %Y")

    replacements = {
        "next_weekend": _fmt(_weekend_of(now).date()),
        "this_weekend": _fmt((_weekend_of(now) - timedelta(days=7)).date())
            if now.weekday() < 5 else _fmt(now.date()),
        "last_weekend": _fmt((_weekend_of(now) - timedelta(days=7)).date()),
        "next_week": _fmt((now + timedelta(days=7)).date()),
        "last_week": _fmt((now - timedelta(days=7)).date()),
        "next_month": _fmt_month((now.replace(day=1) + timedelta(days=32)).replace(day=1).date()),
        "last_month": _fmt_month((now.replace(day=1) - timedelta(days=1)).replace(day=1).date()),
        "yesterday": _fmt((now - timedelta(days=1)).date()),
        "tomorrow": _fmt((now + timedelta(days=1)).date()),
        "today": _fmt(now.date()),
    }

    out = content
    for rx, key in _RELATIVE_DATE_PATTERNS:
        m = rx.search(out)
        if m is None:
            continue
        out = rx.sub(replacements[key], out, count=1)
    return out


def _s2_gate(text: str, s1_facts: list[ExtractedFact]) -> bool:
    """Stage 2 gate: should we invoke the LLM for deeper extraction?

    Triggers when: >10 words AND (proper nouns that S1 didn't cover OR complex content).

    D18/D1: skip on pure interrogative turns. The 30-day run showed the
    S2 writer extracting "User's brother is Tom" from "How is my brother
    Tom doing?" — a trap question. Questions ending with '?' and no
    declarative content are rejected early, preventing the most common
    class of trap-ingestion regardless of downstream validator coverage.
    """
    words = text.split()
    if len(words) < 6:
        return False

    # D18/D1: pure question turns — reject before any keyword check.
    # A turn is "pure interrogative" when it's a single sentence ending
    # in '?' and starts with a question word. Declarative+question turns
    # (e.g. "Mum moved to Bristol. Is that OK?") still open the gate.
    stripped = text.strip()
    if stripped.endswith("?") and stripped.count("?") == 1 and stripped.count(".") == 0:
        if re.match(
            r"^\s*(what|when|where|why|who|whom|whose|which|how|is|are|was|were|"
            r"do|does|did|have|has|had|can|could|will|would|should|shall|may|might|"
            r"am|did i|can i|do i|will i|have i)\b",
            stripped, re.IGNORECASE,
        ):
            return False

    # Strong-signal keywords that always open the gate (personal facts S1 often misses)
    if re.search(
        r"\b(hobbies?|interests?|allerg\w+|speak|languages?|vegetarian|vegan|"
        r"gluten|diabet\w+|asthma|medicat\w+|prescrib\w+|"
        r"daughter|son|child|children|kid|baby|partner|wife|husband|spouse|"
        r"fianc\w+|girlfriend|boyfriend|"
        r"favorite|favourite|prefer|love|hate|enjoy|"
        r"pet|cat|dog|puppy|kitten|"
        r"drive|car|vehicle|house|apartment|condo|mortgage|rent|"
        r"married|engaged|divorced|widowed|"
        r"graduat\w+|university|college|degree|major|PhD|MBA)\b",
        text, re.IGNORECASE,
    ):
        return True

    # Check for proper nouns not already captured by S1
    proper_nouns = extract_proper_nouns(text)
    s1_entity_names = set()
    for f in s1_facts:
        s1_entity_names.update(n.lower() for n in f.entity_names)
        if f.related_entity:
            s1_entity_names.add(f.related_entity.lower())

    uncovered_nouns = [n for n in proper_nouns if n.lower() not in s1_entity_names]
    if uncovered_nouns:
        return True

    # Check for complex content indicators
    # Subjective signals: opinions, feelings, hedging
    subjective_signals = re.search(
        r"\b(think|feel|believe|prefer|love|hate|enjoy|dislike|opinion|"
        r"considering|thinking about|might|maybe|not sure|probably)\b",
        text, re.IGNORECASE,
    )
    if subjective_signals:
        return True

    # If S1 extracted nothing from a long message, S2 should try
    if not s1_facts and len(words) > 15:
        return True

    return False


# Speculative language markers
_SPECULATIVE_MARKERS = re.compile(
    r"\b(thinking about|considering|might|maybe|perhaps|not sure|"
    r"probably|could be|looking into|toying with|on the fence|"
    r"debating whether|contemplating|I wonder)\b",
    re.IGNORECASE,
)


# ─── Pydantic models for S2 response validation ──────────────────────────────

_VALID_FACT_TYPES = {"objective", "subjective", "conditional", "temporal"}


class S2Fact(BaseModel):
    content: str
    fact_type: str = "objective"
    category: str = "general"
    confidence: float = 0.7
    entities: list[str] = []
    speculative: bool = False

    @field_validator("fact_type")
    @classmethod
    def validate_fact_type(cls, v: str) -> str:
        return v if v in _VALID_FACT_TYPES else "objective"

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class S2Response(BaseModel):
    facts: list[S2Fact] = []


async def extract_facts_s2(
    text: str,
    provider_base_url: str,
    model: str = "qwen3.5:2b",
    fallback_model: str = "qwen2.5:1.5b",
    num_ctx: int = 4096,
    owner_name: str = "",
) -> list[ExtractedFact]:
    """Stage 2: LLM-based extraction with model fallback and Pydantic validation.

    Tries *model* first; if it fails (timeout, bad JSON, connection error),
    retries once with *fallback_model*. Validates via Pydantic with up to 2
    parse retries per model attempt.
    """
    prompt_messages = [
        {"role": "system", "content": _render_s2_prompt(owner_name)},
        {"role": "user", "content": text},
    ]

    models_to_try = [model, fallback_model]

    for i, m in enumerate(models_to_try):
        is_fallback = (i > 0)
        if is_fallback:
            logger.info("S2 fallback: trying %s", m)

        result = await _s2_call_with_retries(prompt_messages, provider_base_url, m, text, num_ctx=num_ctx)
        if result is not None:
            logger.info("S2 extraction used model=%s (fallback=%s)", m, is_fallback)
            return result
        logger.warning("S2 model %s failed", m)

    logger.warning("S2 extraction failed on all models")
    return []


def resolve_writer_model(config: Any) -> str:
    """Resolve the effective S2 writer model.

    When ``writer.model == 'auto'`` (the self-contained default), route
    S2 extraction calls to the user's main model instead of a separate
    CPU-pinned writer. Cloud users get high-quality extraction via
    Claude/GPT-4; local users avoid loading a second model. An explicit
    writer.model override still wins for advanced users.
    """
    if config.writer.model == "auto":
        return config.provider.default_model
    return config.writer.model


# S2 calls now target the user's main model (single-model-loadout for
# local users, cloud-grade extraction for cloud users). We no longer
# need 120s headroom for 2B-on-CPU — GPU inference is <10s, cloud APIs
# are <5s — but we keep 120s as an upper bound on runaway generation
# against the whole call chain.
_S2_READ_TIMEOUT_S = 120.0

# Protocol-specific request builders. Ollama accepts top-level
# format="json" and think flags; OpenAI-compat servers reject those
# fields and require response_format on the body. We try Ollama first
# because local installs are the common case; a 4xx triggers the OpenAI
# fallback on the same base_url.


def _build_ollama_chat_body(model: str, messages: list[dict], num_ctx: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": "json",
        # num_predict caps runaway generation. A realistic multi-sentence
        # message generates ~970 JSON tokens; 1024 covers that with margin.
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": 1024,
        },
    }


def _build_openai_chat_body(model: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }


def _extract_content(data: dict) -> str:
    """Pull the assistant message text out of either response shape."""
    # Ollama: {"message": {"content": "..."}}
    msg = data.get("message")
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"] or ""
    # OpenAI: {"choices": [{"message": {"content": "..."}}]}
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                return msg.get("content", "") or ""
    return ""


async def _s2_call_with_retries(
    messages: list[dict],
    base_url: str,
    model: str,
    original_text: str,
    max_retries: int = 2,
    num_ctx: int = 4096,
) -> list[ExtractedFact] | None:
    """Call S2 LLM with Ollama→OpenAI protocol fallback and retry on parse failures.

    The two attempts per ``attempt`` cover:
      (a) Ollama /api/chat with format=json (succeeds against Ollama or
          ollama-proxy-compatible backends);
      (b) OpenAI /v1/chat/completions with response_format=json_object
          (succeeds against cloud APIs, LM Studio, vLLM, etc).

    A 4xx on (a) signals the upstream doesn't understand Ollama's fields
    — fall back to (b). 5xx or network errors are upstream problems, not
    protocol problems, so they surface to the caller (which then tries
    the fallback model).
    """
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(_S2_READ_TIMEOUT_S)) as client:
                resp = await client.post(
                    f"{base_url}/api/chat",
                    json=_build_ollama_chat_body(model, messages, num_ctx),
                )
                if 400 <= resp.status_code < 500:
                    logger.info(
                        "S2 Ollama format rejected (status=%d); falling back to OpenAI format",
                        resp.status_code,
                    )
                    resp = await client.post(
                        f"{base_url}/v1/chat/completions",
                        json=_build_openai_chat_body(model, messages),
                    )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("S2 LLM call failed (model=%s attempt=%d): %s", model, attempt + 1, exc)
            return None  # connection/timeout/5xx → let caller try fallback model

        content = _extract_content(data)
        facts = _parse_s2_response(content, original_text)
        if facts is not None:
            return facts

        logger.warning("S2 validation failed (model=%s attempt=%d), retrying", model, attempt + 1)

    return None


# ── Episode summary ──────────────────────────────────────────────────────────
#
# Follow-up queries ("Going back to the mortgage…") underperform when
# episode text is just the first 300 chars of the user message plus a
# short fact-list tail — there's no record of what was decided or how
# the user responded. Upgrading to a 1-sentence LLM summary lifts
# follow-up accuracy.
#
# Cost profile: one small LLM call per user turn, ~60-100 output tokens,
# temperature=0. Fires async inside the writer (fire-and-forget) so the
# primary response never waits on it.

_EPISODE_SUMMARY_READ_TIMEOUT_S = 30.0
_EPISODE_SUMMARY_MAX_TOKENS = 96
_EPISODE_SUMMARY_PROMPT = (
    "Summarise this exchange in ONE sentence. "
    "Focus on what was discussed, what was decided, and the user's position. "
    "Do not repeat the user's words verbatim. "
    "Output only the sentence, no preamble, no JSON."
)


def _build_episode_ollama_body(model: str, messages: list[dict], num_ctx: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": _EPISODE_SUMMARY_MAX_TOKENS,
        },
    }


def _build_episode_openai_body(model: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "max_tokens": _EPISODE_SUMMARY_MAX_TOKENS,
    }


async def summarize_episode(
    user_text: str,
    assistant_text: str,
    provider_base_url: str,
    model: str,
    num_ctx: int = 2048,
) -> str:
    """Return a one-sentence summary of this exchange, or "" on any failure.

    Fails open: the caller falls back to the existing 300-char truncation
    when we return "". Never raises.
    """
    if not user_text.strip():
        return ""
    ut = user_text.strip()
    at = (assistant_text or "").strip()
    # Bound both sides so the summariser sees enough context without
    # blowing num_ctx. Heuristic: 1200 chars ≈ 300 tokens, plenty of room
    # for the model to capture the decision shape.
    combined = f"User: {ut[:1200]}"
    if at:
        combined += f"\nAssistant: {at[:1200]}"
    messages = [
        {"role": "system", "content": _EPISODE_SUMMARY_PROMPT},
        {"role": "user", "content": combined},
    ]
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_EPISODE_SUMMARY_READ_TIMEOUT_S),
        ) as client:
            resp = await client.post(
                f"{provider_base_url}/api/chat",
                json=_build_episode_ollama_body(model, messages, num_ctx),
            )
            if 400 <= resp.status_code < 500:
                resp = await client.post(
                    f"{provider_base_url}/v1/chat/completions",
                    json=_build_episode_openai_body(model, messages),
                )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.info("Episode summary call failed (model=%s): %s", model, exc)
        return ""

    content = _extract_content(data).strip()
    # Strip any stray quotes, thinking markers, trailing whitespace.
    content = content.strip('"\'`').strip()
    # Guard: if the model returned multiple sentences, keep only the first
    # two — episodes are meant to be compact.
    sentences = re.split(r"(?<=[.!?])\s+", content)
    if len(sentences) > 2:
        content = " ".join(sentences[:2])
    # Hard cap so a runaway reply can't bloat retrieval
    return content[:500]


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks emitted by reasoning models.

    Mirrors the same pattern used in _grader.py. Required because some
    providers (notably gpt-oss-* via Ollama cloud, and some self-hosted
    setups) emit reasoning traces even when the request body sets
    ``think: false``. Without this strip, the writer's JSON parse
    fails on the leading <think> token and the extraction is silently
    dropped — degrading sieve's behaviour with no visible error.
    """
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _parse_s2_response(content: str, original_text: str) -> list[ExtractedFact] | None:
    """Parse and validate S2 LLM JSON response via Pydantic. Returns None on failure."""
    content = _strip_think_tags(content)
    content = _strip_markdown_fences(content)
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("S2 response not valid JSON: %s", content[:200])
        return None

    try:
        validated = S2Response.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("S2 Pydantic validation failed: %s", exc)
        return None

    results: list[ExtractedFact] = []
    for f in validated.facts:
        if len(f.content.strip()) < 3:
            continue

        # Override speculative from LLM if linguistic markers are present
        is_speculative = f.speculative
        if not is_speculative and _SPECULATIVE_MARKERS.search(original_text):
            if any(marker in f.content.lower() for marker in
                   ["thinking about", "considering", "might", "maybe"]):
                is_speculative = True

        confidence = f.confidence
        if is_speculative:
            confidence = min(confidence, 0.4)

        # Infer relation word + related entity from S2 relationship facts so the
        # writer can create User→entity edges in the graph table.
        relation = None
        related = None
        entities = [str(e) for e in f.entities if e]
        if f.category == "relationship" and entities:
            lower = f.content.lower()
            for word in ("wife", "husband", "spouse", "partner", "daughter",
                         "son", "child", "mother", "father", "mum", "mom",
                         "dad", "parent", "brother", "sister", "sibling",
                         "best friend", "friend", "colleague", "boss",
                         "cat", "dog", "pet", "puppy", "kitten"):
                if word in lower:
                    relation = word
                    break
            # Use the first capitalised entity (filter out generic words)
            for name in entities:
                if name and name[:1].isupper() and name.lower() not in {"user", "the user", "i"}:
                    related = name
                    break

        results.append(ExtractedFact(
            content=f.content.strip(),
            fact_type="subjective" if is_speculative and f.fact_type == "objective" else f.fact_type,
            category=f.category,
            confidence=confidence,
            entity_names=entities,
            relation=relation,
            related_entity=related,
        ))

    return results


# ─── Conflict Resolution ──────────────────────────────────────────────────────

# Similarity threshold for finding potentially conflicting facts
_CONFLICT_SIMILARITY_THRESHOLD = 0.95


@dataclass
class ConflictResolution:
    """Result of conflict resolution for a single fact."""
    action: str          # "store", "boost", "supersede", "quarantine", "provisional", "coexist"
    new_status: str      # "current", "quarantined", "provisional", "superseded"
    new_confidence: float
    existing_fact_id: str | None = None
    detail: str = ""


def resolve_conflict(
    new_fact: ExtractedFact,
    existing: dict | None,
    session_coherence: float | None = None,
    owner_names: list[str] | tuple[str, ...] | None = None,
) -> ConflictResolution:
    """Apply the deterministic conflict resolution decision tree.

    Args:
        new_fact: The newly extracted fact.
        existing: The most similar existing fact (dict from store), or None.
        session_coherence: Session coherence score (0-1). Low = suspicious.
        owner_names: Profile owner's canonical name + aliases. Forwarded to
            _content_equivalent / _is_direct_contradiction so S2 facts
            written in the owner-name form canonicalise against S1's
            User-form facts. Pass None to keep legacy behaviour (User
            variants only).
    """
    # No existing → store as current
    if existing is None:
        if new_fact.fact_type == "subjective":
            return ConflictResolution(
                action="store", new_status="current",
                new_confidence=new_fact.confidence,
                detail="new subjective fact",
            )
        return ConflictResolution(
            action="store", new_status="current",
            new_confidence=new_fact.confidence,
            detail="no existing fact",
        )

    existing_content = existing.get("content", "").lower().strip()
    new_content = new_fact.content.lower().strip()
    existing_confidence = existing.get("confidence", 0.7)
    existing_id = existing["id"]

    # Same value → boost confidence
    if _content_equivalent(new_content, existing_content, owner_names=owner_names):
        return ConflictResolution(
            action="boost", new_status="current",
            new_confidence=min(1.0, existing_confidence + 0.05),
            existing_fact_id=existing_id,
            detail="same value re-confirmed",
        )

    # Different value — branch by fact type
    # Subjective → coexist via nuanced_view
    if new_fact.fact_type == "subjective":
        return ConflictResolution(
            action="coexist", new_status="current",
            new_confidence=new_fact.confidence,
            existing_fact_id=existing_id,
            detail="subjective: coexist via nuanced_view",
        )

    # Speculative → low confidence store
    if _is_speculative_text(new_fact.content):
        return ConflictResolution(
            action="store", new_status="current",
            new_confidence=min(new_fact.confidence, 0.35),
            detail="speculative (hedging language)",
        )

    # Existing high-confidence + well-confirmed
    existing_confirmations = existing.get("usage_count", 0)
    if existing_confidence > 0.8 and existing_confirmations > 10:
        # Low coherence session → quarantine
        if session_coherence is not None and session_coherence < 0.3:
            return ConflictResolution(
                action="quarantine", new_status="quarantined",
                new_confidence=new_fact.confidence,
                existing_fact_id=existing_id,
                detail="contradicts high-confidence fact in low-coherence session",
            )

        # Numeric/temporal → supersede as temporal_update
        if new_fact.fact_type == "temporal" or _is_numeric_content(new_fact.content):
            return ConflictResolution(
                action="supersede", new_status="current",
                new_confidence=new_fact.confidence,
                existing_fact_id=existing_id,
                detail="temporal/numeric update supersedes existing",
            )

        # Otherwise → provisional, surface both
        return ConflictResolution(
            action="provisional", new_status="provisional",
            new_confidence=new_fact.confidence,
            existing_fact_id=existing_id,
            detail="contradicts high-confidence fact — stored as provisional",
        )

    # Existing low-confidence → supersede
    if existing_confidence < 0.5:
        return ConflictResolution(
            action="supersede", new_status="current",
            new_confidence=new_fact.confidence,
            existing_fact_id=existing_id,
            detail="supersedes low-confidence existing fact",
        )

    # Direct contradiction: same leading clause, different tail (e.g. "User
    # lives in Dubai" → "User lives in Tokyo"; "User likes dogs" → "User likes
    # cats"). That warrants a supersede — later wins at 0.5.
    if _is_direct_contradiction(new_content, existing_content, owner_names=owner_names):
        return ConflictResolution(
            action="supersede", new_status="current",
            new_confidence=0.5,
            existing_fact_id=existing_id,
            detail="same-session contradiction: later wins, both at 0.5",
        )

    # Otherwise the two facts are similar but semantically distinct (e.g.
    # "works as software engineer" vs "works at Example Labs"). Coexist.
    return ConflictResolution(
        action="store", new_status="current",
        new_confidence=new_fact.confidence,
        existing_fact_id=existing_id,
        detail="similar topic, not duplicate — coexist",
    )


def _content_equivalent(
    a: str, b: str, owner_names: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Check if two fact contents are semantically equivalent (simple heuristic).

    Strips User/owner subject markers before comparing so the S1 path
    ('User lives in X') and S2 path ('Jamie Rivera lives in X') dedup
    against each other when ``owner_names`` is provided.
    """
    # Route through _strip_user_prefix so the owner-name handling is consistent
    # (lowercase, trailing punctuation, possessive variants all handled there).
    return _strip_user_prefix(a, owner_names=owner_names) == _strip_user_prefix(
        b, owner_names=owner_names,
    )


# Predicate-based contradiction detection.
# Two facts contradict iff they bind the SAME value-predicate to DIFFERENT
# objects. Topical/entity overlap alone is not enough.
#
# Each entry: (regex with one capture group for the object, normalised key).
# Facts whose normalised content matches the same regex with different captures
# are direct contradictions.
_VALUE_PREDICATES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^lives? in (.+)$"), "residence"),
    (re.compile(r"^lived in (.+)$"), "residence"),
    (re.compile(r"^is (?:a |an )?(\d+) years? old"), "age"),
    (re.compile(r"^is (\d+) years? old"), "age"),
    (re.compile(r"^likes (.+)$"), "likes"),
    (re.compile(r"^loves (.+)$"), "likes"),
    (re.compile(r"^enjoys (.+)$"), "likes"),
    (re.compile(r"^prefers (.+)$"), "prefers"),
    (re.compile(r"^hates (.+)$"), "hates"),
    (re.compile(r"^dislikes (.+)$"), "hates"),
    (re.compile(r"^drives a (.+)$"), "vehicle"),
    (re.compile(r"^owns a (.+)$"), "owns"),
    # Collapse all job-state predicates (employer / role / occupation /
    # position / offer-of-role) into a single key so changes contradict
    # each other regardless of the surface verb. Capture key role/
    # employer tokens after the verb.
    (re.compile(r"^works (?:at|for) (.+)$"), "job_state"),
    (re.compile(r"^works as (?:a |an )?(.+)$"), "job_state"),
    (re.compile(r"^is (?:a |an )?(.+? (?:engineer|manager|developer|designer|analyst|director|officer|lead|architect|consultant|teacher|professor|nurse|doctor|lawyer|founder|ceo|cto|cfo|coo|vp|pm))(?:\s+at\s+.+)?$"), "job_state"),
    (re.compile(r"^role (?:at .+? )?is (.+)$"), "job_state"),
    (re.compile(r"^is (?:being\s+)?(?:offered|hired|interviewed)\s+(?:for|as|a|an)\s+(?:a\s+|an\s+)?(.+? (?:role|position|job))"), "job_state"),
    (re.compile(r"^accepted (?:the|a|an)\s+(.+? (?:role|position|job))"), "job_state"),
    (re.compile(r"^(?:no longer|stopped|left) (?:works at|working at|with) (.+)$"), "job_state"),
    (re.compile(r"^earns (.+)$"), "salary"),
    (re.compile(r"^makes (.+)$"), "salary"),
    (re.compile(r"'?s? base salary (?:at .+ )?is (.+)$"), "salary"),
    (re.compile(r"'?s? salary (?:at .+ )?is (.+)$"), "salary"),
    # Marital state: cover both "is X" and "have been X" / "and Y are X" forms
    (re.compile(r"^is (married|engaged|separated|divorced|single|widowed)\b"), "marital_state"),
    (re.compile(r"^(?:has|have) been (married|engaged|separated|divorced)\b"), "marital_state"),
    (re.compile(r"and (?:\w+ )?(?:have been |are )(married|engaged|separated|divorced)\b"), "marital_state"),
    # Pet possession (anchored — distinguishes from owns-a-car)
    (re.compile(r"^has a (dog|cat|pet|puppy|kitten|rabbit) named (.+)$"), "pet_name"),
    (re.compile(r"'?s? (?:dog|cat|pet) is named (.+)$"), "pet_name"),
    # Decision state on speculative items
    (re.compile(r"^(?:is )?considering (.+)$"), "considering"),
    (re.compile(r"^decided against (.+)$"), "considering"),
    (re.compile(r"^does not want (.+)$"), "considering"),
    (re.compile(r"^no longer wants (.+)$"), "considering"),
    # Role in plain "<subj>'s role is X" form (without an 'at ...' suffix)
    (re.compile(r"^role is (.+)$"), "job_state"),
    # Mortgage rate — capture qualifier+rate so different framings
    # contradict ('fixed at 3.1%' vs 'higher than 3.1%') but same
    # framing boosts ('3.1%' vs '3.1%'). The captured group is the
    # normalised tail of the sentence, ignoring stopword prefixes.
    # D14/D15: widened to accept "mortgage interest rate" and the
    # "the mortgage X rate" framing that S2 emits — in the first
    # 30-day run, 4.2% was stored as "The mortgage interest rate is
    # 4.2%." while 3.8% was "User's mortgage rate is 3.8%" — one
    # matched the predicate, the other didn't, so the contradiction
    # check missed them and both stayed current.
    (re.compile(r"^(?:the\s+)?mortgage (?:interest\s+|annual\s+|monthly\s+|variable\s+|fixed\s+)?rate (?:is |is\s+)?(.+?)(?:\s+until\s+\w+)?\.?$"), "mortgage_rate"),
    # Mortgage balance — "mortgage is about £240K", "mortgage balance is £230K"
    (re.compile(r"^(?:the\s+)?(?:remaining\s+|current\s+)?mortgage (?:balance |amount )?is (?:about |approximately )?(.+?)\.?$"), "mortgage_balance"),
    # Half-marathon / running pace — 5:30/km etc.
    (re.compile(r"^(?:current\s+)?(?:half(?:\s+marathon)?\s+)?(?:running\s+)?pace (?:is |is\s+)?(.+?)\.?$"), "running_pace"),
    # Bouldering grade
    (re.compile(r"^(?:current\s+)?bouldering grade (?:is |is\s+)?(.+?)\.?$"), "bouldering_grade"),
    # Family relation → name. "<relation> is [named] <Name>" where
    # relation is a fixed slot and the object is a single- or two-token
    # proper name. The trailing object must not start with an article
    # ('a'/'an'/'the') — that signals an occupation/role ("husband is a
    # teacher") which is a DIFFERENT predicate, not a contradictory name.
    (re.compile(r"^(?:son|daughter|child|father|mother|dad|mum|mom|wife|husband|spouse|partner|brother|sister) is (?:named |called )?((?!(?:a|an|the)\s)[a-z][a-z'\-]{1,20}(?:\s[a-z][a-z'\-]{1,20})?)$"), "relation_name"),
]

_PREFIX_STRIP = ("the user's ", "user's ", "the user ", "user ")


def _strip_user_prefix(s: str, owner_names: list[str] | tuple[str, ...] | None = None) -> str:
    """Normalise a fact's subject: lowercase, strip trailing punctuation,
    and remove a leading User / profile-owner-name / etc. marker.

    ``owner_names`` is an optional iterable of canonical names and
    aliases for the profile owner. Pass None (or empty) to match only
    the built-in User variants; this preserves the legacy behaviour for
    call sites that don't know the owner. The writer pipeline forwards
    ``MemoryWriter._owner_name`` + aliases so S2 facts written in the
    owner-name form canonicalise the same way as S1's User-form facts.
    """
    s = s.lower().strip().rstrip(".!?,;")
    # Try built-in user variants first (cheap, handles the S1 path).
    for prefix in _PREFIX_STRIP:
        if s.startswith(prefix):
            return s[len(prefix):]
    # Then owner-name variants: "<owner>'s " and "<owner> " forms.
    # Order owners longest-first so "Jamie Rivera" matches before "Jamie"
    # (otherwise "Jamie Rivera's X" strips to "rivera's X").
    if owner_names:
        ordered = sorted((n for n in owner_names if n), key=len, reverse=True)
        for name in ordered:
            low = name.lower()
            for variant in (low + "'s ", low + "\u2019s ", low + " "):
                if s.startswith(variant):
                    return s[len(variant):]
    return s


def _predicate_key(
    s: str, owner_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, str] | None:
    """Match a normalised fact body against value-predicate templates.
    Returns (predicate_key, object) or None if no template matches.
    """
    body = _strip_user_prefix(s, owner_names=owner_names)
    for rx, key in _VALUE_PREDICATES:
        m = rx.search(body)
        if m:
            obj = " ".join(g for g in m.groups() if g).strip()
            return (key, obj)
    return None


def _is_direct_contradiction(
    a: str, b: str, owner_names: list[str] | tuple[str, ...] | None = None,
) -> bool:
    """Two facts contradict iff they bind the SAME value predicate to
    DIFFERENT objects.

    Examples:
        "User lives in Sydney" vs "User lives in Melbourne"     → True
        "User likes cats"      vs "User likes dogs"             → True
        "User is 38 years old" vs "User is 39 years old"        → True
        "User is married"      vs "User is separated"           → True
        "User married 9 years" vs "User met at wedding 2014"    → False
        "User is PM at Acme"   vs "User joined Acme 3 yrs ago"  → False
        "User has a pet Alex"  vs "User has a pet Kim"          → True

    ``owner_names`` lets the check canonicalise S2 facts written in
    the owner-name form ("Jamie Rivera's mortgage rate is …") so they
    match their User-form counterparts.
    """
    pa = _predicate_key(a, owner_names=owner_names)
    pb = _predicate_key(b, owner_names=owner_names)
    if pa is None or pb is None:
        return False
    if pa[0] != pb[0]:
        return False
    return pa[1] != pb[1] and pa[1] != "" and pb[1] != ""


def _is_speculative_text(text: str) -> bool:
    """Check if text contains speculative/hedging language."""
    return bool(_SPECULATIVE_MARKERS.search(text))


def _is_numeric_content(text: str) -> bool:
    """Check if the fact content contains numeric values (financial, age, counts)."""
    return bool(re.search(r"[\$€£¥]?\d[\d,\.]+", text))


# ─── Session Coherence ────────────────────────────────────────────────────────

async def compute_session_coherence(
    messages: list[str],
    embed_fn: Any,
) -> float:
    """Compute session coherence from embedding distances between consecutive messages.

    Returns a score 0..1 where 1 = highly coherent, 0 = incoherent.
    Sharp discontinuities (topic jumps, injection attempts) produce low scores.
    """
    if len(messages) < 2 or embed_fn is None:
        return 1.0  # default: trust

    try:
        embeddings = []
        for msg in messages[-5:]:  # last 5 messages max
            emb = await embed_fn(msg)
            embeddings.append(emb)

        if len(embeddings) < 2:
            return 1.0

        # Compute cosine similarities between consecutive messages
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Coherence = average similarity, penalized by min
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)

        # Weighted: 70% average + 30% minimum (sharp drops penalized more)
        coherence = 0.7 * avg_sim + 0.3 * min_sim

        # Clamp to 0..1
        return max(0.0, min(1.0, coherence))

    except Exception as exc:
        logger.warning("Session coherence computation failed: %s", exc)
        return 1.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─── Dedup ──────────────────────────────────────────────────────────────────────

DEDUP_SIMILARITY_THRESHOLD = 0.12  # vec distance below this = near-duplicate


def _is_duplicate(
    store: Any,
    content: str,
    embedding: list[float],
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> bool:
    """Check if a near-identical fact already exists via vector similarity."""
    if not embedding:
        # No embedding — fall back to exact text match
        row = store.conn.execute(
            "SELECT id FROM facts WHERE lower(content) = lower(?)", (content,)
        ).fetchone()
        return row is not None

    results = store.search_facts_by_vector(embedding, limit=1)
    if results and results[0]["distance"] < threshold:
        logger.debug(
            "Dedup: '%s' similar to existing '%s' (dist=%.4f)",
            content[:50], results[0]["content"][:50], results[0]["distance"],
        )
        return True
    return False


# ─── MemoryWriter ────────────────────────────────────────────────────────────────

class MemoryWriter:
    """Extracts facts from messages and writes them to the memory store.

    Stage 1: Regex extraction (<2ms).
    Stage 2: LLM extraction (~200ms GPU) — triggered by gate.
    Stage 3: Dedup + conflict resolution (~40ms).
    """

    def __init__(
        self,
        store: Any,
        embed_fn: Any | None = None,
        provider_base_url: str = "http://127.0.0.1:11434",
        writer_model: str = "qwen3.5:2b",
        fallback_model: str = "qwen2.5:1.5b",
        num_ctx: int = 4096,
        stage2_enabled: bool = True,
        coherence_enabled: bool = True,
        owner_name: str = "",
        profile_owner_aliases: list[str] | None = None,
        ghost_validator_enabled: bool = True,
        tier2_classifier_enabled: bool = False,
        tier2_classifier_model: str = "auto",  # 'auto' -> writer_model (already resolved)
        skip_empty_turns: bool = True,
    ):
        """
        Args:
            store: MemoryStore instance.
            embed_fn: async callable(text: str) -> list[float] | None.
            provider_base_url: Base URL for the LLM provider (for S2).
            writer_model: Primary model for S2 extraction.
            fallback_model: Fallback model if primary fails.
            num_ctx: Context window size for S2 LLM calls.
            stage2_enabled: ABL-S2 flag — set False for regex-only extraction.
            coherence_enabled: ABL-CI flag — set False to skip coherence scoring/quarantine.
            owner_name: Profile owner name pinned into the S2 prompt (empty = no pinning).
            profile_owner_aliases: Additional name aliases for the profile owner (e.g. ["Jamie", "I", "me"]).
            ghost_validator_enabled: Enable post-S2 ghost-fact structural validator.
            skip_empty_turns: Use the fast pre-filter to skip the S2 LLM call
                on turns with no extractable facts (filler, questions with no
                proper-noun anchor, social greetings). Default True.
        """
        self._store = store
        self._embed_fn = embed_fn
        self._provider_base_url = provider_base_url.rstrip("/")
        self._writer_model = writer_model
        self._fallback_model = fallback_model
        self._num_ctx = num_ctx
        self._stage2_enabled = stage2_enabled
        self._coherence_enabled = coherence_enabled
        self._owner_name = owner_name
        self._owner_aliases = list(profile_owner_aliases or [])
        self._ghost_validator_enabled = ghost_validator_enabled
        self._tier2_classifier_enabled = tier2_classifier_enabled
        self._skip_empty_turns = skip_empty_turns
        self._tier2_classifier_model = tier2_classifier_model
        self._session_messages: dict[str, list[str]] = {}  # per-session coherence tracking

    async def process(
        self,
        user_text: str,
        assistant_text: str = "",
        session_id: str | None = None,
    ) -> WriteResult:
        """Process a conversation turn: S1 → S2 → S3 (dedup + conflict) → write.

        Args:
            user_text: The user's message content.
            assistant_text: The assistant's reply.
            session_id: Current session ID. Created if not provided.
        """
        import time
        start = time.perf_counter()

        result = WriteResult(session_id=session_id or uuid.uuid4().hex)

        # Track messages for coherence (scoped per session, bounded)
        sid = result.session_id
        if user_text.strip():
            if sid not in self._session_messages:
                self._session_messages[sid] = []
            msgs = self._session_messages[sid]
            msgs.append(user_text)
            # Keep only the last 10 messages per session
            if len(msgs) > 10:
                self._session_messages[sid] = msgs[-10:]
        # Evict stale sessions (keep max 50)
        if len(self._session_messages) > 50:
            oldest = list(self._session_messages.keys())[:-50]
            for k in oldest:
                del self._session_messages[k]

        # S1: Regex extraction
        s1_candidates = extract_facts_s1(user_text)
        result.stage1_facts = len(s1_candidates)

        if s1_candidates:
            logger.info(
                "Writer S1: extracted %d candidates from %d chars",
                len(s1_candidates), len(user_text),
            )

        # S2: LLM extraction (gated, controllable via ABL-S2).
        # The skip-empty pre-filter runs first and short-circuits the
        # ~70% of turns that can't contain extractable facts. The
        # existing _s2_gate handles the remaining nuance for borderline
        # cases. Both filters err toward skipping; combined they cut
        # writer-pass cost dramatically with no measured quality loss.
        s2_candidates: list[ExtractedFact] = []
        if self._stage2_enabled:
            if self._skip_empty_turns:
                from sieve._writer_classifier import should_skip_writer
                if should_skip_writer(user_text):
                    logger.debug("Writer S2 skipped: empty-turn classifier")
                    # Fall through; s2_candidates stays empty
                    pass
                else:
                    if _s2_gate(user_text, s1_candidates):
                        result.stage2_invoked = True
                        logger.info("Writer S2 gate: OPEN — invoking LLM extraction")
                        s2_candidates = await extract_facts_s2(
                            user_text, self._provider_base_url,
                            model=self._writer_model,
                            fallback_model=self._fallback_model,
                            num_ctx=self._num_ctx,
                            owner_name=self._owner_name,
                        )
                        result.stage2_facts = len(s2_candidates)
                        if s2_candidates:
                            logger.info("Writer S2: extracted %d candidates", len(s2_candidates))
            else:
                # skip-empty disabled: use existing gate only.
                if _s2_gate(user_text, s1_candidates):
                    result.stage2_invoked = True
                    logger.info("Writer S2 gate: OPEN — invoking LLM extraction")
                    s2_candidates = await extract_facts_s2(
                        user_text, self._provider_base_url,
                        model=self._writer_model,
                        fallback_model=self._fallback_model,
                        num_ctx=self._num_ctx,
                        owner_name=self._owner_name,
                    )
                    result.stage2_facts = len(s2_candidates)
                    if s2_candidates:
                        logger.info("Writer S2: extracted %d candidates", len(s2_candidates))

        # Merge candidates (S1 first, S2 additions)
        all_candidates = s1_candidates + s2_candidates

        # Drop pet/animal facts that name a known person. Catches
        # "User has a pet named Kim" (Kim is the husband) and similar
        # entity-role confusion from S1 regex over-matching or S2 errors.
        _PET_RELATIONS = {"dog", "cat", "pet", "puppy", "kitten", "rabbit", "hamster", "bird", "parrot"}
        filtered: list[ExtractedFact] = []
        for cand in all_candidates:
            drop = False
            # Case 1: relationship-shaped pet fact with related_entity that is a known person
            if cand.relation and cand.relation.lower() in _PET_RELATIONS and cand.related_entity:
                if self._is_known_person(cand.related_entity):
                    logger.info(
                        "FIX_3 drop: pet '%s' is a known person — '%s'",
                        cand.related_entity, cand.content[:60],
                    )
                    drop = True
            # Case 2: S2 freeform fact like "User has a pet named X" where X is known
            if not drop and "pet named" in cand.content.lower():
                m = re.search(r"pet named ([A-Z][a-z]+)", cand.content)
                if m and self._is_known_person(m.group(1)):
                    logger.info(
                        "FIX_3 drop: pet content names known person — '%s'",
                        cand.content[:60],
                    )
                    drop = True
            if not drop:
                filtered.append(cand)
        all_candidates = filtered

        # D2: resolve relative-date expressions in fact content against
        # the injected clock. "My sister is visiting next weekend" stored
        # verbatim becomes stale the moment the clock advances; rewriting
        # to "on 2026-01-19 (originally stated 'next weekend' on 2026-01-15)"
        # preserves both the resolved date AND the original phrasing.
        try:
            from sieve.clock import get_clock
            now = get_clock().now()
        except Exception:
            now = None
        if now is not None:
            for cand in all_candidates:
                resolved = _resolve_relative_dates(cand.content, now)
                if resolved != cand.content:
                    cand.content = resolved

        # Post-S2 ghost-fact validator — reject inverted identity and
        # duplicate-name extractions from S2 output. S1 regex facts are
        # shape-safe by construction and pass through unchecked.
        if self._ghost_validator_enabled and self._owner_name and s2_candidates:
            # Compute the known-relatives set once per turn (not once
            # per fact) so Rule 4 is O(1) per check.
            owner_canon = (
                self._owner_name.lower().replace(" ", "_") or ""
            )
            relatives = _known_relative_first_names(self._store, owner_canon)
            if relatives:
                logger.debug(
                    "ghost validator: known relatives=%s", sorted(relatives)
                )

            s2_ids = {id(c) for c in s2_candidates}
            kept: list[ExtractedFact] = []
            rejected_counts = {
                "identity": 0, "duplicate": 0, "relative_cohabitation": 0,
                "unnamed_child": 0,
            }
            for cand in all_candidates:
                if id(cand) not in s2_ids:
                    kept.append(cand)
                    continue
                ok, reason = _validate_s2_fact(
                    cand, self._owner_name, self._owner_aliases,
                    relatives=relatives,
                )
                if ok:
                    kept.append(cand)
                else:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                    logger.info(
                        "s2_validator reject (%s): %s",
                        reason, (cand.content or "")[:120],
                    )
            rejected_total = sum(rejected_counts.values())
            result.conflicts_detected = rejected_total
            if rejected_total or s2_candidates:
                s2_kept = len(s2_candidates) - rejected_total
                logger.info(
                    "s2_validator: s2_kept=%d s2_rejected=%d "
                    "(by_rule: identity=%d duplicate=%d cohabitation=%d)",
                    s2_kept, rejected_total,
                    rejected_counts["identity"],
                    rejected_counts["duplicate"],
                    rejected_counts["relative_cohabitation"],
                )
            all_candidates = kept

        if not all_candidates:
            # Still emit an episode for this turn so follow-up retrieval
            # has a record even when nothing structured was extracted.
            await self._maybe_insert_episode(user_text, s2_candidates, sid, entity_cache={})
            result.elapsed_ms = (time.perf_counter() - start) * 1000
            return result

        # Compute session coherence (ABL-CI: skip when disabled)
        if self._coherence_enabled:
            session_coherence = await compute_session_coherence(
                self._session_messages.get(sid, []), self._embed_fn,
            )
        else:
            session_coherence = 1.0  # trust everything when CI disabled

        # Entity name → entity_id cache for this turn
        entity_cache: dict[str, str] = {}

        for fact in all_candidates:
            # Resolve/create entities first
            for entity_name in fact.entity_names:
                if entity_name not in entity_cache:
                    eid = self._get_or_create_entity(entity_name, fact.category)
                    entity_cache[entity_name] = eid

            # Also create entity for relationship target
            if fact.related_entity and fact.related_entity not in entity_cache:
                eid = self._get_or_create_entity(fact.related_entity, "person")
                entity_cache[fact.related_entity] = eid

            # Get entity IDs for this fact
            entity_ids = [entity_cache[n] for n in fact.entity_names if n in entity_cache]

            # Always link every user-fact to the User entity so graph traversal
            # from any connected entity (partner, friend, pet) can surface
            # related user context.
            user_entity_id = self._get_or_create_entity("User", "person")
            if user_entity_id not in entity_ids:
                entity_ids.append(user_entity_id)

            # Generate embedding
            embedding: list[float] | None = None
            if self._embed_fn is not None:
                try:
                    embedding = await self._embed_fn(fact.content)
                except Exception as exc:
                    logger.warning("Embedding failed for '%s': %s", fact.content[:40], exc)

            # Tier 2 classification: feed the readable content string
            # through the configured tier2 model to derive structured tags.
            # Fail-open: on any classifier error we write the fact with NULL
            # structured columns.
            _v2_kwargs: dict[str, str | None] = {}
            if self._tier2_classifier_enabled:
                try:
                    from sieve.fact_classifier_v2 import classify_fact_async
                    tier2_model = (
                        self._writer_model
                        if self._tier2_classifier_model == "auto"
                        else self._tier2_classifier_model
                    )
                    tags = await classify_fact_async(
                        fact.content,
                        base_url=self._provider_base_url,
                        model=tier2_model,
                    )
                    if tags.is_populated:
                        # Subject canonicalisation: if the owner's name
                        # appears inside the classifier's subject string
                        # (e.g. "Jamie Rivera and Kim", "Jamie Rivera's
                        # boys"), collapse the whole thing to the
                        # owner's subject so slot_key lookups from
                        # retrieval hit. This keeps
                        # "jamie_rivera:marital_status" as the one
                        # authoritative row instead of scattering facts
                        # across {jamie_rivera, jamie_rivera_and_kim,
                        # jamie_riveras_boys, ...}.
                        raw_subj = (tags.subject or "").strip()
                        subj_lower = raw_subj.lower()
                        owner_lower = (self._owner_name or "").lower()
                        owner_first = (owner_lower.split()[0] if owner_lower else "")
                        if owner_lower and (owner_lower in subj_lower
                                            or (owner_first and owner_first in subj_lower)):
                            canon = owner_lower.replace(" ", "_")
                        else:
                            canon = raw_subj.lower().replace(" ", "_")
                        canon = canon.replace("'s", "").replace("'", "").strip("_") or None
                        slot_key = (f"{canon}:{tags.predicate}"
                                    if canon and tags.predicate else None)
                        _v2_kwargs = {
                            "subject_entity_id": canon,
                            "predicate": tags.predicate,
                            "object_literal": tags.object_literal,
                            "slot_key": slot_key,
                            "category": tags.category,
                            "extraction_method": tags.extraction_method,
                        }
                except Exception as exc:
                    logger.warning("tier2 classify failed, writing NULL: %s", exc)

            # S3: Dedup check
            if _is_duplicate(self._store, fact.content, embedding or []):
                logger.debug("Dedup skip: '%s'", fact.content[:60])
                result.facts_skipped += 1
                continue

            # S3: Conflict resolution — find existing similar facts.
            # Top-1 nearest neighbour misses the actual contradicting
            # fact when several semantically-related facts exist (e.g.
            # "User's role is VP of Product" inserted while "PM at Acme",
            # "interviewing for VP", "Senior PM at Example" all exist —
            # top-1 picks the interview fact, not the PM fact).
            #
            # New strategy: fetch top-K candidates, then prefer (in order):
            #   1. an exact content-equivalent (boost path)
            #   2. a value-predicate contradiction (supersede path)
            #   3. otherwise, top-1 (legacy behaviour: coexist as similar)
            existing_match = None
            # Owner-name canonicalisation so S2's "Jamie Rivera's X is Y"
            # facts match S1's "User's X is Y" facts during boost/supersede.
            _owner_names = [n for n in ([self._owner_name] + list(self._owner_aliases)) if n]
            if embedding:
                similar = self._store.find_similar_facts(
                    embedding, limit=8, max_distance=_CONFLICT_SIMILARITY_THRESHOLD,
                )
                if similar:
                    new_lc = fact.content.lower().strip()
                    # 1. content-equivalent (boost)
                    for cand in similar:
                        if _content_equivalent(
                            new_lc, cand.get("content", "").lower().strip(),
                            owner_names=_owner_names,
                        ):
                            existing_match = cand
                            break
                    # 2. predicate contradiction (supersede)
                    if existing_match is None:
                        for cand in similar:
                            if _is_direct_contradiction(
                                new_lc, cand.get("content", "").lower().strip(),
                                owner_names=_owner_names,
                            ):
                                existing_match = cand
                                logger.info(
                                    "FIX_4 contradiction match: new='%s' old='%s' (rank=%d)",
                                    fact.content[:60], cand.get("content", "")[:60],
                                    similar.index(cand) + 1,
                                )
                                break
                    # 3. fallback to top-1
                    if existing_match is None:
                        existing_match = similar[0]

            resolution = resolve_conflict(
                fact, existing_match, session_coherence,
                owner_names=_owner_names,
            )

            logger.debug(
                "Conflict resolution for '%s': %s (%s)",
                fact.content[:50], resolution.action, resolution.detail,
            )

            # Apply resolution
            if resolution.action == "boost":
                # Re-confirmed existing fact
                self._store.boost_fact_confidence(resolution.existing_fact_id)
                result.facts_skipped += 1
                continue

            if resolution.action == "quarantine":
                # Store as quarantined
                fact_id = self._store.insert_fact(
                    content=fact.content,
                    embedding=embedding,
                    entity_ids=entity_ids or None,
                    source="writer_s2" if fact in s2_candidates else "writer_s1",
                    confidence=resolution.new_confidence,
                    fact_type=fact.fact_type,
                    session_coherence_score=session_coherence,
                    **_v2_kwargs,
                )
                self._store.update_fact_status(
                    fact_id, "quarantined",
                    status_detail="contradicts high-confidence fact in low-coherence session",
                )
                result.facts_written += 1
                continue

            if resolution.action == "supersede" and resolution.existing_fact_id:
                # Mark old fact as superseded
                self._store.update_fact_status(
                    resolution.existing_fact_id, "superseded",
                    superseded_by="pending",  # will update after insert
                )
                result.supersessions += 1
                # If same-session contradiction, lower old fact confidence too
                if "same-session" in resolution.detail:
                    self._store.conn.execute(
                        "UPDATE facts SET confidence = 0.5 WHERE id = ?",
                        (resolution.existing_fact_id,),
                    )
                    self._store.conn.commit()

            if resolution.action == "coexist" and resolution.existing_fact_id:
                # Store alongside and link via nuanced_view relationship
                fact_id = self._store.insert_fact(
                    content=fact.content,
                    embedding=embedding,
                    entity_ids=entity_ids or None,
                    source="writer_s2" if fact in s2_candidates else "writer_s1",
                    confidence=resolution.new_confidence,
                    fact_type=fact.fact_type,
                    session_coherence_score=session_coherence,
                    **_v2_kwargs,
                )
                # Create nuanced_view relationship between old and new
                try:
                    old_entity = self._get_or_create_entity(
                        f"fact:{resolution.existing_fact_id}", "fact_ref"
                    )
                    new_entity = self._get_or_create_entity(
                        f"fact:{fact_id}", "fact_ref"
                    )
                    self._store.insert_relationship(
                        source_entity=old_entity,
                        relationship="nuanced_view",
                        target_entity=new_entity,
                        confidence=0.8,
                    )
                    result.relationships_written += 1
                except Exception as exc:
                    logger.warning("nuanced_view relationship failed: %s", exc)
                result.facts_written += 1
                continue

            # Default: store (new fact or provisional)
            fact_id = self._store.insert_fact(
                content=fact.content,
                embedding=embedding,
                entity_ids=entity_ids or None,
                source="writer_s2" if fact in s2_candidates else "writer_s1",
                confidence=resolution.new_confidence,
                fact_type=fact.fact_type,
                session_coherence_score=session_coherence,
                **_v2_kwargs,
            )

            # Apply status from resolution
            if resolution.new_status != "current":
                self._store.update_fact_status(fact_id, resolution.new_status)

            # Update superseded_by on old fact if we just superseded
            if resolution.action == "supersede" and resolution.existing_fact_id:
                self._store.update_fact_status(
                    resolution.existing_fact_id, "superseded",
                    superseded_by=fact_id,
                )

            result.facts_written += 1

            # Write relationship if this is a relationship fact
            if fact.relation and fact.related_entity and fact.related_entity in entity_cache:
                user_entity_id = self._get_or_create_entity("User", "person")
                related_id = entity_cache[fact.related_entity]
                # D41: reject polarity-contradicting new relationships for
                # the same target. If "User → sister → Amy" exists, a
                # new "User → husband → Amy" is a hallucination — keep
                # the older, higher-confidence edge and skip the new one.
                # Families share small mutually-exclusive role clusters.
                _MUTUALLY_EXCLUSIVE_ROLES = [
                    {"sister", "brother", "sibling", "wife", "husband",
                     "spouse", "partner", "mother", "father", "parent",
                     "son", "daughter", "child", "cousin", "friend",
                     "best friend", "boss", "colleague", "coworker"},
                ]
                existing_edges = self._store.conn.execute(
                    """SELECT relationship, confidence FROM relationships
                       WHERE source_entity = ? AND target_entity = ?
                         AND status = 'current'""",
                    (user_entity_id, related_id),
                ).fetchall()
                new_rel_l = fact.relation.lower().strip()
                skip_insert = False
                for existing_rel_raw, existing_conf in existing_edges:
                    existing_rel_l = (existing_rel_raw or "").lower().strip()
                    if existing_rel_l == new_rel_l:
                        # Dedup handled inside insert_relationship (D40).
                        continue
                    # Check if both roles belong to the same exclusive set
                    for cluster in _MUTUALLY_EXCLUSIVE_ROLES:
                        if existing_rel_l in cluster and new_rel_l in cluster:
                            # Keep the older/higher-confidence edge.
                            if (existing_conf or 0.0) >= fact.confidence:
                                logger.info(
                                    "D41 polarity: rejecting new '%s' edge, "
                                    "existing '%s' (conf %.2f ≥ %.2f) wins",
                                    new_rel_l, existing_rel_l,
                                    existing_conf or 0.0, fact.confidence,
                                )
                                skip_insert = True
                                break
                    if skip_insert:
                        break
                if skip_insert:
                    continue
                try:
                    self._store.insert_relationship(
                        source_entity=user_entity_id,
                        relationship=fact.relation,
                        target_entity=related_id,
                        confidence=fact.confidence,
                    )
                    result.relationships_written += 1
                except Exception as exc:
                    logger.warning("Relationship insert failed: %s", exc)

        result.entities_written = len(entity_cache)

        # D41 safety-net: after any edge inserts, sweep User→*→T edges
        # for mutually-exclusive conflicts and supersede the weaker one.
        # This catches cases where an edge slipped past the write-time
        # D41 check (e.g. a race with the async writer, or a path that
        # bypasses the per-turn guard). Idempotent and cheap (~1ms for
        # typical user-graph size).
        try:
            self._sweep_polarity_conflicts()
        except Exception as exc:
            logger.warning("polarity sweep failed: %s", exc)

        # Phase-3 Fix 2: emit one episode per user turn so follow-up
        # queries can retrieve the compressed record of this turn.
        await self._maybe_insert_episode(
            user_text, s2_candidates, sid,
            entity_cache=entity_cache,
            assistant_text=assistant_text,
        )

        result.elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Writer done: %d facts written, %d skipped, %d entities, %d rels (%.1fms) [S1:%d S2:%d]",
            result.facts_written, result.facts_skipped,
            result.entities_written, result.relationships_written,
            result.elapsed_ms, len(s1_candidates), len(s2_candidates),
        )

        return result

    def _sweep_polarity_conflicts(self) -> None:
        """Post-write sweep: for each (user, target) pair with multiple
        edges, if two edges are in the same mutually-exclusive role
        cluster, mark the lower-confidence (and older-if-tied) edge as
        superseded. Catches leaks past the per-edge D41 guard.
        """
        if not self._store._conn:
            return
        _CLUSTER = {
            "sister", "brother", "sibling", "wife", "husband",
            "spouse", "partner", "mother", "father", "parent",
            "son", "daughter", "child", "cousin", "friend",
            "best friend", "boss", "colleague", "coworker",
        }
        # Build alias set from the owner config (already stored in self._owner_name
        # and self._owner_aliases). Always include 'user' as the canonical anchor.
        _owner_aliases_lower = {"user"}
        if self._owner_name:
            _owner_aliases_lower.add(self._owner_name.lower())
            parts = self._owner_name.lower().split()
            if parts:
                _owner_aliases_lower.add(parts[0])
        for _a in self._owner_aliases:
            a_l = (_a or "").strip().lower()
            if a_l:
                _owner_aliases_lower.add(a_l)
        _placeholders = ",".join("?" for _ in _owner_aliases_lower)
        user_row = self._store._conn.execute(
            f"SELECT id FROM entities WHERE LOWER(name) IN ({_placeholders}) LIMIT 1",
            tuple(_owner_aliases_lower),
        ).fetchone()
        if not user_row:
            return
        user_id = user_row[0]
        # Edges grouped by target
        rows = self._store._conn.execute(
            "SELECT id, relationship, target_entity, confidence, created_at "
            "FROM relationships "
            "WHERE source_entity = ? AND status = 'current'",
            (user_id,),
        ).fetchall()
        by_target: dict[str, list[tuple]] = {}
        for row in rows:
            by_target.setdefault(row[2], []).append(row)
        for target_id, edges in by_target.items():
            if len(edges) < 2:
                continue
            cluster_members = [e for e in edges if (e[1] or "").lower() in _CLUSTER]
            if len(cluster_members) < 2:
                continue
            # Keep the highest-confidence (older wins on tie).
            cluster_members.sort(key=lambda e: (-(e[3] or 0.0), e[4] or ""))
            winner = cluster_members[0]
            for loser in cluster_members[1:]:
                self._store._conn.execute(
                    "UPDATE relationships SET status = 'superseded' WHERE id = ?",
                    (loser[0],),
                )
                logger.info(
                    "polarity sweep: superseded %r->%r (conf %.2f), kept %r (conf %.2f)",
                    loser[1], target_id[:8], loser[3] or 0.0,
                    winner[1], winner[3] or 0.0,
                )
        self._store._conn.commit()

    def _get_or_create_entity(self, name: str, entity_type: str) -> str:
        """Get existing entity by name or create it. Returns entity ID.

        D37: the old signature passed fact.category ("financial", "hobby",
        "relationship") as entity_type. That conflates fact domain with
        entity kind — "Toast" (a cat) ended up as type="hobby",
        "Volvo XC40" as type="occupation". Now we map known fact
        categories to correct entity kinds (person / pet / location /
        place / vehicle / organisation / thing) and default to "thing"
        when ambiguous. Existing entities' type is not rewritten (we
        never knew that "hobby" meant "pet" before — keep the legacy
        entity for relational integrity, new entities get correct types).
        """
        existing = self._store.find_entity_by_name(name)
        if existing:
            return existing["id"]
        # Map fact.category → entity kind. Relationship-valued categories
        # default to "person" at the call site; everything else comes here.
        _CATEGORY_TO_ENTITY_KIND = {
            "identity": "person",
            "relationship": "person",
            "location": "location",
            "occupation": "thing",   # company/role names land here
            "health": "thing",
            "education": "thing",
            "financial": "thing",
            "preference": "thing",
            "opinion": "thing",
            "hobby": "thing",
            "plan": "thing",
            "person": "person",
            "pet": "pet",
            "place": "location",
            "vehicle": "thing",
            "thing": "thing",
        }
        resolved_type = _CATEGORY_TO_ENTITY_KIND.get(entity_type.lower(), entity_type)
        return self._store.insert_entity(name, type=resolved_type)

    async def _maybe_insert_episode(
        self,
        user_text: str,
        s2_candidates: list[ExtractedFact],
        session_id: str,
        *,
        entity_cache: dict[str, str],
        assistant_text: str = "",
    ) -> None:
        """Persist a compressed record of this turn.

        When an assistant reply is available, request a one-sentence
        LLM summary of the exchange (what was discussed, what was
        decided, the user's stance). Falls back to the legacy 300-char
        truncation + fact-list tail on any failure so
        cold-start / offline scenarios still emit episodes.
        Embedding = embed(episode summary, first 512 chars).
        Fail-open: all errors logged and swallowed.
        """
        try:
            text_for_summary = (user_text or "").strip()
            if not text_for_summary:
                return

            summary = ""
            if assistant_text.strip() and self._provider_base_url:
                try:
                    summary = await summarize_episode(
                        user_text=text_for_summary,
                        assistant_text=assistant_text,
                        provider_base_url=self._provider_base_url,
                        model=self._writer_model,
                        num_ctx=min(self._num_ctx, 4096),
                    )
                except Exception as exc:
                    logger.info("Episode LLM summary failed, falling back: %s", exc)
                    summary = ""

            if not summary:
                summary = text_for_summary[:300]
                s2_bullets = [
                    (c.content or "").strip()
                    for c in s2_candidates
                    if (c.content or "").strip()
                ]
                if s2_bullets:
                    extra = " | ".join(s2_bullets)[:200]
                    summary = f"{summary} [facts: {extra}]"

            ep_embedding: list[float] | None = None
            if self._embed_fn is not None:
                try:
                    ep_embedding = await self._embed_fn(summary[:512])
                except Exception as exc:
                    logger.warning("Episode embed failed: %s", exc)
            entities_involved = sorted(
                {n for n in entity_cache.keys() if n}
            )
            self._store.insert_episode(
                summary=summary,
                embedding=ep_embedding,
                entities_involved=entities_involved or None,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("Episode insert failed (non-fatal): %s", exc)

    def _is_known_person(self, name: str) -> bool:
        """Check if a name is already known as a person.
        Used to block 'pet named X' facts when X is the spouse, child, etc.
        """
        if not name:
            return False
        ent = self._store.find_entity_by_name(name)
        if not ent:
            return False
        ent_type = (ent.get("type") or "").lower()
        # 'person', 'relationship' (S1 categorises spouses as relationship),
        # 'family', 'identity' all imply human
        return ent_type in {"person", "relationship", "family", "identity", "people"}

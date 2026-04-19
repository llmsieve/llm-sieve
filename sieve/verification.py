"""Cycle 19: Response Verification Layer (RVL).



Three-layer anti-hallucination architecture:

Layer 1 — Pre-generation Absence Signalling (`build_absence_signals`)
    Scans the user's query for entity references that are NOT supported by
    retrieved facts. Two checks:
      (a) relationship words ("daughter", "sister", "wife") that the user
          does not have in the store
      (b) proper nouns in the query that are not known store entities
    Returns a list of negative-context lines to inject before LLM call.

Layer 2 — Pre-generation Closed-World Framing (`CLOSED_WORLD_FRAMING`)
    Static framing string appended to the context block to discourage the
    model from filling gaps with assumptions.

Layer 3 — Post-generation Response Verification (`verify_response`)
    After the LLM generates a response, scan it for entity mentions and
    cross-reference each against the store. If a mention is unsupported,
    return a `Verification` with `corrective_prompt` set; the caller is
    expected to issue one regeneration request and return the corrected
    text. Max one correction per response.

All three layers are gated by ablation flags in `AblationConfig`:
    absence_signal, closed_world, response_verification.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger("recall.verification")


# ── Constants ────────────────────────────────────────────────────────────────

CLOSED_WORLD_FRAMING = (
    "\n[The above is the complete known context about this user. "
    "If something is not mentioned above, it is not known. "
    "Do not invent details or assume facts that are not stated.]"
)

# Relationship words that the user might be asked about. If the query mentions
# one and the user does not have a matching stored relationship, the user is
# very likely getting hallucinated context.
_RELATIONSHIP_WORDS = {
    "daughter", "son", "child", "children", "kid", "kids", "baby", "twins",
    "wife", "husband", "spouse", "partner", "fiance", "fiancee",
    "girlfriend", "boyfriend",
    "mother", "mom", "mum", "father", "dad", "parent", "parents",
    "brother", "sister", "sibling", "siblings",
    "grandmother", "grandfather", "grandparent", "grandparents",
    "uncle", "aunt", "cousin", "niece", "nephew",
    "dog", "cat", "pet", "puppy", "kitten",
}

# Words that look like proper nouns but should not be flagged as entity
# references if missing from store — cities, common nouns, etc.
_PROPER_NOUN_NOISE = frozenset({
    "Mary", "User",  # the user themselves
    # Months / days are filtered by writer.extract_proper_nouns already, but
    # safety net here as well.
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Question wrappers / sentence starters
    "Tell", "What", "Why", "How", "When", "Where", "Who", "Does", "Did",
    "Has", "Have", "Is", "Are", "Will", "Can", "Could", "Would", "Should",
    "The", "This", "That", "These", "Those", "Here", "There",
    "Hello", "Hi", "Hey", "Yes", "No", "Ok", "Okay", "Sure", "Thanks", "Thank",
    "However", "Therefore", "Although", "But", "And", "Or", "So", "Then",
    "Note", "Based", "According", "Mary's",
    # Pronouns (capitalised when sentence-initial)
    "My", "Our", "Your", "His", "Her", "Their", "I", "We", "You", "They",
    "She", "He", "It", "Me", "Us", "Him", "Them",
    # Common UK cities (Albert-runner context — Bristol/London appear in
    # every query; they are never meaningful as "not in records" signals)
    "Bristol", "London", "Cardiff", "Bath", "Manchester", "Birmingham",
    "Edinburgh", "Glasgow", "Liverpool", "Leeds", "Oxford", "Cambridge",
    "UK", "England", "Scotland", "Wales",
})

_PROPER_NOUN_PATTERN = re.compile(r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)\b")


# ── Layer 1: Absence signalling ──────────────────────────────────────────────


@dataclass
class AbsenceSignal:
    """One injected negative context line."""
    text: str
    reason: str  # "relationship_word" | "proper_noun" — for telemetry


def _extract_relationship_words(query: str) -> set[str]:
    """Return relationship words mentioned in the query."""
    found = set()
    lower = query.lower()
    for w in _RELATIONSHIP_WORDS:
        # Use word boundaries to avoid partial matches
        if re.search(rf"\b{re.escape(w)}\b", lower):
            found.add(w)
    return found


def _extract_query_proper_nouns(query: str) -> list[str]:
    """Extract proper nouns from a query, ignoring sentence-initial wrappers."""
    out: list[str] = []
    for m in _PROPER_NOUN_PATTERN.finditer(query):
        word = m.group(1)
        head = word.split()[0]
        if head in _PROPER_NOUN_NOISE:
            continue
        out.append(word)
    # dedupe while preserving order
    return list(dict.fromkeys(out))


def _user_relationships(store: Any) -> dict[str, list[str]]:
    """Return a {relation_word: [target_entity_name, ...]} map for the user.

    Reads the relationships table for relationships sourced from the User entity.
    Uses fetchall() so the cursor isn't reused mid-iteration.
    """
    rel_map: dict[str, list[str]] = {}
    if store is None or store._conn is None:
        return rel_map
    try:
        cur = store._conn.cursor()
        row = cur.execute(
            "SELECT id FROM entities WHERE lower(name) IN ('user','mary','mary chen') LIMIT 1"
        ).fetchone()
        if not row:
            return rel_map
        user_id = row[0]
        rel_rows = cur.execute(
            "SELECT relationship, target_entity FROM relationships "
            "WHERE source_entity = ? AND status='current'",
            (user_id,),
        ).fetchall()
        for rel_row in rel_rows:
            rel = (rel_row[0] or "").lower()
            target_id = rel_row[1]
            tgt_row = cur.execute(
                "SELECT name FROM entities WHERE id = ?", (target_id,)
            ).fetchone()
            target_name = tgt_row[0] if tgt_row else "?"
            rel_map.setdefault(rel, []).append(target_name)
    except Exception as exc:
        logger.warning("user_relationships lookup failed: %s", exc)
    return rel_map


def _has_relationship_in_facts(rel_word: str, retrieved_facts: Iterable[dict]) -> bool:
    """Check if any retrieved fact mentions the relationship word."""
    needle = rel_word.lower()
    for f in retrieved_facts:
        if needle in (f.get("content", "") or "").lower():
            return True
    return False


# Map common query relationship words to canonical relationship names stored
# in the relationships table.
_REL_CANONICAL = {
    "daughter": "daughter", "son": "son",
    "child": "child", "children": "child", "kid": "child", "kids": "child",
    "baby": "child", "twins": "child",
    "wife": "wife", "husband": "husband", "spouse": "spouse", "partner": "partner",
    "girlfriend": "partner", "boyfriend": "partner",
    "mother": "mother", "mom": "mother", "mum": "mother",
    "father": "father", "dad": "father", "parent": "parent", "parents": "parent",
    "brother": "brother", "sister": "sister", "sibling": "sibling", "siblings": "sibling",
    "dog": "dog", "cat": "cat", "pet": "pet", "puppy": "dog", "kitten": "cat",
}

# Cycle 30 Fix 4: canonical category buckets used for store coverage scoring.
# Absence signals only fire when the store has enough evidence to be
# authoritative about a category — a Day-2 store with 3 facts should not
# be telling the model "the user has no daughter" just because no
# daughter edge has been written yet.
_FAMILY_RELATIONS = frozenset({
    "daughter", "son", "child", "children", "kid", "kids", "baby", "twin",
    "twins", "wife", "husband", "spouse", "partner", "fiance", "fiancee",
    "girlfriend", "boyfriend", "mother", "mom", "mum", "father", "dad",
    "parent", "parents", "brother", "sister", "sibling", "siblings",
    "grandmother", "grandfather", "grandparent", "grandparents",
    "uncle", "aunt", "cousin", "niece", "nephew",
})
_PET_RELATIONS = frozenset({"dog", "cat", "pet", "puppy", "kitten"})

# Minimum coverage score before an absence signal is allowed to fire.
# Calibrated so Days 1–5 (≤30 facts) stay silent, Day 10+ (≥65 facts
# with 3+ family entities) speak up. See plan Fix 4.
_ABSENCE_COVERAGE_GATE = 0.5


def _relation_category(word: str) -> str:
    """Return the canonical category bucket for a relationship word."""
    canon = _REL_CANONICAL.get(word, word)
    if word in _PET_RELATIONS or canon in _PET_RELATIONS:
        return "pet"
    if word in _FAMILY_RELATIONS or canon in _FAMILY_RELATIONS:
        return "family"
    return "other"


def _store_coverage_score(store: Any, category: str) -> float:
    """Compute a coverage confidence score in [0, 1] for *category*.

    Formula (per plan Fix 4):
        coverage = min(facts / 100, 1.0) * min(category_entities / 3, 1.0)

    "category_entities" counts entities related to the User via an edge
    whose canonical relationship falls in the category bucket. This
    treats "the store knows about this user's family" and "the store
    has lots of facts" as two independent axes of confidence — a store
    with 200 facts but zero family edges is not authoritative about
    whether the user has a daughter, and a store with 2 facts but one
    known daughter is not authoritative either.

    Returns 0.0 on any error / missing store so callers stay silent
    when confidence cannot be established.
    """
    if store is None or getattr(store, "_conn", None) is None:
        return 0.0
    try:
        cur = store._conn.cursor()
        facts_count_row = cur.execute(
            "SELECT count(*) FROM facts WHERE status IN ('current','provisional')"
        ).fetchone()
        facts_count = int(facts_count_row[0]) if facts_count_row else 0

        if category == "family":
            rel_set = _FAMILY_RELATIONS
        elif category == "pet":
            rel_set = _PET_RELATIONS
        else:
            # Unknown category → fall back to a pure-facts confidence.
            return min(facts_count / 100.0, 1.0)

        # Count distinct target entities linked to the user via a
        # relationship inside the bucket.
        user_row = cur.execute(
            "SELECT id FROM entities "
            "WHERE lower(name) IN ('user','mary','mary chen') LIMIT 1"
        ).fetchone()
        cat_entities = 0
        if user_row:
            user_id = user_row[0]
            rows = cur.execute(
                "SELECT DISTINCT target_entity, relationship "
                "FROM relationships WHERE source_entity = ? AND status='current'",
                (user_id,),
            ).fetchall()
            for _target, rel in rows:
                rel_norm = (rel or "").lower()
                if rel_norm in rel_set or _REL_CANONICAL.get(rel_norm, rel_norm) in rel_set:
                    cat_entities += 1

        facts_score = min(facts_count / 100.0, 1.0)
        cat_score = min(cat_entities / 3.0, 1.0)
        return facts_score * cat_score
    except Exception as exc:
        logger.warning("store_coverage_score(%s) failed: %s", category, exc)
        return 0.0


_POSSESSIVE_ASSERTION_RX = re.compile(
    r"\b(?:my|our)\s+(\w+)",
    re.IGNORECASE,
)

# Sentences that start with interrogative words are questions, not
# assertions. "What was the name of my cat?" does NOT assert the user
# has a cat — it asks. Split the query on sentence boundaries and skip
# any sentence that opens with a wh-word or auxiliary.
_INTERROGATIVE_OPENERS = re.compile(
    r"^\s*(?:what|who|whom|whose|when|where|why|how|which|"
    r"is|are|was|were|do|does|did|can|could|would|should|"
    r"will|may|might|has|have|had)\b",
    re.IGNORECASE,
)


def _assertion_terms_in_query(query: str) -> set[str]:
    """Relation / entity terms the user asserts ownership of in the query.

    "I need to pick up my daughter from school" yields {"daughter"}.
    "My wife Sophie's birthday" yields {"wife"}.
    "What was the name of my cat?" yields an empty set — interrogative.

    Only first-person possessives ("my", "our") count as assertions.
    Second-person ("your") and third-person ("Albert's", "the user's")
    are how the model asks about the user, not how the user claims
    ownership, so they never trigger suppression.
    """
    if not query:
        return set()
    out: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", query):
        s = sentence.strip()
        if not s:
            continue
        if _INTERROGATIVE_OPENERS.match(s):
            continue
        for m in _POSSESSIVE_ASSERTION_RX.finditer(s):
            out.add(m.group(1).lower())
    return out


# Proper nouns introduced by a possessive phrase in the current message,
# e.g. "my wife Sophie", "my son Oscar", "our dog Biscuit". When this
# pattern appears in an assertion (non-interrogative sentence), the
# capitalised name is being introduced by the user and should not
# trigger a negative signal — they are telling us about a new entity.
#
# The `my|our` prefix uses IGNORECASE so sentence-initial "My" matches,
# but we explicitly require the introduced noun to start with an
# uppercase letter to avoid matching the relation word itself ("my wife"
# would otherwise yield "wife"). Python's re flags apply globally, so
# the uppercase constraint is enforced via an explicit character-class
# rather than case. To keep that constraint active, we compile without
# re.IGNORECASE and instead do the case-insensitive `my|our` match by
# covering both capitalisations explicitly.
_INTRODUCED_NOUN_RX = re.compile(
    r"\b(?:[Mm]y|[Oo]ur)\s+(?:\w+\s+)?([A-Z][a-zA-Z]+)",
)


def _introduced_proper_nouns(query: str) -> set[str]:
    """Return the lowercased proper nouns introduced by a possessive
    assertion in a non-interrogative sentence."""
    if not query:
        return set()
    out: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", query):
        s = sentence.strip()
        if not s or _INTERROGATIVE_OPENERS.match(s):
            continue
        for m in _INTRODUCED_NOUN_RX.finditer(s):
            out.add(m.group(1).lower())
    return out


def _mentions_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    return bool(re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))


def _recent_turn_contents(recent_turns: list[dict] | None) -> list[str]:
    """Extract stringified content from the last-N turn dicts.

    Accepts the shape used throughout the proxy: {"role": "...",
    "content": "..."}. Non-string content is JSON-serialised so we can
    still grep through it without exploding on tool-call dicts.
    """
    if not recent_turns:
        return []
    out: list[str] = []
    for msg in recent_turns:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(content)
        elif content is not None:
            import json as _json
            try:
                out.append(_json.dumps(content))
            except Exception:
                pass
    return out


def build_absence_signals(
    query: str,
    retrieved_facts: list[dict],
    store: Any,
    recent_turns: list[dict] | None = None,
) -> list[AbsenceSignal]:
    """Cycle 19 Layer 1 (v3 — Q64 widening): inject absence signals only
    when the query references a relation / proper noun that has NO
    supporting evidence across ANY available context surface.

    Evidence surfaces consulted, in order:

      1. The relationships graph (canonical edges rooted at User).
      2. Content of every retrieved fact.
      3. Content of the last-N conversation turns (the caller passes
         ``recent_turns`` — typically the last 3 user/assistant turns
         already bound for the lean payload).
      4. A possessive-assertion scan of the query itself (``my daughter``,
         ``our wife``, ``Albert's dog``). When the user asserts they
         have X, we must never tell the model X doesn't exist.

    If any of (1)-(4) mentions the term, the signal is suppressed. A
    trap query like "What was the name of Albert's cat again?" still
    fires because the cat word is not asserted ("Albert's cat" is a
    question, not a claim), and no fact / turn / edge backs it.
    Conversely "I need to pick up my daughter from school" has
    "daughter" in the possessive-assertion set and never fires a
    false negative.
    """
    signals: list[AbsenceSignal] = []

    asserted_terms = _assertion_terms_in_query(query)
    turn_texts = _recent_turn_contents(recent_turns)

    def _covered_by_assertion(word: str) -> bool:
        """Handle both the literal word and canonical siblings ("kids"
        covered by "daughter"/"son", "parent" covered by "mother"/"father")."""
        if word in asserted_terms:
            return True
        canonical = _REL_CANONICAL.get(word, word)
        if canonical in asserted_terms:
            return True
        # Group-wise siblings: if the user asserts any specific child,
        # the generic "kids/children" query should not fire a negative.
        child_set = {"son", "daughter", "child", "children", "kid", "kids",
                     "baby", "twin", "twins"}
        if (word in child_set or canonical in child_set) and (
            asserted_terms & child_set
        ):
            return True
        parent_set = {"mother", "mom", "mum", "father", "dad", "parent",
                      "parents"}
        if (word in parent_set or canonical in parent_set) and (
            asserted_terms & parent_set
        ):
            return True
        pet_set = {"dog", "cat", "pet", "puppy", "kitten"}
        if (word in pet_set or canonical in pet_set) and (
            asserted_terms & pet_set
        ):
            return True
        partner_set = {"wife", "husband", "spouse", "partner"}
        if (word in partner_set or canonical in partner_set) and (
            asserted_terms & partner_set
        ):
            return True
        return False

    # (a) Relationship-word check
    # Cycle 30 Fix 4: gated on per-category store coverage. We never fire
    # "the user has no daughter" when the store is still too sparse to be
    # authoritative — below _ABSENCE_COVERAGE_GATE we stay silent and let
    # the model answer from context rather than risk a false negative.
    user_rels = _user_relationships(store)
    _coverage_cache: dict[str, float] = {}

    def _coverage(category: str) -> float:
        if category not in _coverage_cache:
            _coverage_cache[category] = _store_coverage_score(store, category)
        return _coverage_cache[category]

    for word in _extract_relationship_words(query):
        canonical = _REL_CANONICAL.get(word, word)
        in_store = canonical in user_rels or any(
            canonical in k for k in user_rels.keys()
        )
        in_facts = _has_relationship_in_facts(word, retrieved_facts)
        in_turns = any(_mentions_term(t, word) or _mentions_term(t, canonical)
                       for t in turn_texts)
        if _covered_by_assertion(word):
            continue
        if in_store or in_facts or in_turns:
            continue
        category = _relation_category(word)
        coverage = _coverage(category)
        if coverage <= _ABSENCE_COVERAGE_GATE:
            logger.info(
                "ABL-AS: suppressed absence signal for %r "
                "(category=%s coverage=%.2f ≤ gate=%.2f)",
                word, category, coverage, _ABSENCE_COVERAGE_GATE,
            )
            continue
        # Build a richer signal that also lists the actual related entities,
        # so the model has positive grounding instead of just a negative.
        existing = ", ".join(
            f"{rel} {' or '.join(names)}"
            for rel, names in sorted(user_rels.items())
            if rel in {"son", "daughter", "child", "wife", "husband", "spouse",
                       "partner", "mother", "father", "brother", "sister"}
        )
        text = f"FACT: the user has no {word}."
        if existing:
            text += f" The user's actual family members on record are: {existing}."
        signals.append(AbsenceSignal(text=text, reason="relationship_word"))

    # (b) Proper-noun check
    #
    # Suppress when the noun is introduced by the user in the CURRENT
    # message alongside a possessive assertion ("my wife Sophie"). If
    # the user is telling us about a new entity this turn, firing a
    # negative signal against it is a false refusal.
    #
    # Cycle 30 Fix 4: also gated on an unconditional facts-density
    # confidence (no category context here — the noun could be anything),
    # so a sparse store stays silent rather than emitting "'Whiskers' is
    # not in the user's records" and risking a fabricated rewrite.
    introduced_nouns = _introduced_proper_nouns(query)
    facts_only_coverage = _coverage("other")
    if facts_only_coverage <= _ABSENCE_COVERAGE_GATE:
        logger.info(
            "ABL-AS: suppressed proper-noun absence signals "
            "(facts coverage=%.2f ≤ gate=%.2f)",
            facts_only_coverage, _ABSENCE_COVERAGE_GATE,
        )
    elif store is not None and store._conn is not None:
        try:
            cur = store._conn.cursor()
            for noun in _extract_query_proper_nouns(query):
                if noun.lower() in introduced_nouns:
                    continue
                row = cur.execute(
                    "SELECT 1 FROM entities WHERE lower(name) = lower(?) LIMIT 1",
                    (noun,),
                ).fetchone()
                if row:
                    continue
                if any(noun.lower() in (f.get("content", "") or "").lower()
                       for f in retrieved_facts):
                    continue
                if any(_mentions_term(t, noun) for t in turn_texts):
                    continue
                signals.append(AbsenceSignal(
                    text=f"FACT: '{noun}' is not in the user's records.",
                    reason="proper_noun",
                ))
        except Exception as exc:
            logger.warning("proper_noun absence check failed: %s", exc)

    return signals


# ── Layer 3: Response verification ──────────────────────────────────────────


@dataclass
class FlaggedClaim:
    """A response claim that contradicts a stored fact."""
    subject: str          # entity the claim is about (e.g. "Tom", "Mary")
    predicate: str        # normalised predicate key (e.g. "job_state")
    claimed: str          # what the response said the object was
    stored: str           # what the store actually has
    sentence: str         # the full sentence the claim came from


@dataclass
class Verification:
    """Result of post-generation verification."""
    is_clean: bool                       # True if no hallucinations detected
    flagged_entities: list[str] = field(default_factory=list)
    flagged_relations: list[str] = field(default_factory=list)
    flagged_claims: list[FlaggedClaim] = field(default_factory=list)
    corrective_prompt: str | None = None  # set if regeneration is needed
    reason: str = ""


def _extract_response_proper_nouns(response: str) -> list[str]:
    """Extract proper nouns from freeform response text.

    v3 (cycle 20): markdown-aware — skips captures that appear inside
    bullet-list headers or **emphasis** markers, since those are almost
    never real named entities (they're section labels chosen by the
    model). The prose-flow check runs first; noise-word and first-word
    filtering still apply.
    """
    if not response:
        return []
    out: list[str] = []
    text = response
    lines = text.splitlines()
    # Build a set of character offsets that are inside markdown emphasis
    # (**X**, __X__) or are bullet-list headers (first capitalised word
    # right after a list marker, particularly if followed by ':' or end).
    skip_spans: list[tuple[int, int]] = []
    offset = 0
    for line in lines:
        lstripped = line.lstrip()
        leading = len(line) - len(lstripped)
        # Detect list-item lines: "- ...", "* ...", "•", numeric "1. "
        is_list_item = bool(re.match(r"^(?:[-*•]|\d+[.)])\s", lstripped))
        # Skip any markdown **emphasis** content
        for em in re.finditer(r"\*\*([^*]+)\*\*|__([^_]+)__", line):
            skip_spans.append((offset + em.start(), offset + em.end()))
        # For bullet-list lines, skip proper-noun extraction on the WHOLE
        # line. Bullet bodies are almost always labels or suggestion lists,
        # never real structural claims by the model. Real hallucinations
        # show up in the prose lead-in, not in the bullets.
        if is_list_item:
            skip_spans.append((offset, offset + len(line)))
        offset += len(line) + 1  # +1 for newline

    def _in_skip(pos: int) -> bool:
        return any(lo <= pos < hi for lo, hi in skip_spans)

    # A capture is only flagged if it looks like it is attached to a verb —
    # i.e. "studied X" / "at X" / "is X" / "X said" — the kind of structural
    # claim the LLM makes in prose. List-body "X" fragments inside bullets
    # are typically just labels.
    _ATTACHED_CONTEXT = re.compile(
        r"\b(?:is|was|are|were|studied|attended|graduated|went|lives?|lived|"
        r"works?|worked|from|at|in|of|for|to|with|met|named|called|"
        r"joined|left|founded|married|dated)\s*$",
        re.IGNORECASE,
    )
    for m in _PROPER_NOUN_PATTERN.finditer(text):
        word = m.group(1)
        head = word.split()[0]
        if head in _PROPER_NOUN_NOISE:
            continue
        start = m.start()
        if _in_skip(start):
            continue
        is_multi_word = " " in word
        if not is_multi_word and start <= 5:
            preceding = text[:start].strip()
            if not preceding:
                continue
        # Require a verb/preposition right before the capture. This filters
        # bullet-body list fragments that have no grammatical attachment.
        lookback = text[max(0, start - 25):start]
        if not _ATTACHED_CONTEXT.search(lookback):
            continue
        out.append(word)
    return list(dict.fromkeys(out))


def _is_known_entity(name: str, store: Any) -> bool:
    if store is None or store._conn is None:
        return True  # fail-open: don't flag if store is unavailable
    try:
        cur = store._conn.cursor()
        # Check exact name match (case-insensitive)
        row = cur.execute(
            "SELECT 1 FROM entities WHERE lower(name) = lower(?) LIMIT 1",
            (name,),
        ).fetchone()
        if row:
            return True
        # Also check if the name appears in any fact content
        row = cur.execute(
            "SELECT 1 FROM facts WHERE lower(content) LIKE lower(?) LIMIT 1",
            (f"%{name}%",),
        ).fetchone()
        return row is not None
    except Exception as exc:
        logger.warning("entity lookup failed for %s: %s", name, exc)
        return True


# ── Claim-vs-store contradiction detection (v3) ─────────────────────────────
#
# Primary detector for Layer 3 v3. Extracts (subject, predicate_key, object)
# triples from both the LLM's response sentences and from stored fact rows,
# and flags a contradiction when a response claim binds the same predicate
# key to a different object than the store does for the same subject.
#
# The predicate templates here are intentionally more permissive than the
# writer's (writer.py `_VALUE_PREDICATES`) because LLM freeform responses
# use language that the writer's strict templates reject — e.g. "Tom is
# still working as a lawyer" or "User's spouse Tom is a high school history
# teacher". This module needs to normalise both sides to the same shape.

_JOB_ROLES = (
    "engineer", "manager", "developer", "designer", "analyst", "director",
    "officer", "lead", "architect", "consultant", "teacher", "professor",
    "nurse", "doctor", "lawyer", "attorney", "founder", "ceo", "cto", "cfo",
    "coo", "vp", "pm", "product manager", "product owner", "scientist",
    "researcher", "accountant", "banker", "chef", "artist",
)
_JOB_ROLES_RX = "|".join(re.escape(r) for r in _JOB_ROLES)

# Response-side claim extractors. Each captures (predicate_key, object).
# Only one capture group — the object — per pattern.
_V3_RESPONSE_CLAIMS: list[tuple[re.Pattern, str]] = [
    # "is/was a history teacher", "is currently a product manager",
    # "is still working as a lawyer", "works as a lawyer",
    # "is currently the CTO", "was previously a senior PM"
    (re.compile(
        rf"\b(?:is|was|are|were)\b\s+"
        rf"(?:(?:still|currently|now|previously|formerly|a\s+former|the)\s+)*"
        rf"(?:working\s+)?(?:as\s+)?(?:a\s+|an\s+|the\s+)?"
        rf"((?:\w+\s+){{0,2}}(?:{_JOB_ROLES_RX}))\b",
        re.IGNORECASE,
    ), "job_role"),
    # "works at X", "works for X" (capture up to a comma/period)
    (re.compile(r"\bworks?\s+(?:at|for)\s+([A-Z][\w& ]+?)(?:[\.,;]|$)"), "job_employer"),
    # "lives in X"
    (re.compile(r"\blives?\s+in\s+([A-Z][\w ]+?)(?:[\.,;]|$)"), "residence"),
    # "is N years old"
    (re.compile(r"\bis\s+(\d+)\s+years?\s+old\b", re.IGNORECASE), "age"),
    # "is married/separated/divorced/engaged/single/widowed"
    (re.compile(r"\bis\s+(married|separated|divorced|engaged|single|widowed)\b", re.IGNORECASE), "marital_state"),
    # "has a daughter/son/dog/cat named X"
    (re.compile(r"\bhas\s+(?:a|an)\s+(?:daughter|son|dog|cat|pet)\s+named\s+([A-Z][a-zA-Z]+)\b"), "named_child_pet"),
]

# Store-side fact extractors. Handles "User's spouse Tom is a ..." style
# fact formats that the writer's templates don't capture for cross-entity
# attributes. Output shape: (subject_hint, predicate_key, object).
_V3_STORE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # "User's role is VP of Product" / "Tom's job is teacher" — possessive form
    (re.compile(
        rf"(?P<subj>user|mary|the user|[A-Z][a-zA-Z]+)'s\s+"
        rf"(?:current\s+|new\s+)?(?:role|job|position|occupation|title)\s+is\s+"
        rf"(?:a\s+|an\s+|the\s+)?(?P<obj>(?:\w+\s+){{0,3}}(?:{_JOB_ROLES_RX}))\b",
        re.IGNORECASE,
    ), "job_role"),
    # "<anything> Tom is a history teacher" — capture role tail.
    (re.compile(
        rf"(?P<subj>[A-Z][a-zA-Z]+)\s+is\s+(?:a\s+|an\s+)?"
        rf"(?:\w+\s+){{0,3}}(?P<obj>(?:\w+\s+)*(?:{_JOB_ROLES_RX}))\b",
        re.IGNORECASE,
    ), "job_role"),
    # "<subj> works at X" / "works for X"
    (re.compile(
        r"(?P<subj>[A-Z][a-zA-Z]+)\s+works?\s+(?:at|for)\s+(?P<obj>[\w& ]+?)(?:[\.,;]|$)",
        re.IGNORECASE,
    ), "job_employer"),
    # "User lives in X" — present tense only, past tense does not set a
    # current residence and therefore cannot drive a contradiction.
    (re.compile(
        r"(?P<subj>user|mary|the user)\s+lives?\s+in\s+(?P<obj>[\w ]+?)(?:[\.,;]|$)",
        re.IGNORECASE,
    ), "residence"),
    # "User/Mary is married/separated/..." and "User and Tom are separated"
    (re.compile(
        r"(?P<subj>user|mary|the user)\s+(?:and\s+\w+\s+)?(?:is|are|have been|has been)\s+"
        r"(?P<obj>married|engaged|separated|divorced|single|widowed)\b",
        re.IGNORECASE,
    ), "marital_state"),
    # "User is N years old"
    (re.compile(
        r"(?P<subj>user|mary|the user)\s+is\s+(?P<obj>\d+)\s+years?\s+old\b",
        re.IGNORECASE,
    ), "age"),
]


def _split_sentences(text: str) -> list[str]:
    """Split response text into sentences."""
    parts: list[str] = []
    for seg in re.split(r"(?<=[.!?])\s+|\n+", text):
        seg = seg.strip(" \t*-•")
        if seg:
            parts.append(seg)
    return parts


_SUBJECT_RX = re.compile(
    r"^(?:Yes[,\s]+|No[,\s]+|Actually[,\s]+|Well[,\s]+)?"
    r"(?P<subj>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?|The user|She|He)\b"
    r"(?P<possessive>'s\s+\w+)?"
)


def _sentence_subject(sentence: str) -> str | None:
    """Best-effort guess at the subject of a sentence.
    Returns a capitalised name, or None. If the sentence uses a possessive
    form ('Mary's parents live in X'), return None — the grammatical subject
    is the possessed noun, not the named entity, and treating it as a claim
    about the entity causes false positives on family-relation lookups.
    """
    m = _SUBJECT_RX.match(sentence.strip())
    if not m:
        return None
    if m.group("possessive"):
        return None  # possessive construct — let the claim pass
    subj = m.group("subj").strip()
    if subj.lower() in {"she", "he", "the user"}:
        return "the_user"
    return subj


def _normalise_role(role: str) -> str:
    """Collapse a role phrase to its canonical form.
    'high school history teacher' -> 'teacher', 'VP of Product' -> 'vp',
    'Senior PM' -> 'pm'. Picks the last job-role token in the phrase.
    """
    r = role.lower().strip()
    # Find the last matching job role token
    last = None
    for token in _JOB_ROLES:
        if re.search(rf"\b{re.escape(token)}\b", r):
            last = token
    return last or r


def _extract_response_claims(sentence: str) -> list[tuple[str, str, str]]:
    """Extract (subject, predicate_key, object) triples from a response
    sentence. Returns a list (a sentence can contain multiple claims).
    """
    triples: list[tuple[str, str, str]] = []
    subj = _sentence_subject(sentence)
    if subj is None:
        return triples
    body = sentence
    # Strip appositive clauses ("Mary, as the CTO at Nexus, is not...") —
    # claims inside them are usually framing, not assertions, and a
    # negation on the main verb can be missed by the sentence-level check.
    body = re.sub(r",\s*as\s+(?:the\s+|a\s+|an\s+)?[^,]+,", ",", body)
    for rx, key in _V3_RESPONSE_CLAIMS:
        for m in rx.finditer(body):
            if key == "job_role":
                role = _normalise_role(m.group(1))
                if role:
                    triples.append((subj, "job_role", role))
            elif key == "job_employer":
                emp = m.group(1).strip().rstrip(".,;")
                triples.append((subj, "job_employer", emp.lower()))
            elif key == "residence":
                triples.append((subj, "residence", m.group(1).strip().lower()))
            elif key == "age":
                triples.append((subj, "age", m.group(1)))
            elif key == "marital_state":
                triples.append((subj, "marital_state", m.group(1).lower()))
            elif key == "named_child_pet":
                triples.append((subj, "named_child_pet", m.group(1)))
    return triples


def _fetch_entity_facts(name: str, store: Any, limit: int = 500) -> list[str]:
    """Return current-status fact contents that mention the named entity."""
    if store is None or store._conn is None:
        return []
    try:
        cur = store._conn.cursor()
        like = f"%{name.lower()}%"
        rows = cur.execute(
            "SELECT content FROM facts WHERE lower(content) LIKE ? "
            "AND status='current' LIMIT ?",
            (like, limit),
        ).fetchall()
        return [r[0] for r in rows if r and r[0]]
    except Exception as exc:
        logger.warning("fetch_entity_facts(%s) failed: %s", name, exc)
        return []


_SPECULATIVE_STORE_MARKERS = re.compile(
    r"\b(?:is\s+considering|considered|might|may\b|could|would|thinking\s+about|"
    r"planning\s+to|hopes?\s+to|wants?\s+to|decided\s+against|no\s+longer\s+wants|"
    r"speculating|speculated)\b",
    re.IGNORECASE,
)


def _extract_store_triples(content: str) -> list[tuple[str, str, str]]:
    """Extract (subject_hint, predicate_key, object) triples from a stored
    fact content. Speculative / hypothetical facts are skipped so that
    'User is considering a senior PM role' does not contradict a later
    'User is VP of Product'.
    """
    out: list[tuple[str, str, str]] = []
    # Skip speculative facts entirely — they should never drive a
    # contradiction fire since they never represent a current ground truth.
    if _SPECULATIVE_STORE_MARKERS.search(content):
        return out

    text = content.strip()
    # Strip common user-possessive prefixes ("User's spouse Tom" -> "Tom")
    stripped = re.sub(
        r"^(?:User's|the user's|Mary's)\s+(?:spouse|husband|wife|partner|friend|"
        r"colleague|boss|manager|mother|father|parent|son|daughter|child|"
        r"brother|sister|best friend|neighbor|neighbour)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    for rx, key in _V3_STORE_PATTERNS:
        for m in rx.finditer(stripped):
            try:
                subj = m.group("subj")
            except IndexError:
                continue
            try:
                obj = m.group("obj")
            except IndexError:
                continue
            if not subj or not obj:
                continue
            subj_norm = "the_user" if subj.lower() in {"theuser", "user", "mary", "the user"} else subj
            if key == "job_role":
                role = _normalise_role(obj)
                if role:
                    out.append((subj_norm, "job_role", role))
            elif key == "job_employer":
                out.append((subj_norm, "job_employer", obj.strip().rstrip(".,;").lower()))
            elif key == "residence":
                out.append((subj_norm, "residence", obj.strip().lower()))
            elif key == "marital_state":
                out.append((subj_norm, "marital_state", obj.strip().lower()))
            elif key == "age":
                out.append((subj_norm, "age", obj.strip()))
    return out


def _subjects_equivalent(a: str, b: str) -> bool:
    """Whether two subject hints refer to the same entity."""
    a_l = a.lower().strip()
    b_l = b.lower().strip()
    if a_l == b_l:
        return True
    user_aliases = {"the_user", "theuser", "user", "mary", "mary chen", "the user"}
    if a_l in user_aliases and b_l in user_aliases:
        return True
    if not a_l or not b_l:
        return False
    # Allow first-name match (Tom ≡ Tom), but only for non-user tokens
    if a_l in user_aliases or b_l in user_aliases:
        return False
    return a_l.split()[0] == b_l.split()[0]


def _detect_claim_contradictions(
    response_text: str,
    store: Any,
) -> list[FlaggedClaim]:
    """Primary v3 detector: scan the response for predicate claims about
    known entities that contradict stored facts.
    """
    claims: list[FlaggedClaim] = []
    if store is None or store._conn is None:
        return claims

    # Build per-subject store triple index lazily (one lookup per subject).
    store_cache: dict[str, list[tuple[str, str, str, str]]] = {}

    def _triples_for_subject(subject: str) -> list[tuple[str, str, str, str]]:
        """Return list of (subject_hint, key, object, source_fact)."""
        key_norm = subject.lower()
        if key_norm in store_cache:
            return store_cache[key_norm]
        lookup_name = "user" if subject == "the_user" else subject
        facts = _fetch_entity_facts(lookup_name, store)
        # Also pull user facts for Mary
        if lookup_name.lower() == "mary":
            facts += _fetch_entity_facts("user", store)
        triples: list[tuple[str, str, str, str]] = []
        for f in facts:
            for (subj_h, k, obj) in _extract_store_triples(f):
                triples.append((subj_h, k, obj, f))
        store_cache[key_norm] = triples
        return triples

    user_aliases = {"mary", "mary chen", "the user", "user", "the_user", "she", "he"}

    def _canon_subj(subj: str) -> str:
        return "the_user" if subj.lower() in user_aliases else subj

    seen: set[tuple[str, str, str]] = set()
    for sentence in _split_sentences(response_text):
        for (subject, key, claimed) in _extract_response_claims(sentence):
            canon = _canon_subj(subject)
            dedup = (canon.lower(), key, claimed)
            if dedup in seen:
                continue
            seen.add(dedup)
            # Skip hedged / speculative claims (valid inference)
            slow = sentence.lower()
            if any(h in slow for h in (
                "might", "could be", "may be", "perhaps", "possibly",
                "would be", "should consider", "could consider",
                "suggest", "recommend", "i don't", "i do not", "no record",
                "no mention", "not mentioned", "doesn't have", "does not have",
                "isn't", "is not", "there is no", "there are no",
            )):
                continue
            # Look up stored triples for this subject (keyed by canonical subj)
            stored_triples = _triples_for_subject(canon)
            matching_same_key = [
                t for t in stored_triples
                if t[1] == key and _subjects_equivalent(t[0], canon)
            ]
            if not matching_same_key:
                continue  # no stored fact on this predicate → inference OK
            # Contradiction iff no stored triple matches the claimed object
            if any(t[2] == claimed for t in matching_same_key):
                continue  # agreement
            stored_fact = matching_same_key[0][3]
            claims.append(FlaggedClaim(
                subject=subject,
                predicate=key,
                claimed=claimed,
                stored=stored_fact,
                sentence=sentence.strip(),
            ))
    return claims


def verify_response(
    query: str,
    response_text: str,
    store: Any,
    retrieved_facts: list[dict] | None = None,
) -> Verification:
    """Cycle 19 Layer 3: scan response text for unsupported entities and claims.

    v2 (post-pass-2): the v1 implementation skipped refusals entirely and
    used overly conservative noun extraction, so it never fired in practice.
    v2 changes:
      - A refusal *prefix* doesn't make the whole response clean — keep
        verifying any concrete claims that follow.
      - Catch relationship-word claims: if the response asserts a specific
        property of a relationship the user doesn't have (e.g. "Mary's
        daughter is named Sarah" when there is no daughter), flag it.
      - Tighten proper-noun extraction: include single-word capitalised
        names mid-sentence; only ignore the very first word of the
        response itself.
      - Cross-check the relationship-word check against the user's
        relationships table — same store lookup as Layer 1.
    """
    if not response_text or len(response_text) < 5:
        return Verification(is_clean=True)

    facts = retrieved_facts or []
    flagged_entities: list[str] = []
    flagged_relations: list[str] = []

    # ── Claim-vs-store contradiction check (v3 primary) ──────────────────────
    # Scan response sentences for predicate claims about known entities that
    # directly contradict stored facts. This catches "Tom is a lawyer" when
    # the store has "Tom is a history teacher".
    flagged_claims = _detect_claim_contradictions(response_text, store)

    # ── Proper noun check ────────────────────────────────────────────────────
    for noun in _extract_response_proper_nouns(response_text):
        if noun.lower() in query.lower():
            continue
        if _is_known_entity(noun, store):
            continue
        if any(noun.lower() in (f.get("content", "") or "").lower() for f in facts):
            continue
        flagged_entities.append(noun)

    # ── Relationship-word check ──────────────────────────────────────────────
    # If the response asserts something concrete about a relationship the
    # user does not have (e.g. "Mary's daughter is named Sarah"), flag it.
    user_rels = _user_relationships(store)
    rl = response_text.lower()
    # Expand the store's relationship-key set so that generic terms
    # ("child", "pet", "parent") are satisfied whenever a concrete form
    # exists. Without this the detector false-positives on multi-hop
    # answers that use generic vocabulary.
    rel_keys_present = set(user_rels.keys())
    parent_markers = {"mother", "father", "mom", "dad", "mum", "parent", "parents"}
    if rel_keys_present & parent_markers:
        rel_keys_present.update({"parent", "parents", "mother", "father"})
    if "brother" in rel_keys_present or "sister" in rel_keys_present:
        rel_keys_present.add("sibling")
    child_markers = {"son", "daughter", "child", "children", "kid", "kids",
                     "twin", "twins", "baby"}
    if rel_keys_present & child_markers:
        rel_keys_present.update({"child", "children", "kid", "kids", "baby",
                                 "twins", "twin"})
    pet_markers = {"dog", "cat", "pet", "puppy", "kitten"}
    if rel_keys_present & pet_markers:
        rel_keys_present.update({"pet", "puppy", "kitten"})
    partner_markers = {"husband", "wife", "spouse", "partner", "fiance",
                       "fiancee", "girlfriend", "boyfriend"}
    if rel_keys_present & partner_markers:
        rel_keys_present.update(partner_markers)
    for word in _extract_relationship_words(response_text):
        canonical = _REL_CANONICAL.get(word, word)
        if canonical in rel_keys_present or any(canonical in k for k in rel_keys_present):
            continue
        # Also accept if any retrieved fact mentions the word
        if any(word in (f.get("content", "") or "").lower() for f in facts):
            continue
        # Skip if the response itself is making a negative claim about the word
        # ("there is no daughter", "no record of a daughter", "mary doesn't have a daughter")
        negation_markers = (
            f"no {word}",
            f"no record of a {word}", f"no record of any {word}",
            f"doesn't have a {word}", f"does not have a {word}",
            f"does not have any {word}", f"doesn't have any {word}",
            f"there is no {word}", f"there are no {word}",
            f"no {word} in", f"no {word} on record",
        )
        if any(m in rl for m in negation_markers):
            continue
        flagged_relations.append(word)

    if not flagged_entities and not flagged_relations and not flagged_claims:
        return Verification(is_clean=True)

    # Build a targeted corrective prompt. Design goals (cycle21):
    #   - Short and imperative — long prompts push qwen3.5:9b into refusal.
    #   - Name the SPECIFIC contradicted field, not a category.
    #   - Explicitly instruct REWRITE, never refuse or disclaim.
    #   - Preserve every correct piece of the previous answer.
    reason = ""
    fixes: list[str] = []

    if flagged_claims:
        reason = "claim_contradiction"
        for c in flagged_claims:
            # Try to extract a human-readable "right answer" from the stored fact
            right = c.stored.strip().rstrip(".")
            fixes.append(
                f'You wrote "{c.claimed}" for {c.subject}. '
                f"The correct value is in this record: \"{right}\". "
                f"Replace the wrong value with the correct one."
            )

    if flagged_entities:
        if not reason:
            reason = "unsupported_entities"
        names = ", ".join(flagged_entities)
        fixes.append(
            f"Remove any mention of: {names}. "
            "These names are not in the user's records. "
            "Keep everything else from your previous answer."
        )

    if flagged_relations:
        if not reason:
            reason = "unsupported_relations"
        rels = ", ".join(flagged_relations)
        fixes.append(
            f"The user does not have a {rels} on record. "
            "Remove any invented attributes you claimed about them, "
            "but keep every other factual piece from your previous answer."
        )

    corrective = (
        "Rewrite your previous answer with these specific fixes:\n\n"
        + "\n\n".join(f"- {f}" for f in fixes)
        + "\n\nImportant: Do NOT refuse or say you lack information. "
        "Reuse every correct sentence from your previous answer verbatim. "
        "Only change the specific wrong parts named above. Produce a "
        "complete, usable answer to the user's original question."
    )

    return Verification(
        is_clean=False,
        flagged_entities=flagged_entities,
        flagged_relations=flagged_relations,
        flagged_claims=flagged_claims,
        corrective_prompt=corrective,
        reason=reason,
    )


# ── Response body helpers ────────────────────────────────────────────────────


def extract_response_text(body: bytes, api_format: str) -> str | None:
    """Extract the assistant message text from a non-streaming response body.

    Returns None if the body cannot be parsed.
    """
    import json as _json
    try:
        data = _json.loads(body)
    except Exception:
        return None
    if api_format == "ollama":
        msg = data.get("message") or {}
        return msg.get("content")
    if api_format == "openai":
        choices = data.get("choices") or []
        if not choices:
            return None
        return (choices[0].get("message") or {}).get("content")
    return None


def replace_response_text(body: bytes, api_format: str, new_text: str) -> bytes:
    """Return a new body bytes with the assistant text replaced."""
    import json as _json
    try:
        data = _json.loads(body)
    except Exception:
        return body
    if api_format == "ollama":
        if "message" not in data or not isinstance(data["message"], dict):
            data["message"] = {}
        data["message"]["content"] = new_text
    elif api_format == "openai":
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].setdefault("message", {})
            msg["content"] = new_text
    return _json.dumps(data).encode("utf-8")


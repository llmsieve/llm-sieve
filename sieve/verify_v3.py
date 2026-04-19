"""Layer 3 v3 — verify-facts as tool result.

Implements the 2026-04-13 design spec: instead of telling the LLM it was
wrong and asking it to regenerate, we splice a synthetic `verify_facts`
tool exchange into the assistant turn and prefill the model to continue
from there. The model sees its own previous text, its "own" tool call,
and a tool result carrying the authoritative data. Its continuation
reconciles with the new facts the same way it handles any authoritative
data mid-turn.

Design summary (five decisions locked during brainstorming):
  Q1 — Automatic interception (not proactive tool)
  Q2 — Pattern + store-entity match (two extractors, no LLM)
  Q3 — Asymmetric two-field tool result (user_facts + not_in_records)
  Q4 — Continuation via assistant-prefill, empty-args call, ~200 tokens
  Q5 — Sentence-level splice (keep good, drop flagged, append continuation)

This module reuses the claim-vs-store detector from
src/verification.py (which already had TP=5, FP=0 on the targeted
dataset). The v3 contribution is sentence-index tracking, the fabricated-
relationship extractor, the asymmetric tool result, and the splice +
prefill flow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sieve.verification import (
    FlaggedClaim,
    _detect_claim_contradictions,
    _extract_response_claims,
    _fetch_entity_facts,
    _is_known_entity,
    _split_sentences,
    _user_relationships,
)

logger = logging.getLogger("recall.verify_v3")


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class V3FlaggedAttribute:
    """An attribute contradiction — goes into `user_facts` in the tool result.

    Wraps a FlaggedClaim with the source sentence index so we can
    splice it out of the original response.
    """
    subject: str
    predicate: str
    wrong_value: str
    correct_value: str
    sentence_index: int
    sentence: str


@dataclass
class V3FabricatedRelationship:
    """An unknown proper noun bound to a known entity via a relationship
    predicate — goes into `not_in_records` in the tool result.
    """
    anchor: str              # known entity the fabrication is attached to
    relationship: str        # e.g. "daughter", "nanny", "cousin"
    unknown_entity: str | None  # the fabricated name, if named
    sentence_index: int
    sentence: str


@dataclass
class V3Verification:
    """Result of the v3 verification pass."""
    is_clean: bool
    attributes: list[V3FlaggedAttribute] = field(default_factory=list)
    fabricated: list[V3FabricatedRelationship] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)

    @property
    def flagged_sentence_indices(self) -> set[int]:
        idx: set[int] = set()
        for a in self.attributes:
            idx.add(a.sentence_index)
        for f in self.fabricated:
            idx.add(f.sentence_index)
        return idx


# ── Attribute extractor ──────────────────────────────────────────────────────


def extract_attribute_contradictions(
    response_text: str,
    store: Any,
) -> list[V3FlaggedAttribute]:
    """Extractor 1: attribute contradictions on known entities.

    Delegates the hard work to the _detect_claim_contradictions
    pipeline (which already has the predicate patterns, normalisation, and
    store-triple matching), then projects each FlaggedClaim back onto a
    sentence index by walking the sentence list and finding the first
    unclaimed sentence that matches the claim's source text.
    """
    if not response_text:
        return []
    sentences = _split_sentences(response_text)
    flagged_claims: list[FlaggedClaim] = _detect_claim_contradictions(
        response_text, store
    )
    if not flagged_claims:
        return []

    out: list[V3FlaggedAttribute] = []
    used_indices: set[int] = set()
    for claim in flagged_claims:
        # v3 veto: earlier iterations compared claimed vs stored object
        # via strict string equality. That false-positives when the
        # response says "Example Corp" but the store fact says "at
        # Example" (captured without the suffix). If the claimed value
        # is a substring of any fact mentioning the subject, treat it as
        # agreement, not contradiction.
        if _claim_matches_any_stored_fact(claim, store):
            continue
        idx = _find_sentence_index(claim.sentence, sentences, used_indices)
        if idx is None:
            logger.warning(
                "v3 attribute: could not resolve sentence for claim %s/%s",
                claim.subject, claim.predicate,
            )
            continue
        used_indices.add(idx)
        correct_value = _consolidate_correction(claim.subject, claim.predicate, store)
        if not correct_value:
            correct_value = _summarise_store_fact(claim.stored)
        out.append(V3FlaggedAttribute(
            subject=claim.subject,
            predicate=claim.predicate,
            wrong_value=claim.claimed,
            correct_value=correct_value,
            sentence_index=idx,
            sentence=sentences[idx],
        ))
    return out


def _claim_matches_any_stored_fact(claim: FlaggedClaim, store: Any) -> bool:
    """Substring-aware veto for exact-match false positives.

    The attribute-correction detector compares `claimed == stored_object`
    exactly, but the response text and the store often have slightly
    different surface forms ("Example Corp" vs "Example"). If `claimed`
    is a substring of any current fact mentioning the subject, or vice
    versa, treat the claim as consistent with the store.
    """
    if store is None or getattr(store, "_conn", None) is None:
        return False
    lookup = "user" if claim.subject == "the_user" else claim.subject
    facts = _fetch_entity_facts(lookup, store)
    if lookup.lower() == "jamie":
        facts = facts + _fetch_entity_facts("user", store)
    claimed_l = (claim.claimed or "").strip().lower()
    if not claimed_l:
        return False
    for content in facts:
        cl = content.lower()
        if claimed_l in cl:
            return True
        # Token-level containment for multi-word claims: "example corp"
        # matches a fact that says "at example" because "example" ⊆ both.
        tokens = [t for t in claimed_l.split() if len(t) > 3]
        if tokens and any(t in cl for t in tokens):
            # Also require the fact to be predicate-relevant, not just
            # mentioning any token. Narrow by asking the store pattern
            # to produce at least one matching predicate triple.
            from sieve.verification import _extract_store_triples
            for subj_h, key, obj in _extract_store_triples(content):
                if key != claim.predicate:
                    continue
                if any(t in obj.lower() for t in tokens):
                    return True
    return False


def _consolidate_correction(subject: str, predicate: str, store: Any) -> str:
    """Merge all current stored facts about (subject, predicate) into one
    correction string.

    Without this, the attribute-correction detector would hand us a
    single fact ("Kim is a high school history teacher"), but the store
    often splits related attributes across multiple rows ("Kim is a
    history teacher", "Kim works at Brookline High"). The continuation
    model can't recover the missing piece if we only feed it one fact.

    Strategy: fetch all current facts mentioning the subject, keep those
    whose content looks relevant to the predicate (simple keyword filter),
    strip user-possessive prefixes, and join the two most informative
    clauses into a compact correction value.
    """
    if store is None or getattr(store, "_conn", None) is None:
        return ""
    lookup = "user" if subject == "the_user" else subject
    facts = _fetch_entity_facts(lookup, store)
    if not facts:
        return ""

    predicate_keywords: dict[str, tuple[str, ...]] = {
        "job_role": ("teacher", "lawyer", "pm", "manager", "analyst",
                     "director", "engineer", "vp", "consultant", "doctor",
                     "professor", "partner", "nurse", "writer", "designer",
                     "product", "product manager"),
        "job_employer": ("at ", "works at", "works for", "works in",
                         "employer", "company"),
        "residence": ("lives in", "lives at", "home", "condo", "apartment",
                      "house", "moved to"),
        "marital_state": ("married", "separated", "divorced", "single",
                          "engaged", "husband", "wife", "spouse"),
        "age": ("years old", "age ", "turned", "is "),
    }
    keywords = predicate_keywords.get(predicate, ())
    scored: list[tuple[int, str]] = []
    for content in facts:
        cl = content.lower()
        # Speculative / hypothetical facts never carry ground truth.
        if any(m in cl for m in (
            "is considering", "might", "may ", "could", "thinking about",
            "hopes to", "wants to", "decided against", "no longer wants",
        )):
            continue
        # Only keep facts that carry a positive predicate keyword. The old
        # length bonus rewarded short-but-irrelevant facts like "Kim is a
        # carnivore" when the target predicate was job_role. Demanding at
        # least one keyword match eliminates that noise.
        if not keywords:
            scored.append((0, content))
            continue
        kw_hits = sum(1 for k in keywords if k in cl)
        if kw_hits == 0:
            continue
        score = kw_hits * 2
        # Prefer facts that name the subject explicitly.
        if subject.lower() in cl or "user" in cl:
            score += 1
        scored.append((score, content))
    scored.sort(key=lambda p: (-p[0], len(p[1])))

    snippets: list[str] = []
    seen_cores: set[str] = set()
    for _, content in scored[:6]:
        trimmed = _summarise_store_fact(content)
        if not trimmed:
            continue
        core = trimmed.lower()[:40]
        if core in seen_cores:
            continue
        seen_cores.add(core)
        snippets.append(trimmed)
        if len(snippets) >= 2:
            break
    if not snippets:
        return ""
    if len(snippets) == 1:
        return snippets[0]
    return f"{snippets[0]}; {snippets[1]}"


def _find_sentence_index(
    needle: str,
    sentences: list[str],
    used: set[int],
) -> int | None:
    needle_norm = needle.strip()
    for i, s in enumerate(sentences):
        if i in used:
            continue
        if s.strip() == needle_norm:
            return i
    # Fallback: first containment match.
    for i, s in enumerate(sentences):
        if i in used:
            continue
        if needle_norm in s or s.strip() in needle_norm:
            return i
    return None


def _summarise_store_fact(fact_content: str) -> str:
    """Trim a stored fact to a compact correction value for the tool result."""
    text = fact_content.strip().rstrip(". ")
    # Common prefixes the writer emits — strip them so the tool result reads
    # as the pure fact value.
    for prefix in (
        "User's ", "the user's ", "User ", "the user ",
        "Jamie's ", "Jamie ", "FACT: ", "Fact: ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text[:200]


# ── Fabricated-relationship extractor ────────────────────────────────────────


# Shared relationship vocabulary used by both B extractor patterns.
_REL_VOCAB = (
    r"daughter|son|sister|brother|mother|father|mom|dad|"
    r"cousin|aunt|uncle|niece|nephew|grandmother|grandfather|"
    r"nanny|babysitter|housekeeper|chef|driver|assistant|"
    r"dog|cat|pet|puppy|kitten|bird|rabbit|hamster|"
    r"golden\s+retriever|labrador|poodle|terrier"
)

# Possessive fabrication pattern:
#   "Jamie's daughter Sam", "Jamie's cousin Lisa", "Jamie's nanny Clara",
#   "Kim's brother David". Anchored on a known entity via 's.
_POSSESSIVE_FAB_RX = re.compile(
    r"\b(?P<anchor>[A-Z][a-zA-Z]+)'s\s+"
    r"(?P<rel>" + _REL_VOCAB + r")"
    r"(?:\s+(?:named\s+)?(?P<name>[A-Z][a-zA-Z]+))?"
)

# Active "has a" fabrication pattern:
#   "Jamie has a daughter named Sam", "Jamie has a golden retriever named Max",
#   "Jamie has a nanny Clara". Covers the construction the possessive rx misses.
_HASA_FAB_RX = re.compile(
    r"\b(?P<anchor>[A-Z][a-zA-Z]+)\s+has\s+(?:a|an)\s+"
    r"(?P<rel>" + _REL_VOCAB + r")"
    r"(?:\s+(?:named\s+|called\s+)?(?P<name>[A-Z][a-zA-Z]+))?"
)

# Active fabrication pattern:
#   "Derek manages Jamie's team", "Lisa lives near Jamie".
#   Unknown proper noun + verb + known entity.
_ACTIVE_FAB_RX = re.compile(
    r"\b(?P<unknown>[A-Z][a-zA-Z]+)\s+"
    r"(?P<verb>manages|supervises|runs|leads|reports\s+to|helps|babysits|"
    r"tutors|teaches|coaches|drives|cooks\s+for)"
    r"\s+[^.]*?\b(?P<anchor>[A-Z][a-zA-Z]+)\b"
)


def extract_fabricated_relationships(
    response_text: str,
    store: Any,
) -> list[V3FabricatedRelationship]:
    """Extractor 2: unknown proper nouns bound to known entities.

    Guardrail (per design spec Q2): unknown proper nouns with NO connection
    to any known entity are NO DATA (pass through, not flagged). We only
    flag fabrications that claim a relationship to a known entity or the
    user.
    """
    if not response_text or store is None:
        return []
    sentences = _split_sentences(response_text)
    user_rels = _user_relationships(store)
    out: list[V3FabricatedRelationship] = []

    def _relationship_exists(anchor: str, rel: str) -> bool:
        """Does the store have a relationship of type `rel` anchored on `anchor`?"""
        rel_l = rel.lower()
        anchor_l = anchor.lower()
        # Canonicalise the anchor: Jamie / the user / user all map together.
        user_aliases = {"jamie", "jamie rivera", "user", "the_user"}
        is_user_anchor = anchor_l in user_aliases
        # Check against user_rels directly for user-anchored relationships.
        if is_user_anchor and rel_l in user_rels and user_rels[rel_l]:
            return True
        # For non-user anchors, fall back to a text search in the anchor's
        # facts — e.g. "Kim's brother" lives in Kim's facts, not user_rels.
        facts = _fetch_entity_facts(anchor, store)
        needle = f" {rel_l}"
        for f in facts:
            if needle in f" {f.lower()}":
                return True
        return False

    def _anchor_is_known(anchor: str) -> bool:
        return _is_known_entity(anchor, store)

    def _unknown_is_known(name: str) -> bool:
        return _is_known_entity(name, store)

    def _flag_match(i: int, sentence: str, anchor: str, rel: str, name: str | None) -> None:
        if not _anchor_is_known(anchor):
            return  # anchor itself isn't in the store — NO DATA
        rel_norm = rel.lower()
        if _relationship_exists(anchor, rel_norm):
            # Relationship exists. If a name is given, check whether the
            # stored facts mention that name — if not, the specific person
            # is fabricated even though the relationship type is real.
            if name is None:
                return
            if any(
                name.lower() in f.lower()
                for f in _fetch_entity_facts(anchor, store)
            ):
                return
            out.append(V3FabricatedRelationship(
                anchor=anchor,
                relationship=rel_norm,
                unknown_entity=name,
                sentence_index=i,
                sentence=sentence,
            ))
            return
        out.append(V3FabricatedRelationship(
            anchor=anchor,
            relationship=rel_norm,
            unknown_entity=name,
            sentence_index=i,
            sentence=sentence,
        ))

    # ── Possessive + has-a patterns ──────────────────────────────────────
    for i, sentence in enumerate(sentences):
        sl = sentence.lower()
        # Skip negative sentences — the response is disclaiming the relationship.
        if any(neg in sl for neg in (
            "no record", "not in", "doesn't have", "does not have",
            "there is no", "there are no", "no sibling", "no brother",
            "no sister", "no daughter", "no son", "no dog", "no cat",
            "no pet", "no cousin", "no aunt", "no uncle", "not on record",
        )):
            continue
        already_flagged_here: set[tuple[str, str]] = set()
        for rx in (_POSSESSIVE_FAB_RX, _HASA_FAB_RX):
            for m in rx.finditer(sentence):
                anchor = m.group("anchor")
                rel = m.group("rel")
                name = m.group("name") if "name" in m.groupdict() else None
                dedup_key = (anchor.lower(), rel.lower())
                if dedup_key in already_flagged_here:
                    continue
                already_flagged_here.add(dedup_key)
                _flag_match(i, sentence, anchor, rel, name)

    # ── Active patterns ──────────────────────────────────────────────────
    for i, sentence in enumerate(sentences):
        sl = sentence.lower()
        if any(neg in sl for neg in ("no record", "not in", "there is no")):
            continue
        for m in _ACTIVE_FAB_RX.finditer(sentence):
            unknown = m.group("unknown")
            anchor = m.group("anchor")
            if unknown == anchor:
                continue
            if not _anchor_is_known(anchor):
                continue  # neither side known — NO DATA
            if _unknown_is_known(unknown):
                continue  # both known, not a fabrication
            # Don't double-flag a sentence already flagged by the possessive pass.
            if any(f.sentence_index == i for f in out):
                continue
            out.append(V3FabricatedRelationship(
                anchor=anchor,
                relationship=m.group("verb"),
                unknown_entity=unknown,
                sentence_index=i,
                sentence=sentence,
            ))

    return out


# ── Top-level verification ───────────────────────────────────────────────────


def verify_response_v3(
    response_text: str,
    store: Any,
) -> V3Verification:
    """Run both extractors on the response and return a structured verdict.

    Pure-Python, deterministic, <5ms on clean responses. No LLM calls.
    """
    sentences = _split_sentences(response_text or "")
    if not response_text or len(response_text.strip()) < 5:
        return V3Verification(is_clean=True, sentences=sentences)

    attributes = extract_attribute_contradictions(response_text, store)
    fabricated = extract_fabricated_relationships(response_text, store)

    is_clean = not attributes and not fabricated
    return V3Verification(
        is_clean=is_clean,
        attributes=attributes,
        fabricated=fabricated,
        sentences=sentences,
    )


# ── Tool result builder ──────────────────────────────────────────────────────


def build_tool_result(
    verification: V3Verification,
    store: Any | None = None,
) -> dict[str, Any]:
    """Build the asymmetric two-field tool result.

    Per design spec Q3: user_facts for attribute corrections (pure data,
    model reconciles silently), not_in_records for fabrications (explicit
    negative, model has to drop the reference). No prose. No confirmations.
    Only what was flagged.

    Enrichment: when a subject has a flagged attribute on predicate X, we
    also include that subject's other covered predicates as bonus context.
    The correction for "Kim is a lawyer" should carry BOTH "role=history
    teacher" and "employer=Brookline High" so the model can produce a
    complete replacement sentence, not just the specific contradicted
    attribute. Bonus predicates are only filled if `store` is provided.
    """
    user_facts: dict[str, dict[str, str]] = {}
    flagged_subjects: set[str] = set()
    for a in verification.attributes:
        subj = a.subject
        flagged_subjects.add(subj)
        if subj not in user_facts:
            user_facts[subj] = {}
        user_facts[subj][a.predicate] = a.correct_value

    # Bonus context: fill in the other covered predicates for every
    # flagged subject, so the model has a complete view of that entity.
    if store is not None:
        for subj in flagged_subjects:
            existing = user_facts.get(subj, {})
            for predicate in ("job_role", "job_employer", "residence",
                              "marital_state", "age"):
                if predicate in existing:
                    continue
                value = _consolidate_correction(subj, predicate, store)
                if value:
                    existing[predicate] = value
            user_facts[subj] = existing

    not_in_records: list[str] = []
    seen: set[str] = set()
    for f in verification.fabricated:
        if f.unknown_entity:
            key = f"{f.unknown_entity} ({f.anchor}'s {f.relationship})"
        else:
            key = f"{f.anchor}'s {f.relationship}"
        if key not in seen:
            seen.add(key)
            not_in_records.append(key)

    result: dict[str, Any] = {}
    if user_facts:
        result["user_facts"] = user_facts
    if not_in_records:
        result["not_in_records"] = not_in_records
    return result


# ── Sentence-level splice ────────────────────────────────────────────────────


def splice_response(
    verification: V3Verification,
    continuation: str,
) -> str:
    """Assemble the final response: keep non-flagged sentences, append the
    model's continuation.

    Per design spec Q5: deletion-based splice, not rewriting. The
    continuation carries the corrected material — the model generated it
    with the tool result in hand.
    """
    kept_sentences = [
        s for i, s in enumerate(verification.sentences)
        if i not in verification.flagged_sentence_indices
    ]
    kept_text = " ".join(kept_sentences).strip()
    cont = (continuation or "").strip()
    if not kept_text:
        return cont
    if not cont:
        return kept_text
    return f"{kept_text} {cont}"


# ── Continuation payload builder ─────────────────────────────────────────────
#
# We do NOT invent a new prefill mechanism. We reuse the exact structural
# shape that the existing internal recall tool flow uses in
# src/recall_tool.py (RecallHandler.handle_chat, lines 117–176): append the
# original assistant turn verbatim, then append a new turn carrying the
# tool result, then ask the model to generate a fresh continuation. This is
# fresh-generation conditioned on new context — not prefill.
#
# Two wire shapes are supported, selectable via `shape`:
#
#   "tool_role" (Shape A) — fabricate an assistant.tool_calls=[verify_facts]
#       entry and append a {role:"tool", content:json} message. Matches
#       the real recall flow structurally. Risk: verify_facts is not in
#       the advertised tool schema, so Ollama/the model may find a tool
#       message orphaned. Best-case: cleanest semantically.
#
#   "user_data" (Shape C) — append the tool result as a {role:"user"}
#       message whose content is ONLY the JSON tool result (no prose, no
#       "please revise"). Always works mechanically because it's a plain
#       user turn. The earlier failure was the *prompt content* (corrective
#       instructions), not the role=user structure.
#
# The smoke test on 3-5 queries will decide which shape is empirically
# best. See `layer3_v3_smoke.py`.


def build_continuation_payload(
    lean_payload: dict[str, Any],
    original_assistant_text: str,
    original_assistant_message: dict[str, Any] | None,
    tool_result: dict[str, Any],
    shape: str = "user_data",
    continuation_max_tokens: int = 200,
    api_format: str = "ollama",
) -> dict[str, Any]:
    """Produce the Ollama /api/chat payload for v3 continuation.

    Args:
        lean_payload: the payload as sent to the LLM for the original call.
        original_assistant_text: the assistant's generated text (we return
            this separately because we need to splice it in Q5 later, and
            it may come from either the live response or a canned string
            in smoke tests).
        original_assistant_message: the full assistant message dict from
            the upstream response if available, else None. Used by
            shape="tool_role" to preserve tool_calls and formatting.
        tool_result: the asymmetric two-field dict from build_tool_result.
        shape: "user_data" or "tool_role". See module comment.
        continuation_max_tokens: cap on the continuation length.
        api_format: "ollama" or "openai".

    Returns:
        A dict ready to send to the upstream chat endpoint.
    """
    import json as _json

    payload = dict(lean_payload)
    messages = list(lean_payload.get("messages") or [])

    # 1. Append the original assistant turn verbatim. Mirror recall_tool.py's
    #    _extract_assistant_message — preserve structure if we have it,
    #    otherwise synthesise a minimal assistant turn from the text.
    if original_assistant_message is not None:
        messages.append(dict(original_assistant_message))
    else:
        messages.append({"role": "assistant", "content": original_assistant_text})

    # 2. Append the tool result in the chosen shape.
    if shape == "tool_role":
        # Fabricate a tool_calls entry on the preceding assistant turn so
        # the tool-role message has an anchor. Ollama's chat template
        # expects this pairing.
        if messages and messages[-1].get("role") == "assistant":
            asst = dict(messages[-1])
            existing_tcs = list(asst.get("tool_calls") or [])
            existing_tcs.append({
                "function": {"name": "verify_facts", "arguments": {}},
            })
            asst["tool_calls"] = existing_tcs
            messages[-1] = asst
        messages.append({
            "role": "tool",
            "content": _json.dumps(tool_result, ensure_ascii=False),
        })
    elif shape == "user_data":
        # Plain user turn carrying only the JSON tool result. No prose, no
        # instructions. The model sees authoritative data and continues.
        messages.append({
            "role": "user",
            "content": _json.dumps(tool_result, ensure_ascii=False),
        })
    else:
        raise ValueError(f"Unknown continuation shape: {shape!r}")

    payload["messages"] = messages
    payload["stream"] = False

    options = dict(payload.get("options") or {})
    options["num_predict"] = continuation_max_tokens
    payload["options"] = options

    return payload

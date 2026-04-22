"""Tier 2 fact classifier.

Takes a free-text fact (a readable sentence the existing writer already
produces well) and classifies it into structured tags — subject,
predicate, object, category — via Ollama native `/api/generate` with
format=JSON schema.

This module exists because forcing a small writer model to produce
structured slots AND readable content in one shot makes it hallucinate
values to fill slots it doesn't understand. The fix is to separate
concerns:

- Tier 1 writer -> content column (readable sentence)
- Tier 2 classifier (this module) -> structured columns

The classifier model is NOT hardcoded here; the caller must supply it
via the ``model=`` kwarg (resolved from ``ablation.tier2_classifier_model``
in config, defaulting to ``provider.default_model`` when set to 'auto').

If classification fails, the fact is still stored with NULL structured
columns. It remains searchable via vector similarity; it just does not
get slot indexing. Never discard a fact because classification failed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("recall.fact_classifier_v2")


# Closed sets. These intentionally overlap with QueryClassifierV2's slot
# predicates so slot_key lookups from retrieval can match writer-side tags.
PREDICATES: list[str] = [
    "marital_status", "spouse", "residence", "employer", "role",
    "age", "children", "health", "finances", "education", "interests",
    "relationships", "goals", "family", "location_change", "listing",
    "generic",
]
CATEGORIES: list[str] = [
    "personal", "work", "family", "relationship", "housing", "finance",
    "health", "education", "hobby", "goal",
]

_CLASSIFIER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "predicate": {"type": "string", "enum": PREDICATES},
        "object": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
    },
    "required": ["subject", "predicate", "object", "category"],
}

_PROMPT = """Classify the following fact into structured tags.

PREDICATES (pick ONE that best describes the type of information):
marital_status, spouse, residence, employer, role, age, children, health,
finances, education, interests, relationships, goals, family,
location_change, listing, generic

CATEGORIES (pick ONE):
personal, work, family, relationship, housing, finance, health, education,
hobby, goal

Rules:
- subject: the entity the fact is about (as a name, not a pronoun).
- predicate: the TYPE of information from the list above.
- object: the value or details (short phrase, not a full sentence).
- category: the broad category.

Fact: {content}

Respond with JSON only."""


@dataclass
class FactClassification:
    """Structured tags produced by the Tier 2 classifier.

    Any field can be None when the classifier failed or refused. The
    caller MUST treat None as "skip slot indexing, write with NULL
    structured columns" — never as a reason to discard the fact.
    """
    subject: str | None
    predicate: str | None
    object_literal: str | None
    category: str | None
    slot_key: str | None
    extraction_method: str = "tier2_llm_classifier"

    @property
    def is_populated(self) -> bool:
        return self.predicate is not None and self.subject is not None


def _slot_key_for(subject: str | None, predicate: str | None) -> str | None:
    """Produce a canonical slot_key like 'jamie_rivera:employer' or None."""
    if not subject or not predicate:
        return None
    sub = subject.strip().lower().replace(" ", "_")
    # Strip possessives / punctuation commonly emitted by LLMs.
    sub = sub.replace("'s", "").replace("'", "").rstrip("_")
    if not sub:
        return None
    return f"{sub}:{predicate}"


async def classify_fact_async(
    content: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str,  # required — resolved by caller from ablation.tier2_classifier_model
    num_ctx: int = 4096,
    timeout: float = 30.0,
) -> FactClassification:
    """Classify a free-text fact into structured tags.

    ``model`` is required; callers must resolve it from
    ``config.ablation.tier2_classifier_model`` (treating 'auto' as
    ``config.provider.default_model``).

    Returns a FactClassification; on any failure the fields are None
    (fact still gets written, just without slot indexing).
    """
    body: dict[str, Any] = {
        "model": model,
        "prompt": _PROMPT.format(content=content),
        "stream": False,
        "format": _CLASSIFIER_SCHEMA,
        "options": {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx},
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(f"{base_url}/api/generate", json=body)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
    except Exception as exc:
        logger.warning("tier2 classifier call failed: %s", exc)
        return FactClassification(None, None, None, None, None)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("tier2 classifier returned non-JSON: %s", raw[:200])
        return FactClassification(None, None, None, None, None)

    subject = (data.get("subject") or "").strip() or None
    predicate = data.get("predicate")
    obj = (data.get("object") or "").strip() or None
    category = data.get("category")
    # Validate closed sets — schema enforcement should already handle this
    # but be defensive against format=json fallback text.
    if predicate not in PREDICATES:
        predicate = None
    if category not in CATEGORIES:
        category = None

    return FactClassification(
        subject=subject,
        predicate=predicate,
        object_literal=obj,
        category=category,
        slot_key=_slot_key_for(subject, predicate),
    )


def classify_fact_sync(content: str, **kwargs: Any) -> FactClassification:
    """Synchronous wrapper for contexts that are not already in an event loop."""
    import asyncio
    return asyncio.run(classify_fact_async(content, **kwargs))

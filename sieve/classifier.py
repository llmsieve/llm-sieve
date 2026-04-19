"""Classifier — determines whether a query needs personal context retrieval.

Three levels, applied in order of cost:

  Level 0  Rule-based heuristics (zero compute, ~0.1ms)
           Personal pronouns, proper-noun entity references, temporal markers,
           prior-context signals.  Catches ~60-70% of cases.
           Default bias: RETRIEVE unless a clear reason not to.

  Level 1  Embedding similarity (<10ms with a loaded embed model)
           Encode the query, compare against the store's top-1 vector result.
           If max cosine similarity < threshold (default 0.7) → nothing relevant.

  Level 2  Tiny LLM (Phase 8 — not implemented here)

Usage::

    classifier = QueryClassifier(store, embed_fn=embedding_client.embed)
    decision = await classifier.classify("what's the weather where I live?")
    if decision.needs_retrieval:
        ...
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("recall.classifier")

# ─── L0 patterns ──────────────────────────────────────────────────────────────

# Signals that personal context IS likely needed
_PERSONAL_PRONOUN = re.compile(
    r"\b(my|mine|our|ours|I|I'm|I've|I'd|I'll|myself|we|we're|we've|"
    r"the\s+family|my\s+family)\b",
    re.IGNORECASE,
)
# Phrasal 'me' that is NOT personal (e.g. "tell me about", "show me", "explain to me")
_PHRASAL_ME = re.compile(
    r"\b(tell|show|give|explain|teach|describe|help|walk)\s+(to\s+)?me\b",
    re.IGNORECASE,
)
_PRIOR_CONTEXT = re.compile(
    r"\b(last time|previously|before|earlier|yesterday|last week|"
    r"you (said|told|mentioned|remember)|remember when|as (I|we) (said|mentioned|discussed))\b",
    re.IGNORECASE,
)
_TEMPORAL_PERSONAL = re.compile(
    r"\b(when did I|how long (have|has) (I|it)|since (I|we)|"
    r"how (old|long|many|much) (am I|do I|have I))\b",
    re.IGNORECASE,
)
_PERSONAL_QUESTION = re.compile(
    r"\b(where (do I|am I|did I|should I)|what (do I|am I|should I|did I)|"
    r"who (am I|do I|are my)|how (do I|am I|should I))\b",
    re.IGNORECASE,
)
# Cycle 19: possessive references to a person attribute. Catches queries
# like "Tell me about Mary's daughter" or "What is John's salary?" — the
# classifier previously missed these because they hit GENERIC_FACTUAL.
_POSSESSIVE_PERSON_ATTR = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:'s|\u2019s)\s+"
    r"(daughter|son|child|children|kid|kids|baby|twins|"
    r"wife|husband|spouse|partner|fianc[eé]e?|girlfriend|boyfriend|"
    r"mother|mom|mum|father|dad|parent|parents|"
    r"brother|sister|sibling|siblings|"
    r"grandmother|grandfather|grandparent|uncle|aunt|cousin|niece|nephew|"
    r"dog|cat|pet|puppy|kitten|"
    r"job|role|salary|income|company|employer|colleague|boss|career|career\s+path|"
    r"home|house|condo|apartment|address|residence|"
    r"opinion|preference|hobby|interest|favourite|favorite|"
    r"age|birthday|name|relationship|family|marriage|divorce)\b",
    re.IGNORECASE,
)
# Same idea but for "her/his/their X" — third-person possessives
_THIRD_PERSON_POSSESSIVE = re.compile(
    r"\b(?:her|his|their)\s+"
    r"(daughter|son|child|children|wife|husband|spouse|partner|"
    r"mother|father|parent|brother|sister|"
    r"job|role|salary|company|employer|colleague|"
    r"home|house|condo|apartment|residence|"
    r"opinion|preference|hobby)\b",
    re.IGNORECASE,
)

# Signals that NO personal context is needed (strong negative signals)
_GENERIC_FACTUAL = re.compile(
    r"^(what is|what are|who is|who are|when (is|was|did)|where is|how (does|do|is|are)|"
    r"define|explain|tell me about|describe|list|calculate|convert|translate)\b",
    re.IGNORECASE,
)
_PURE_TASK = re.compile(
    r"^(write|create|generate|make|build|code|program|draft|summarize|"
    r"format|fix|debug|calculate|convert|translate|list|help me (write|create|build|code))\b",
    re.IGNORECASE,
)

# Minimum query length to bother with retrieval at all
_MIN_RETRIEVAL_LENGTH = 4

# Phase-3 Fix 1: complexity detection. Queries that need a wider retrieval
# budget (multi-hop synthesis, explicit reference to a prior exchange,
# temporal "when should I / how long" questions) get complexity=2 so
# main.py can request top_k=10 instead of the default 5.
_FOLLOWUP_MARKERS = re.compile(
    r"\b(going back to|back to the|about the|regarding|as for the|"
    r"on the topic of|"
    r"you mentioned|we discussed|earlier you said|remember when|"
    r"what about the|any update on)\b",
    re.IGNORECASE,
)
_MULTIHOP_MARKERS = re.compile(
    r"\b(given that|considering|with everything|"
    r"priorities|planning)\b",
    re.IGNORECASE,
)
_COMPLEX_TEMPORAL = re.compile(
    r"\b(when should I|how long (?:have|has|will|should|do|does)|"
    r"timeline)\b",
    re.IGNORECASE,
)
# Rough proper-noun detection: capitalised tokens not at sentence start.
# Two or more signals multi-hop even without an explicit marker.
_PROPER_NOUN = re.compile(r"(?<!^)\b[A-Z][a-z]+\b")


# ─── Decision ────────────────────────────────────────────────────────────────

@dataclass
class ClassificationDecision:
    needs_retrieval: bool
    level: int                     # 0, 1, or 2 — which level decided
    reason: str                    # human-readable explanation
    confidence: float = 1.0       # 0..1
    topics: list[str] = field(default_factory=list)  # e.g. ["location", "family"]
    # Phase-3 Fix 1: retrieval complexity. 0=trivial/general, 1=simple
    # personal (default), 2=complex (follow-up / multi-hop / temporal).
    # main.py maps this to top_k ∈ {3, 5, 10}.
    complexity: int = 1


# ─── Classifier ───────────────────────────────────────────────────────────────

class QueryClassifier:
    """3-level query classifier. Decides whether context retrieval is needed.

    Args:
        store: MemoryStore instance (used for L1 entity name lookup + vector search).
        embed_fn: async callable(text) -> list[float].  Required for L1.
                  If None, L1 is skipped (L0 only).
        l1_threshold: cosine similarity threshold for L1 (default 0.7).
                      Above this → relevant context exists → retrieve.
    """

    def __init__(
        self,
        store: Any,
        embed_fn: Any | None = None,
        l1_threshold: float = 0.7,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._l1_threshold = l1_threshold

    async def classify(self, query: str) -> ClassificationDecision:
        """Classify a query. Returns a ClassificationDecision."""
        if not query or len(query.strip()) < _MIN_RETRIEVAL_LENGTH:
            return ClassificationDecision(
                needs_retrieval=False,
                level=0,
                reason="query too short",
                complexity=0,
            )

        complexity = _detect_complexity(query)

        # ── Level 0 ──────────────────────────────────────────────────────────
        l0 = self._classify_l0(query)
        if l0 is not None:
            logger.debug("L0 decision for %r: %s (%s)", query[:60], l0.needs_retrieval, l0.reason)
            # If L0 is a definitive NO, skip L1
            if not l0.needs_retrieval and l0.confidence >= 0.8:
                # Confident NO → trivial regardless of marker detection.
                l0.complexity = 0
                return l0
            # If L0 is a definitive YES, return it directly.
            # L1 is used as a tiebreaker for ambiguous cases, not to veto L0.
            if l0.needs_retrieval:
                l0.complexity = complexity
                return l0

        # ── Level 1: L0 was inconclusive, try embedding similarity ─────────
        if self._embed_fn is not None and self._store._conn is not None:
            l1 = await self._classify_l1(query, [])
            if l1 is not None:
                l1.complexity = complexity if l1.needs_retrieval else 0
                return l1

        # ── Default: retrieve (better safe than miss) ─────────────────────
        return ClassificationDecision(
            needs_retrieval=True,
            level=0,
            reason="default: retrieve (no strong negative signal)",
            confidence=0.5,
            complexity=complexity,
        )

    def _classify_l0(self, query: str) -> ClassificationDecision | None:
        """L0: pure regex. Returns decision or None if inconclusive."""
        query_stripped = query.strip()
        topics: list[str] = []

        # ── Positive signals ─────────────────────────────────────────────────
        has_personal_pronoun = bool(_PERSONAL_PRONOUN.search(query))
        has_prior_context = bool(_PRIOR_CONTEXT.search(query))
        has_temporal_personal = bool(_TEMPORAL_PERSONAL.search(query))
        has_personal_question = bool(_PERSONAL_QUESTION.search(query))
        # Cycle 19: possessive person references ("Mary's daughter", "his salary")
        has_possessive_person = bool(
            _POSSESSIVE_PERSON_ATTR.search(query)
            or _THIRD_PERSON_POSSESSIVE.search(query)
        )

        if has_personal_pronoun:
            topics.append("personal")
        if has_prior_context:
            topics.append("prior_context")
        if has_temporal_personal:
            topics.append("temporal")
        if has_possessive_person:
            topics.append("possessive_person")

        # ── Negative signals ─────────────────────────────────────────────────
        is_generic_factual = bool(_GENERIC_FACTUAL.match(query_stripped))
        is_pure_task = bool(_PURE_TASK.match(query_stripped))

        # Named entity references: check if any known entity name appears
        entity_hit = self._entity_in_query(query) if self._store._conn is not None else None
        if entity_hit:
            topics.append("entity")

        # ── Decision logic ───────────────────────────────────────────────────

        # Strong positive: personal pronoun, explicit personal question,
        # or possessive reference to a person attribute (cycle 19).
        if (has_personal_pronoun or has_personal_question or has_temporal_personal
                or has_possessive_person):
            reason = "personal pronoun / personal question detected"
            if has_possessive_person and not (has_personal_pronoun or has_personal_question):
                reason = "possessive person reference detected"
            return ClassificationDecision(
                needs_retrieval=True,
                level=0,
                reason=reason,
                confidence=0.85,
                topics=topics,
            )

        # Prior context reference
        if has_prior_context:
            return ClassificationDecision(
                needs_retrieval=True,
                level=0,
                reason="prior context reference detected",
                confidence=0.9,
                topics=topics,
            )

        # Known entity reference
        if entity_hit:
            return ClassificationDecision(
                needs_retrieval=True,
                level=0,
                reason=f"known entity '{entity_hit}' referenced in query",
                confidence=0.9,
                topics=topics,
            )

        # Strong negative: pure generic factual with no personal signals
        if is_generic_factual and not topics:
            return ClassificationDecision(
                needs_retrieval=False,
                level=0,
                reason="generic factual query, no personal signals",
                confidence=0.9,
            )

        # Pure task (write/generate/calculate) with no personal signals
        if is_pure_task and not topics:
            return ClassificationDecision(
                needs_retrieval=False,
                level=0,
                reason="pure task query, no personal signals",
                confidence=0.85,
            )

        return None  # inconclusive → fall through to L1 or default

    def _entity_in_query(self, query: str) -> str | None:
        """Return first known entity name found in query, or None."""
        try:
            rows = self._store.conn.execute(
                "SELECT name FROM entities ORDER BY name"
            ).fetchall()
        except Exception:
            return None

        query_lower = query.lower()
        for (name,) in rows:
            if name and len(name) > 2 and name.lower() in query_lower:
                return name
        return None

    async def _classify_l1(
        self, query: str, topics: list[str]
    ) -> ClassificationDecision | None:
        """L1: embed the query and check max similarity against store.

        Returns decision or None if embedding fails.
        """
        try:
            embedding = await self._embed_fn(query)
        except Exception as exc:
            logger.warning("L1 embed failed: %s", exc)
            return None

        try:
            results = self._store.search_facts_by_vector(embedding, limit=1)
        except Exception as exc:
            logger.warning("L1 vector search failed: %s", exc)
            return None

        if not results:
            return ClassificationDecision(
                needs_retrieval=False,
                level=1,
                reason="L1: no facts in store",
                confidence=0.95,
                topics=topics,
            )

        # sqlite-vec returns L2 distance — convert to pseudo-similarity
        # distance=0 → similarity=1, distance≥2 → similarity≈0
        distance = results[0].get("distance", 2.0)
        similarity = max(0.0, 1.0 - distance / 2.0)

        logger.debug(
            "L1: query=%r top_fact=%r dist=%.4f sim=%.4f threshold=%.2f",
            query[:50],
            results[0].get("content", "")[:50],
            distance,
            similarity,
            self._l1_threshold,
        )

        if similarity >= self._l1_threshold:
            return ClassificationDecision(
                needs_retrieval=True,
                level=1,
                reason=f"L1: relevant fact found (sim={similarity:.2f})",
                confidence=similarity,
                topics=topics,
            )
        else:
            return ClassificationDecision(
                needs_retrieval=False,
                level=1,
                reason=f"L1: no sufficiently relevant facts (sim={similarity:.2f} < {self._l1_threshold})",
                confidence=1.0 - similarity,
                topics=topics,
            )


def _detect_complexity(query: str) -> int:
    """Return 0/1/2 — the retrieval budget level for *query*.

    0: trivial / general knowledge (caller decides whether to even retrieve)
    1: simple personal — default for "my X", "where do I live", etc.
    2: complex — follow-up reference to a prior exchange, multi-hop
       synthesis, or temporal planning. main.py widens top_k to 10.
    """
    if not query:
        return 1
    if _FOLLOWUP_MARKERS.search(query):
        return 2
    if _COMPLEX_TEMPORAL.search(query):
        return 2
    if _MULTIHOP_MARKERS.search(query):
        return 2
    # Multi-entity synthesis heuristic: two or more proper nouns in one
    # query usually means the user is asking about several people/things.
    if len(_PROPER_NOUN.findall(query)) >= 2:
        return 2
    return 1


# ─── Tool selection classifier ───────────────────────────────────────────────
#
# Decides which of the agent's tools to inject into each outbound payload.
# Conceptually separate from QueryClassifier (fact retrieval) but lives in
# the same file because they share the embed_fn plumbing.
#
# L0 (keyword → category → tools) runs first. If it finds any match, that's
# the result. If not, L1 (embed(query) · tool.embedding > threshold) runs
# over all active tools. If BOTH are empty AND the query is non-trivial
# (≥5 words) AND fallback_include_all is on, return ALL active tools up to
# max_tools. Very short or trivial queries return an empty list (recall-only).
#
# Tools tagged with category "other" are never matched by L0 (by design —
# no keyword maps to them) but ARE fully eligible for L1 and fallback.

import math as _tool_math

from sieve.tool_registry import (
    CATEGORY_KEYWORDS,
    OTHER_CATEGORY,
    ToolRecord,
    ToolRegistry,
)


@dataclass
class ToolSelection:
    tools: list[dict]           # full_schema or lean_schema per registry compression
    reason: str
    level: int                  # 0=L0, 1=L1, -1=fallback, -2=empty
    confidence: float


class ToolClassifier:
    """Select which tools to inject into the lean payload for a given query.

    L0 keyword → category → tools runs first. If it finds a match, it wins
    outright and L1 is skipped. Otherwise L1 (query embedding · tool embedding)
    runs over all active tools. If both are empty AND the query is non-trivial
    (≥ fallback_min_words words) AND fallback_include_all is True, the fallback
    returns ALL active tools up to max_tools. Very short/trivial queries return
    an empty list (recall-only).

    Tools tagged with category "other" are never matched by L0 (by design — no
    keyword maps to them) but ARE fully eligible for L1 and fallback. This is
    how tools like image_generate, calendar_read, translate get picked up: L1
    similarity catches them when the query is relevant, and the fallback path
    catches them when the query is ambiguous.

    Args:
        registry:             ToolRegistry to enumerate active tools from.
        embed_fn:             async callable(text) -> list[float] for L1.
        l1_threshold:         cosine similarity cutoff for L1 selection (default 0.5).
        max_tools:            hard cap on injected tools (default 10).
        fallback_include_all: if True, ambiguous non-trivial queries get all tools.
        fallback_min_words:   query must have ≥ this many words to trigger fallback (default 5).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        embed_fn: Any | None = None,
        l1_threshold: float = 0.5,
        max_tools: int = 10,
        fallback_include_all: bool = True,
        fallback_min_words: int = 5,
    ) -> None:
        self._registry = registry
        self._embed_fn = embed_fn
        self._l1_threshold = l1_threshold
        self._max_tools = max_tools
        self._fallback_include_all = fallback_include_all
        self._fallback_min_words = fallback_min_words

    async def select(self, query: str) -> ToolSelection:
        """Return the set of tools to inject for this query.

        The returned tools use the stored `lean_schema` (which equals the full
        schema when compression='none'). Tools with category='other' are never
        matched by L0 but are eligible for L1 and fallback.
        """
        records = self._registry.get_active_records()
        if not records:
            return ToolSelection(tools=[], reason="empty registry", level=-2, confidence=1.0)

        q_lower = query.lower()
        q_words = [w for w in q_lower.split() if w]

        # --- L0: keyword → category → tools ---
        # For single-word keywords we use word-boundary matching so "run" doesn't
        # match "runway" and "file" doesn't match "profile". Multi-word keywords
        # (e.g. "read file") use substring match because they're already specific.
        matched_categories: set[str] = set()
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if " " in kw:
                    if kw in q_lower:
                        matched_categories.add(cat)
                        break
                else:
                    if re.search(rf"\b{re.escape(kw)}\b", q_lower):
                        matched_categories.add(cat)
                        break

        l0_tools = [
            r for r in records
            if r.category in matched_categories and r.category != OTHER_CATEGORY
        ]
        if l0_tools:
            chosen = l0_tools[: self._max_tools]
            logger.info(
                "Tool selection: level=L0 chosen=%s reason='category match: %s'",
                [r.name for r in chosen], sorted(matched_categories),
            )
            return ToolSelection(
                tools=[r.lean_schema for r in chosen],
                reason=f"L0 category match: {sorted(matched_categories)}",
                level=0,
                confidence=0.9,
            )

        # --- L1: embedding similarity ---
        if self._embed_fn is not None:
            try:
                q_vec = await self._embed_fn(query)
            except Exception as exc:
                logger.warning("Tool L1 embed failed: %s", exc)
                q_vec = None

            if q_vec is not None:
                scored: list[tuple[float, ToolRecord]] = []
                for r in records:
                    if r.embedding is None:
                        continue
                    sim = _tool_cosine(q_vec, r.embedding)
                    if sim >= self._l1_threshold:
                        scored.append((sim, r))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    chosen = [r for _, r in scored[: self._max_tools]]
                    logger.info(
                        "Tool selection: level=L1 chosen=%s reason='similarity ≥ %.2f'",
                        [r.name for r in chosen], self._l1_threshold,
                    )
                    return ToolSelection(
                        tools=[r.lean_schema for r in chosen],
                        reason=f"L1 similarity ≥ {self._l1_threshold}",
                        level=1,
                        confidence=scored[0][0],
                    )

        # --- Fallback ---
        if self._fallback_include_all and len(q_words) >= self._fallback_min_words:
            chosen = records[: self._max_tools]
            logger.info(
                "Tool selection: level=fallback chosen=%s reason='ambiguous query'",
                [r.name for r in chosen],
            )
            return ToolSelection(
                tools=[r.lean_schema for r in chosen],
                reason="fallback: ambiguous query",
                level=-1,
                confidence=0.3,
            )

        # Trivial query → no tools (recall-only)
        logger.info(
            "Tool selection: level=empty chosen=[] reason='trivial query, no matches'"
        )
        return ToolSelection(
            tools=[], reason="trivial query", level=-2, confidence=0.9,
        )


def _tool_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = _tool_math.sqrt(sum(x * x for x in a))
    nb = _tool_math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

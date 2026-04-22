"""Keyword-based query classifier for SlotRetriever routing.

Classifies a natural-language query into one of:
    slot_lookup         — current value of a single slot
                          ("what's Jamie's current job", "where does she live")
    temporal_sequence   — values over time
                          ("how has X changed", "over the last few years")
    multi_hop           — requires joining facts about several entities
                          ("who in her network", "birthday gifts given her situation")
    generic             — default; falls back to vector retrieval

The classifier is intentionally simple: a small ruleset with conservative
matches. Unknown queries route to generic, so the fall-back never
regresses.

Validated against the 10 simulation queries in
tests/test_query_classifier_v2.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["QueryClass", "QueryClassification", "classify_query", "slot_from_query"]


class QueryClass:
    SLOT_LOOKUP = "slot_lookup"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    MULTI_HOP = "multi_hop"
    GENERIC = "generic"


@dataclass
class QueryClassification:
    query_class: str
    slot_predicate: str | None = None        # populated for slot_lookup matches
    trigger: str | None = None                # which rule fired (for logging)


# ── Temporal cues — "over time" phrasing. Any hit routes to temporal. ──────
_TEMPORAL_PATTERNS = [
    r"\bover time\b",
    r"\bover the (?:last|past) (?:few |several )?(?:years?|months?|weeks?|days?)\b",
    r"\bacross the (?:story|conversation|timeline)\b",
    r"\bchanged? over\b",
    r"\bchange[sd]? over time\b",
    r"\bhow (?:has|have|did) .* changed\b",
    r"\bhow (?:has|have) .* evolved\b",
    r"\bwalk me through\b",
    r"\btimeline\b",
    r"\bhistory of\b",
    r"\bprogression\b",
    r"\bacross the\b.*\b(?:story|timeline|conversation)\b",
    r"\bhappened with .* (?:across|over|through)\b",
]
_TEMPORAL_RE = [re.compile(p, re.IGNORECASE) for p in _TEMPORAL_PATTERNS]


# ── Slot lookup rules — (regex, slot_predicate) ordered by specificity. ─
# Each regex fires on its keyword set. First match wins.
_SLOT_RULES: list[tuple[re.Pattern, str]] = [
    # Employment
    (re.compile(r"\b(current|present|right now|these days)\b.*\b(job|role|position|title|work)\b", re.I), "role"),
    (re.compile(r"\b(job title|job|role|position|title)\b.*\b(current|now|these days|currently)\b", re.I), "role"),
    (re.compile(r"\bwhat(?:'s| is) .* (?:job title|current job|current role|role|position)\b", re.I), "role"),
    (re.compile(r"\bwhere does .* work\b", re.I), "employer"),
    (re.compile(r"\bwho does .* work for\b", re.I), "employer"),
    (re.compile(r"\bcurrent employer\b", re.I), "employer"),
    (re.compile(r"\b\w+'?s (?:current )?(?:employer|company)\b", re.I), "employer"),
    # Residence
    (re.compile(r"\bwhere does .* live\b", re.I), "residence_city"),
    (re.compile(r"\bwhere .* lives\b", re.I), "residence_city"),
    (re.compile(r"\bcurrent (?:living situation|residence|address|home|address)\b", re.I), "residence_city"),
    (re.compile(r"\bwhat(?:'s| is) .* current living situation\b", re.I), "residence_city"),
    # Marital / relationship status
    (re.compile(r"\bis .* (?:still )?married\b", re.I), "marital_status"),
    (re.compile(r"\bmarital status\b", re.I), "marital_status"),
    (re.compile(r"\bis .* divorced\b", re.I), "marital_status"),
    (re.compile(r"\bis .* separated\b", re.I), "marital_status"),
    # Housing costs
    (re.compile(r"\bhow much does .* (?:spend on|pay for) housing\b", re.I), "monthly_mortgage"),
    (re.compile(r"\bhousing (?:budget|cost)\b", re.I), "housing_budget"),
    (re.compile(r"\brent(?: cost)?\b", re.I), "monthly_rent"),
    (re.compile(r"\bmortgage\b", re.I), "monthly_mortgage"),
    # Identity basics
    (re.compile(r"\bhow old\b", re.I), "age"),
    (re.compile(r"\bwhat(?:'s| is) .* age\b", re.I), "age"),
    (re.compile(r"\bwhere was .* born\b", re.I), "birthplace"),
    # Salary
    (re.compile(r"\b(salary|pay|compensation|base)\b", re.I), "salary"),
]


# ── Multi-hop / generic cues — these hint at multi-fact joins ──────────────
_MULTI_HOP_PATTERNS = [
    r"\bwho in .* (?:network|circle|family|team)\b",
    r"\bwho could (?:help|support|advise)\b",
    r"\bprofessional network\b",
    r"\bwhat .* given .* (?:situation|circumstances|financial)\b",
    r"\bgiven (?:her|his|their|\w+'?s) current\b",
    r"\bwhat .* work (?:for|with)\b.*\b(?:given|considering)\b",
    r"\breasonably (?:commute|afford|manage)\b",
    r"\bshould .* consider\b",
    r"\bwhat (?:health|financial|personal) (?:considerations?|factors?)\b",
]
_MULTI_HOP_RE = [re.compile(p, re.IGNORECASE) for p in _MULTI_HOP_PATTERNS]


def classify_query(query: str) -> QueryClassification:
    """Classify *query* into one of the four routing classes.

    Order of precedence (highest first):
    1. Temporal cues  → temporal_sequence
    2. Multi-hop cues → multi_hop
    3. Slot rules     → slot_lookup (with slot_predicate)
    4. Default        → generic
    """
    q = query.strip()
    if not q:
        return QueryClassification(QueryClass.GENERIC, trigger="empty")

    # Temporal first — "how has X changed" beats "current X".
    for rx in _TEMPORAL_RE:
        if rx.search(q):
            return QueryClassification(
                QueryClass.TEMPORAL_SEQUENCE, trigger=rx.pattern
            )

    # Multi-hop cues next — they outrank slot rules because a multi-hop
    # query like "who in her network could help with a career transition"
    # would otherwise match the employment slot rule on the word "career".
    for rx in _MULTI_HOP_RE:
        if rx.search(q):
            return QueryClassification(
                QueryClass.MULTI_HOP, trigger=rx.pattern
            )

    # Slot rules — first match wins, ordered by specificity above.
    for rx, predicate in _SLOT_RULES:
        if rx.search(q):
            return QueryClassification(
                QueryClass.SLOT_LOOKUP,
                slot_predicate=predicate,
                trigger=rx.pattern,
            )

    return QueryClassification(QueryClass.GENERIC, trigger="default")


def slot_from_query(query: str, profile_owner_name: str) -> str | None:
    """Convenience: return a canonical slot_key for a slot_lookup query.

    Returns None if the query does not classify as slot_lookup.
    """
    cls = classify_query(query)
    if cls.query_class != QueryClass.SLOT_LOOKUP or not cls.slot_predicate:
        return None
    subject = re.sub(r"[^a-z0-9]+", "_", profile_owner_name.lower()).strip("_")
    return f"{subject}:{cls.slot_predicate}"

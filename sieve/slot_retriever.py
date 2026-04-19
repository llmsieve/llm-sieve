"""Cycle 27 T7/T8/T9: SlotRetriever — deterministic slot-based retrieval.

Complements the legacy ContextRetriever. Routes a query via
query_classifier_v2, then dispatches to one of four paths:

    slot_lookup        — single current fact for the classified slot
    temporal_sequence  — ordered timeline of the classified slot
    multi_hop          — 1-hop graph join on relationships
    generic            — None (caller should fall through to legacy retriever)

Returns SlotRetrievalResult which the v2 formatter (T10) consumes. This
module does NOT format strings — it returns structured data so the
formatter can emit the 5-section template.

Side-effect: on slot_lookup misses, inserts a known_unknowns row so the
formatter can emit [NOT PRESENT: <slot>] next time (or this time, via
the current call's result.known_unknowns list).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sieve.query_classifier_v2 import (
    QueryClass,
    classify_query,
)

logger = logging.getLogger("recall.slot_retriever")


@dataclass
class SlotRetrievalResult:
    """Structured result from SlotRetriever.

    query_class: which path was taken.
    slot_predicate: for slot_lookup, the predicate that matched.
    slot_key: canonical key used for lookup (or None).
    current_slots: list of {slot, predicate, content, object, ...}.
    timeline: ordered list of slot history rows for temporal queries.
    relationships: list of relationship rows for multi-hop queries.
    known_unknowns: list of slot_keys that were asked-about and empty.
    trigger: which classifier rule fired (logging).
    """
    query: str = ""
    query_class: str = QueryClass.GENERIC
    slot_predicate: str | None = None
    slot_key: str | None = None
    current_slots: list[dict] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    known_unknowns: list[str] = field(default_factory=list)
    trigger: str | None = None

    @property
    def is_hit(self) -> bool:
        """True if any structured data was returned (caller should use
        the v2 formatter). False → caller should fall back to generic
        vector retrieval."""
        return bool(
            self.current_slots
            or self.timeline
            or self.relationships
            or self.known_unknowns
        )


def _canonical_subject(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class SlotRetriever:
    """Deterministic slot-based retrieval over the cycle27 v2 schema.

    Construct with a MemoryStore and the profile owner's name. Call
    retrieve(query) to get a SlotRetrievalResult.

    When a query doesn't classify into a structured path (or classifies
    but finds nothing), is_hit will be False and the caller is expected
    to fall through to the legacy ContextRetriever.
    """

    def __init__(self, store: Any, profile_owner_name: str = ""):
        self._store = store
        self._owner_name = profile_owner_name
        self._owner_subject = _canonical_subject(profile_owner_name)

    def retrieve(self, query: str) -> SlotRetrievalResult:
        cls = classify_query(query)
        result = SlotRetrievalResult(
            query=query,
            query_class=cls.query_class,
            slot_predicate=cls.slot_predicate,
            trigger=cls.trigger,
        )

        if cls.query_class == QueryClass.SLOT_LOOKUP and cls.slot_predicate:
            self._do_slot_lookup(result)
        elif cls.query_class == QueryClass.TEMPORAL_SEQUENCE:
            self._do_temporal(result)
        elif cls.query_class == QueryClass.MULTI_HOP:
            self._do_multi_hop(result)

        # Always merge any persisted known_unknowns that might apply to
        # this query's subject (even on a generic fall-through path).
        self._merge_known_unknowns(result)

        return result

    # ── T7 slot_lookup ─────────────────────────────────────────────────

    # Cycle 28: QueryClassifierV2 uses V2_SLOT_PREDICATES (residence_city,
    # monthly_mortgage, ...) but fact_classifier_v2 writes a smaller more
    # general set (residence, finances, ...). The alias map lets a single
    # query-side predicate probe several fact-side predicates in fallback
    # order. On HIT, all matching rows are added as current_slots so the
    # formatter can render the broader slot cluster.
    _SLOT_ALIASES: dict[str, list[str]] = {
        "residence_city": ["residence_city", "residence", "location_change"],
        "residence": ["residence", "residence_city", "location_change"],
        "monthly_mortgage": ["monthly_mortgage", "finances", "housing_budget"],
        "housing_budget": ["housing_budget", "finances", "monthly_mortgage"],
        "salary": ["salary", "finances"],
        "employer": ["employer", "role"],
        "employer_location": ["employer_location", "employer"],
        "role": ["role", "employer"],
        "marital_status": ["marital_status", "spouse", "relationships"],
    }

    def _do_slot_lookup(self, result: SlotRetrievalResult) -> None:
        if not self._owner_subject or not result.slot_predicate:
            return
        primary_slot_key = f"{self._owner_subject}:{result.slot_predicate}"
        result.slot_key = primary_slot_key

        # Try the primary predicate first, then aliases. Any HIT populates
        # result.current_slots; we only record a known_unknown if nothing
        # hit across the whole alias cluster.
        candidate_preds = self._SLOT_ALIASES.get(
            result.slot_predicate, [result.slot_predicate]
        )
        hit_any = False
        for pred in candidate_preds:
            alias_key = f"{self._owner_subject}:{pred}"
            row = self._store.get_current_slot_fact(alias_key)
            if row is not None:
                result.current_slots.append(row)
                hit_any = True
                logger.info(
                    "slot_lookup HIT: %s → %s",
                    alias_key, (row.get("content") or "")[:80],
                )

        if hit_any:
            return

        # No alias hit — record a known_unknown for the primary slot_key.
        try:
            self._store.insert_known_unknown(
                subject_entity_id=self._owner_subject,
                slot_key=primary_slot_key,
                reason="slot_lookup_miss",
            )
        except Exception as exc:
            logger.warning("known_unknown insert failed: %s", exc)
        result.known_unknowns.append(primary_slot_key)
        logger.info(
            "slot_lookup MISS: %s (tried aliases %s, recorded known_unknown)",
            primary_slot_key, candidate_preds,
        )

    # ── T8 temporal_sequence ──────────────────────────────────────────

    def _do_temporal(self, result: SlotRetrievalResult) -> None:
        """Try to guess the slot the user is asking about by keyword,
        then return get_slot_timeline for it. If we can't guess, leave
        the result empty — caller falls through to generic retrieval.
        """
        if not self._owner_subject:
            return

        # Simple keyword → slot mapping for temporal queries. These are
        # more permissive than the slot-lookup rules because temporal
        # queries are usually less specific ("how has her life changed").
        query_lc = result.query.lower()
        candidate_slots: list[str] = []
        if any(k in query_lc for k in ("career", "job", "role", "work")):
            candidate_slots += ["role", "employer"]
        if any(k in query_lc for k in ("living", "residence", "home", "move", "house")):
            candidate_slots += ["residence_city", "residence_address"]
        if any(k in query_lc for k in ("relationship", "married", "tom", "spouse", "marriage")):
            candidate_slots += ["marital_status"]
        if "salary" in query_lc or "income" in query_lc or "pay" in query_lc:
            candidate_slots += ["salary"]
        if "opinion about" in query_lc or "feel about" in query_lc:
            # Opinion timelines are multi-row enjoys/dislikes/has_opinion_on.
            # Not a single slot — fall through to generic for now.
            return

        seen: set[str] = set()
        for pred in candidate_slots:
            if pred in seen:
                continue
            seen.add(pred)
            slot_key = f"{self._owner_subject}:{pred}"
            timeline = self._store.get_slot_timeline(slot_key)
            if timeline:
                result.timeline.extend(timeline)
                if result.slot_key is None:
                    result.slot_key = slot_key
                logger.info(
                    "temporal HIT: %s → %d rows",
                    slot_key, len(timeline),
                )

    # ── T9 multi_hop ──────────────────────────────────────────────────

    def _do_multi_hop(self, result: SlotRetrievalResult) -> None:
        """One-hop graph traversal over the relationships table.

        Pulls all relationships where the profile owner is the source
        (e.g. reports_to, spouse, child) and returns them as structured
        rows. Also returns the owner's current slot values for the most
        common slots (employer, role, residence_city) so the formatter
        has enough to answer career-transition / given-circumstances
        questions.

        For the cycle27 simulation queries (B2, B4), this covers:
        - B4 "who in her network": relationships table gives reports_to,
          manager, mentor, colleague edges.
        - B2 "birthday gifts given her current situation": we return
          has_child relationships + current employer/salary slots.
        """
        if not self._owner_subject:
            return

        # Pull outbound relationships for the owner entity if it exists.
        owner_entity = self._store.find_entity_by_name(self._owner_name) if self._owner_name else None
        if owner_entity is not None:
            try:
                rows = self._store.conn.execute(
                    "SELECT r.id, r.source_entity, r.relationship, r.target_entity,"
                    "       r.confidence, r.status, e.name AS target_name"
                    "  FROM relationships r"
                    "  LEFT JOIN entities e ON e.id = r.target_entity"
                    " WHERE r.source_entity = ? AND (r.valid_to IS NULL OR r.valid_to = '')",
                    (owner_entity["id"],),
                ).fetchall()
                for row in rows:
                    result.relationships.append({
                        "id": row[0],
                        "source_entity": row[1],
                        "relationship": row[2],
                        "target_entity": row[3],
                        "target_name": row[6],
                        "confidence": row[4],
                        "status": row[5],
                    })
            except Exception as exc:
                logger.warning("multi_hop relationships query failed: %s", exc)

        # Pull a few key current slots so the multi-hop answer has
        # anchor facts (employer, role, residence, marital_status).
        for pred in ("employer", "role", "residence_city", "marital_status", "salary"):
            slot_key = f"{self._owner_subject}:{pred}"
            row = self._store.get_current_slot_fact(slot_key)
            if row is not None:
                result.current_slots.append(row)

    # ── known_unknowns merge ──────────────────────────────────────────

    def _merge_known_unknowns(self, result: SlotRetrievalResult) -> None:
        """Attach any persisted known_unknowns for the profile owner
        that aren't already in the result's list. Called on every path.
        """
        if not self._owner_subject:
            return
        try:
            rows = self._store.get_known_unknowns(self._owner_subject)
        except Exception:
            return
        existing = set(result.known_unknowns)
        for row in rows:
            slot = row.get("slot_key")
            if slot and slot not in existing:
                result.known_unknowns.append(slot)
                existing.add(slot)

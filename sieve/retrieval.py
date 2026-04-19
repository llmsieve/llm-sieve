"""Retrieval — fetch relevant context from the memory store.

Pipeline:
  1. Vector search the facts table for top-k matches
  2. Extract entity IDs from those results
  3. Graph-traverse one hop on the relationships table → related facts
  4. De-duplicate, filter status=current, sort by confidence + recency
  5. Assemble a short context block (~150 tokens) ready to inject into the payload

Usage::

    retriever = ContextRetriever(store, embed_fn=embedding_client.embed)
    ctx = await retriever.retrieve("weather where I live", top_k=5)
    # ctx.text  → "## Recalled context\n- User lives in Dubai\n..."
    # ctx.facts → list of fact dicts
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sieve.store import deserialize_float32

logger = logging.getLogger("recall.retrieval")

# ── Cycle 18: context_format auto dispatch ───────────────────────────────────
# Queries asking about evolution / progression / history benefit from the
# structured (timeline-grouped) format. Single-fact lookups regress under
# structured because of elaboration noise — the cycle 12 finding. The auto
# mode dispatches based on a small keyword set in the query text.
_TEMPORAL_QUERY_PATTERN = re.compile(
    r"\b(over time|over the years|progression|progress|"
    r"has changed|have changed|how (?:has|have).+changed|"
    r"career path|trajectory|evolution|evolved|"
    r"timeline|life story|family history|relationship history|work history|"
    r"journey|across the story|throughout the story|over the story|"
    r"walk (?:me|us) through|describe (?:.+ )?(?:life|career|history|journey)|"
    # Require "tell me about" to be paired with a temporal noun, not bare —
    # bare "tell me about X" is a single-fact query (cycle 18 D2 fix).
    r"tell (?:me|us) about (?:.+ )?(?:life|story|career|history|journey|past)|"
    r"used to|previously|"
    r"first.+then|started.+now|"
    r"changes? in|"
    r"development of|develop(?:ed|ing)\s+(?:into|from)|"
    r"how did .+ (?:become|get|end up))\b",
    re.IGNORECASE,
)


def _pick_format(query: str) -> str:
    """Cycle 18: dispatch context format based on query type.

    Returns "structured" for queries about progression / change / history,
    otherwise "flat". Used when ContextRetriever is initialised with
    context_format="auto".
    """
    if not query:
        return "flat"
    if _TEMPORAL_QUERY_PATTERN.search(query):
        return "structured"
    return "flat"

# Maximum facts to surface in the context block
_DEFAULT_TOP_K = 15
_MAX_FACTS = 15
# Rough token budget for the context block (injected as system message)
_CONTEXT_TOKEN_BUDGET = 150

# Cycle 9 — retrieval dedup + MMR diversity re-ranking
_CANDIDATE_MULTIPLIER = 2          # fetch 2× top_k candidates before dedup
_DEDUP_SIMILARITY = 0.9            # cosine similarity threshold for content dedup
_TEMPORAL_DEDUP_SIMILARITY = 0.85  # Cycle 26 Fix 2 — separate from content dedup
# Cycle 26 Fix 3a: the legacy format cap (_CONTEXT_TOKEN_BUDGET=150) was
# aspirational — the old format functions did not actually truncate. Setting
# the new max_tokens default to the tier-0 budget from the adaptive budget
# table preserves pre-Cycle-26 behavior for callers that don't pass a budget
# explicitly (e.g., recall_tool.py inner loop, tests, future callers).
_LEGACY_DEFAULT_FORMAT_TOKENS = 1300
_GRAPH_RESERVE = 3                 # slots always reserved for graph-hop facts
_MMR_LAMBDA = 0.7                  # 0.7 relevance / 0.3 diversity
_MMR_ENABLED = True                # kill-switch: False → behave like Cycle 8


@dataclass
class RetrievedContext:
    """Result of a retrieval operation."""
    facts: list[dict] = field(default_factory=list)
    text: str = ""                  # ready-to-inject string
    query: str = ""
    token_estimate: int = 0
    retrieved_from_graph: int = 0   # facts added via graph traversal


class ContextRetriever:
    """Retrieves relevant context from the memory store for a given query.

    Args:
        store: MemoryStore instance.
        embed_fn: async callable(text) -> list[float].
                  If None, retrieval falls back to recency-ordered facts (no vector search).
        top_k: number of vector results to fetch.
    """

    def __init__(
        self,
        store: Any,
        embed_fn: Any | None = None,
        top_k: int = _DEFAULT_TOP_K,
        graph_traversal: bool = True,
        temporal_versioning: bool = True,
        context_format: str = "auto",
        temporal_dedup_enabled: bool = True,
        reranker: Any | None = None,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._top_k = top_k
        self._graph_traversal = graph_traversal
        self._temporal_versioning = temporal_versioning
        self._context_format = (
            context_format if context_format in ("flat", "structured", "auto") else "auto"
        )
        self._temporal_dedup_enabled = temporal_dedup_enabled
        # Cycle 30 Fix 5: optional RerankerService applied to primary
        # candidates after vector search. No-op if None or if the service
        # reports unavailable.
        self._reranker = reranker

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        include_episodes: bool = False,
    ) -> RetrievedContext:
        """Retrieve relevant facts for *query*.

        Returns a RetrievedContext with a pre-formatted text block.

        Args:
            top_k: hard cap on fact count for this query.
            include_episodes: Phase-3 Fix 2 — when True, vector-search the
                episodes table and append a ``## Recalled episodes``
                footer with up to 2 summaries of prior conversation
                turns. Only set True for follow-up queries; it roughly
                doubles context-block tokens otherwise.
        """
        k = top_k or self._top_k

        if self._store._conn is None:
            return RetrievedContext(query=query)

        # ── 1. Vector search — widened candidate pool ─────────────────────
        candidate_limit = k * _CANDIDATE_MULTIPLIER if _MMR_ENABLED else k
        vector_facts: list[dict] = []
        if self._embed_fn is not None:
            try:
                embedding = await self._embed_fn(query)
                vector_facts = self._store.search_facts_by_vector(embedding, limit=candidate_limit)
            except Exception as exc:
                logger.warning("Vector search failed: %s", exc)

        if not vector_facts:
            vector_facts = self._fallback_recent_facts(candidate_limit)

        # ── 1b. Cross-encoder re-rank (Cycle 30 Fix 5) ────────────────────
        # Run the cross-encoder over the widened candidate pool to re-order
        # by genuine query-fact relevance before dedup and graph-hop
        # merging. No-op if the reranker is unavailable or absent.
        reranker_available = (
            self._reranker is not None
            and getattr(self._reranker, "available", False)
            and bool(vector_facts)
        )
        if reranker_available:
            contents = [(f.get("content") or "") for f in vector_facts]
            scores = self._reranker.rerank(query, contents)
            if scores is not None and len(scores) == len(vector_facts):
                ordered = sorted(
                    zip(scores, vector_facts),
                    key=lambda p: p[0],
                    reverse=True,
                )
                # Preserve existing keys; add rerank_score for telemetry
                # and dedup layers (MMR is unaffected since it keys off
                # `distance`, which we leave untouched).
                vector_facts = []
                for score, fact in ordered:
                    fact_copy = dict(fact)
                    fact_copy["rerank_score"] = score
                    vector_facts.append(fact_copy)
                logger.info(
                    "Rerank: scored %d candidates, top score=%.3f, bottom=%.3f",
                    len(ordered),
                    ordered[0][0] if ordered else 0.0,
                    ordered[-1][0] if ordered else 0.0,
                )

        # ── 2. Filter status, collect entity IDs ──────────────────────────
        entity_ids: set[str] = set()
        entity_ids_by_fact: dict[str, set[str]] = {}
        fact_ids_seen: set[str] = set()
        primary_candidates: list[dict] = []
        for f in vector_facts:
            if self._temporal_versioning:
                if f.get("status") not in (None, "current", "provisional"):
                    continue
            primary_candidates.append(f)
            fact_ids_seen.add(f["id"])
            f_ents = set(self._get_entity_ids_for_fact(f["id"]))
            entity_ids_by_fact[f["id"]] = f_ents
            entity_ids.update(f_ents)

        # Final cap: honour caller's top_k but never exceed the retriever's
        # hard _MAX_FACTS ceiling. This is the number of facts that will
        # ultimately be returned to the caller.
        final_cap = min(k, _MAX_FACTS)

        # ── 3. Content-similarity dedup over primaries ────────────────────
        content_embs: dict[str, list[float]] = {}
        if _MMR_ENABLED:
            content_embs = _load_content_embeddings(
                self._store, [f["id"] for f in primary_candidates]
            )
            primary_cap = max(1, final_cap - _GRAPH_RESERVE)
            primary_facts = _dedup_by_content(
                primary_candidates, content_embs,
                threshold=_DEDUP_SIMILARITY, cap=primary_cap,
            )
        else:
            primary_facts = primary_candidates[:final_cap]

        fact_ids_seen = {f["id"] for f in primary_facts}

        # ── 4. Graph traversal — 1 hop (ABL-GR) ───────────────────────────
        graph_facts: list[dict] = []
        graph_hit_count = 0

        if not self._graph_traversal:
            logger.info("ABL-GR: graph traversal disabled")
            entity_ids = set()

        # Graph-hop budget. With MMR enabled we always reserve at least
        # _GRAPH_RESERVE slots; the final MMR cap below trims back to
        # final_cap regardless.
        if _MMR_ENABLED:
            _GRAPH_ADD_CAP = max(_GRAPH_RESERVE, final_cap - len(primary_facts))
        else:
            _GRAPH_ADD_CAP = max(1, final_cap - len(primary_facts))

        for entity_id in entity_ids:
            if graph_hit_count >= _GRAPH_ADD_CAP:
                break
            related = self._store.get_related_entities(entity_id)
            for rel_entity in related:
                if graph_hit_count >= _GRAPH_ADD_CAP:
                    break
                rel_facts = self._get_facts_for_entity(rel_entity["id"])
                for f in rel_facts:
                    if graph_hit_count >= _GRAPH_ADD_CAP:
                        break
                    if f["id"] not in fact_ids_seen:
                        status_ok = (
                            not self._temporal_versioning
                            or f.get("status") in (None, "current", "provisional")
                        )
                        if status_ok:
                            graph_facts.append(f)
                            fact_ids_seen.add(f["id"])
                            entity_ids_by_fact[f["id"]] = set(
                                self._get_entity_ids_for_fact(f["id"])
                            )
                            graph_hit_count += 1

        # ── 5. Merge + rank ────────────────────────────────────────────────
        if _MMR_ENABLED:
            missing_ids = [f["id"] for f in graph_facts if f["id"] not in content_embs]
            if missing_ids:
                content_embs.update(_load_content_embeddings(self._store, missing_ids))
            combined = primary_facts + graph_facts
            all_facts = _mmr_rerank(combined, content_embs, lam=_MMR_LAMBDA, k=final_cap)
        else:
            _FALLBACK_DISTANCE = 2.0
            def _score(f: dict) -> float:
                d = f.get("distance")
                if d is None:
                    d = _FALLBACK_DISTANCE
                conf = f.get("confidence") or 0.75
                return float(d) - 0.15 * float(conf)
            all_facts = primary_facts + graph_facts
            all_facts.sort(key=_score)
            all_facts = all_facts[:final_cap]

        # ── 5b. Cycle 26 Fix 2: post-retrieval temporal dedup ───────────────
        dedup_dropped = 0
        if self._temporal_dedup_enabled and len(all_facts) > 1:
            # Ensure every fact in the final set has a content embedding loaded.
            # The MMR branch above already loads these; this is a safety net for
            # the non-MMR path and any future code changes.
            missing = [f["id"] for f in all_facts if f["id"] not in content_embs]
            if missing:
                content_embs.update(_load_content_embeddings(self._store, missing))
            before_count = len(all_facts)
            all_facts = _temporal_dedup(
                all_facts, content_embs, entity_ids_by_fact,
            )
            dedup_dropped = before_count - len(all_facts)

        # ── 6. Assemble context block ─────────────────────────────────────
        # Cycle 18: when mode is "auto", dispatch per-query.
        if self._context_format == "auto":
            chosen_format = _pick_format(query)
        else:
            chosen_format = self._context_format

        if chosen_format == "structured":
            text = _format_context_block_structured(
                all_facts, max_tokens=_LEGACY_DEFAULT_FORMAT_TOKENS,
            )
        else:
            text = _format_context_block(
                all_facts, max_tokens=_LEGACY_DEFAULT_FORMAT_TOKENS,
            )

        # Phase-3 Fix 2: append episode footer for follow-up queries.
        episodes_used = 0
        if include_episodes and self._embed_fn is not None:
            try:
                ep_embedding = await self._embed_fn(query)
                episodes = self._store.search_episodes_by_vector(
                    ep_embedding, limit=2,
                )
                if episodes:
                    ep_text = _format_episode_footer(episodes)
                    if ep_text:
                        text = (text + "\n\n" + ep_text) if text else ep_text
                        episodes_used = len(episodes)
            except Exception as exc:
                logger.warning("Episode retrieval failed (non-fatal): %s", exc)

        token_estimate = max(1, len(text) // 4)

        logger.info(
            "Retrieval: query=%r → %d primary + %d graph facts "
            "(+%d temporal_dedup, MMR=%s, fmt=%s%s, episodes=%d), ~%d tokens",
            query[:60], len(primary_facts), graph_hit_count, dedup_dropped,
            _MMR_ENABLED, chosen_format,
            "/auto" if self._context_format == "auto" else "",
            episodes_used, token_estimate,
        )

        return RetrievedContext(
            facts=all_facts,
            text=text,
            query=query,
            token_estimate=token_estimate,
            retrieved_from_graph=graph_hit_count,
        )

    async def retrieve_multi(
        self,
        queries: list[str],
        per_query_top_k: int = 3,
        final_top_k: int = 10,
        include_episodes: bool = False,
    ) -> RetrievedContext:
        """Cycle 30 Fix 3: merge-retrieve across multiple sub-queries.

        Runs the normal retrieval pipeline once per sub-query at a small
        per_query_top_k, merges the fact lists (dedup by fact id,
        preserving first-seen order), caps at final_top_k, and re-renders
        the context block. Used by the multi-hop path in main.py._retrieve_context
        when the classifier tags a query as complexity=2.

        Falls back to single-query retrieve() of the first query on any
        empty / degenerate result so complexity=2 queries never return
        empty if a single-query retrieve would have succeeded.
        """
        if not queries:
            return RetrievedContext()

        # First sub-query drives the query label + context format choice.
        primary = queries[0]

        merged: list[dict] = []
        seen: set[str] = set()
        for q in queries:
            ctx = await self.retrieve(q, top_k=per_query_top_k)
            for f in ctx.facts:
                fid = f.get("id")
                if not fid or fid in seen:
                    continue
                seen.add(fid)
                merged.append(f)
            if len(merged) >= final_top_k * 2:
                # Enough coverage — stop doing additional searches.
                break

        if not merged:
            # Safety net: if nothing came back (e.g. all sub-queries
            # missed), fall back to a single-query retrieve on the full
            # sub-query list concatenated.
            return await self.retrieve(primary, top_k=final_top_k)

        merged = merged[:final_top_k]

        # Re-render using the same format dispatch the normal path uses.
        if self._context_format == "auto":
            chosen_format = _pick_format(primary)
        else:
            chosen_format = self._context_format

        if chosen_format == "structured":
            text = _format_context_block_structured(
                merged, max_tokens=_LEGACY_DEFAULT_FORMAT_TOKENS,
            )
        else:
            text = _format_context_block(
                merged, max_tokens=_LEGACY_DEFAULT_FORMAT_TOKENS,
            )

        episodes_used = 0
        if include_episodes and self._embed_fn is not None:
            try:
                ep_embedding = await self._embed_fn(primary)
                episodes = self._store.search_episodes_by_vector(
                    ep_embedding, limit=2,
                )
                if episodes:
                    ep_text = _format_episode_footer(episodes)
                    if ep_text:
                        text = (text + "\n\n" + ep_text) if text else ep_text
                        episodes_used = len(episodes)
            except Exception as exc:
                logger.warning("Episode retrieval failed (non-fatal): %s", exc)

        token_estimate = max(1, len(text) // 4)
        logger.info(
            "Multi-retrieval: %d sub-queries -> %d unique facts (fmt=%s, episodes=%d), ~%d tokens",
            len(queries), len(merged), chosen_format, episodes_used, token_estimate,
        )
        return RetrievedContext(
            facts=merged,
            text=text,
            query=primary,
            token_estimate=token_estimate,
            retrieved_from_graph=0,  # not tracked across sub-queries
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fallback_recent_facts(self, limit: int) -> list[dict]:
        """Return the most recently added current facts (no embeddings needed)."""
        try:
            rows = self._store.conn.execute("""
                SELECT id, content, confidence, fact_type, status, created_at
                FROM facts
                WHERE status IN ('current', 'provisional')
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [
                {
                    "id": r[0], "content": r[1], "confidence": r[2],
                    "fact_type": r[3], "status": r[4], "created_at": r[5],
                    "embedding": None, "distance": None,
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("Fallback recent facts failed: %s", exc)
            return []

    def _get_entity_ids_for_fact(self, fact_id: str) -> list[str]:
        """Return entity IDs linked to a fact via the entity_ids JSON column."""
        try:
            row = self._store.conn.execute(
                "SELECT entity_ids FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()
            if row and row[0]:
                import json
                ids = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                return ids if isinstance(ids, list) else []
            return []
        except Exception:
            return []

    def _get_facts_for_entity(self, entity_id: str) -> list[dict]:
        """Return facts linked to an entity via the entity_ids JSON column."""
        try:
            # Use LIKE pre-filter to avoid full table scan
            status_filter = "AND status IN ('current', 'provisional')" if self._temporal_versioning else ""
            rows = self._store.conn.execute(f"""
                SELECT id, content, confidence, fact_type, status, created_at, entity_ids
                FROM facts
                WHERE entity_ids LIKE ?
                  {status_filter}
            """, (f"%{entity_id}%",)).fetchall()
            import json
            result = []
            for r in rows:
                eids = json.loads(r[6]) if isinstance(r[6], str) else (r[6] or [])
                if entity_id in eids:  # precise check after LIKE pre-filter
                    result.append({
                        "id": r[0], "content": r[1], "confidence": r[2],
                        "fact_type": r[3], "status": r[4], "created_at": r[5],
                        "embedding": None, "distance": None,
                    })
            return result
        except Exception as exc:
            logger.warning("Facts for entity failed: %s", exc)
            return []


def _format_context_block(
    facts: list[dict],
    max_tokens: int = _LEGACY_DEFAULT_FORMAT_TOKENS,
) -> str:
    """Format a list of facts into a context block bounded by max_tokens.

    Truncation is at whole-fact boundaries (never mid-sentence). The
    token→character heuristic is the same len()//4 used elsewhere.
    """
    if not facts:
        return ""

    char_budget = max(1, max_tokens * 4)
    header = "## Recalled context"
    lines: list[str] = [header]
    total_chars = len(header) + 1  # +1 for newline
    for f in facts:
        content = f.get("content", "").strip()
        if not content:
            continue
        conf = f.get("confidence") or 0.75
        # Only annotate low-confidence facts (avoid noise for high-confidence ones)
        if conf < 0.5:
            line = f"- {content} (low confidence)"
        else:
            line = f"- {content}"
        # Always accept the first fact line regardless of budget so an
        # extremely small cap still produces output.
        if len(lines) > 1 and (total_chars + len(line) + 1) > char_budget:
            break
        lines.append(line)
        total_chars += len(line) + 1  # +1 for newline

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


def _format_episode_footer(episodes: list[dict]) -> str:
    """Phase-3 Fix 2: compact footer listing prior-turn summaries.

    One line per episode. Summaries already capped at ~500 chars by the
    writer; this caps at 240 chars/line so two episodes stay ~120
    tokens total.
    """
    if not episodes:
        return ""
    lines = ["## Recalled episodes"]
    for ep in episodes[:2]:
        summary = (ep.get("summary") or "").strip()
        if not summary:
            continue
        # Line-cap: prefer cutting at the fact-bullet marker boundary.
        if len(summary) > 240:
            summary = summary[:237] + "..."
        lines.append(f"- {summary}")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


# ─── Cycle 12 — structured context formatter ──────────────────────────────

_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Family", ("wife", "husband", "partner", "spouse", "married", "separated",
                "divorced", "engaged", "son", "daughter", "child", "children",
                "kid", "kids", "twin", "twins", "mother", "father", "mom", "dad",
                "parent", "brother", "sister", "sibling", "family")),
    ("Identity", ("named", "age", "years old", "birthday", "born", "gender",
                  "pronouns", "nationality", "ethnicity")),
    ("Career", ("job", "work", "career", "employer", "company", "role", "title",
                "cto", "ceo", "vp", "manager", "engineer", "promoted", "hired",
                "fired", "resigned", "salary", "colleague", "office", "startup",
                "profession", "occupation")),
    ("Residence", ("lives", "living", "home", "house", "apartment", "condo",
                   "rent", "mortgage", "address", "neighborhood", "district",
                   "city", "moved", "relocat")),
    ("Health", ("health", "anxiety", "depression", "therapy", "therapist",
                "doctor", "medication", "diagnosed", "illness", "exercise",
                "running", "gym", "yoga", "sleep", "diet")),
    ("Interests", ("hobby", "hobbies", "interest", "enjoys", "loves", "likes",
                   "dislikes", "hates", "reading", "book", "movie", "film",
                   "music", "game", "pottery", "hiking", "travel", "cooking",
                   "favorite")),
    ("Languages", ("language", "speaks", "fluent", "english", "mandarin",
                   "spanish", "french", "german", "japanese", "chinese")),
    ("Connections", ("friend", "best friend", "colleague", "acquaintance",
                     "mentor", "neighbor")),
    ("Finances", ("money", "salary", "savings", "investment", "stock", "loan",
                  "debt", "budget", "$", "dollar", "net worth", "mba", "tuition",
                  "tesla", "car", "vehicle")),
    ("Education", ("school", "university", "college", "degree", "mba", "phd",
                   "studied", "graduated", "student")),
    ("Opinions", ("believes", "thinks", "opinion", "views", "political",
                  "religious")),
]

_DEFAULT_CATEGORY = "Other"
_CATEGORY_ORDER = [name for name, _ in _CATEGORY_KEYWORDS] + [_DEFAULT_CATEGORY]


def _categorise_fact(content: str) -> str:
    c = content.lower()
    for name, keywords in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in c:
                return name
    return _DEFAULT_CATEGORY


_STRUCTURED_HEADER = (
    "[User context from memory — use only these facts when answering personal questions]"
)


def _format_context_block_structured(
    facts: list[dict],
    max_tokens: int = _LEGACY_DEFAULT_FORMAT_TOKENS,
) -> str:
    """Group facts by category, order by recency, annotate supersession.

    Telegraph style. Empty categories are dropped entirely (empty headers
    invite hallucination). Single-fact categories render inline on the same
    line as the category label so the model doesn't see a header as an
    invitation to elaborate. Header + footer sentinels bracket the block
    to make "don't go beyond this" unambiguous.

    Truncation is at whole-category-line boundaries to stay under max_tokens.
    """
    if not facts:
        return ""

    buckets: dict[str, list[dict]] = {}
    for f in facts:
        content = (f.get("content") or "").strip()
        if not content:
            continue
        cat = _categorise_fact(content)
        buckets.setdefault(cat, []).append(f)

    if not buckets:
        return ""

    # Recency sort within each bucket
    for items in buckets.values():
        items.sort(key=lambda f: f.get("created_at") or "", reverse=True)

    char_budget = max(1, max_tokens * 4)
    lines: list[str] = [_STRUCTURED_HEADER]
    total_chars = len(_STRUCTURED_HEADER) + 1
    wrote_any = False
    for cat in _CATEGORY_ORDER:
        items = buckets.get(cat)
        if not items:
            continue  # drop empty categories entirely
        parts: list[str] = []
        for f in items:
            content = (f.get("content") or "").strip().rstrip(".")
            if not content:
                continue
            conf = f.get("confidence") or 0.75
            status = f.get("status") or "current"
            marker = ""
            if status == "superseded":
                marker = " (previously)"
            elif status == "provisional":
                marker = " (provisional)"
            elif conf < 0.5:
                marker = " (low confidence)"
            parts.append(content + marker)
        if not parts:
            continue
        # Inline single-fact and multi-fact rendering share the same shape;
        # no blank line before the category — keep it tight.
        # Truncate parts within the category if needed to stay under budget.
        cat_prefix = f"{cat}: "
        # Always include at least one part; then add more while budget allows.
        truncated_parts: list[str] = [parts[0]]
        running = len(cat_prefix) + len(parts[0])
        for p in parts[1:]:
            added = len("; ") + len(p)
            if (total_chars + running + added + 1) > char_budget:
                break
            truncated_parts.append(p)
            running += added
        cat_line = cat_prefix + "; ".join(truncated_parts)
        # Always include the first category line regardless of budget.
        if wrote_any and (total_chars + len(cat_line) + 1) > char_budget:
            break
        lines.append(cat_line)
        total_chars += len(cat_line) + 1
        wrote_any = True

    if not wrote_any:
        return ""
    return "\n".join(lines)


# ─── Cycle 9 helpers — dedup + MMR diversity re-ranking ────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 on empty, zero-magnitude, or length mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def _load_content_embeddings(store: Any, fact_ids: list[str]) -> dict[str, list[float]]:
    """Batch-load stored content embeddings for the given fact IDs.

    Returns a dict keyed by fact_id. Facts with NULL embeddings or that do not
    exist are silently omitted. Dimension is inferred from blob length
    (float32 = 4 bytes per element).
    """
    if not fact_ids:
        return {}
    placeholders = ",".join("?" for _ in fact_ids)
    try:
        rows = store.conn.execute(
            f"SELECT id, embedding FROM facts WHERE id IN ({placeholders})",
            tuple(fact_ids),
        ).fetchall()
    except Exception as exc:
        logger.warning("Content-embedding load failed: %s", exc)
        return {}
    out: dict[str, list[float]] = {}
    for fid, blob in rows:
        if blob is None:
            continue
        try:
            dim = len(blob) // 4
            if dim <= 0:
                continue
            out[fid] = deserialize_float32(blob, dim)
        except Exception as exc:
            logger.warning("Deserialize failed for fact %s: %s", fid, exc)
    return out


def _dedup_by_content(
    facts: list[dict],
    embeddings: dict[str, list[float]],
    threshold: float,
    cap: int,
) -> list[dict]:
    """Keep facts whose content is not near-duplicate of an already-kept fact.

    Assumes *facts* is pre-sorted (best first — typically by vector distance).
    A fact with no stored embedding is always kept (no basis for rejection).
    Stops after *cap* kept facts.
    """
    kept: list[dict] = []
    kept_vecs: list[list[float]] = []
    for f in facts:
        if len(kept) >= cap:
            break
        emb = embeddings.get(f["id"])
        if emb is None:
            kept.append(f)
            continue
        is_dup = False
        for kv in kept_vecs:
            if _cosine(emb, kv) >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(f)
            kept_vecs.append(emb)
    return kept


def _mmr_rerank(
    facts: list[dict],
    embeddings: dict[str, list[float]],
    lam: float,
    k: int,
) -> list[dict]:
    """Greedy Maximal Marginal Relevance selection.

    Relevance is normalised from vector distance into [0.3, 1.0] (lower
    distance → higher relevance). The 0.3 floor ensures the worst-ranked
    primary is still competitive with graph-hop facts and can lose to a
    more diverse alternative, but the best primary remains clearly
    preferred. Facts without a distance (graph-hop) get a fixed 0.3
    baseline. Confidence adds a gentle ±0.05 bias around 0.75. Redundancy
    is the max cosine similarity to any already-selected fact.
    """
    if not facts or k <= 0:
        return []

    # Floor chosen so that at lam=0.7, diversity can actually overcome
    # relevance for near-duplicates. Math: need (1-lam)*1 > lam*(1-floor),
    # i.e. floor > 2 - 1/lam. At lam=0.7 that's floor > 0.571; 0.6 gives
    # a small margin. Also keeps graph-hop relevance (fixed 0.3 historical
    # baseline) well below primary.
    _REL_FLOOR = 0.6
    primary_dists = [f["distance"] for f in facts if f.get("distance") is not None]
    if primary_dists:
        d_min = min(primary_dists)
        d_max = max(primary_dists)
    else:
        d_min = d_max = 0.0
    d_range = (d_max - d_min) if (d_max - d_min) > 1e-9 else 1.0

    _GRAPH_HOP_REL = 0.3

    def _relevance(f: dict) -> float:
        d = f.get("distance")
        if d is None:
            base = _GRAPH_HOP_REL
        else:
            # Map [d_min, d_max] → [1.0, _REL_FLOOR]
            base = _REL_FLOOR + (1.0 - _REL_FLOOR) * (d_max - d) / d_range
        conf = f.get("confidence") or 0.75
        return max(0.0, min(1.0, base + 0.05 * (conf - 0.75)))

    rel = {f["id"]: _relevance(f) for f in facts}

    remaining = list(facts)
    selected: list[dict] = []
    selected_vecs: list[list[float]] = []

    while remaining and len(selected) < k:
        best = None
        best_score = -1e9
        for f in remaining:
            emb = embeddings.get(f["id"])
            if emb is None or not selected_vecs:
                redundancy = 0.0
            else:
                redundancy = max(_cosine(emb, sv) for sv in selected_vecs)
            score = lam * rel[f["id"]] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score = score
                best = f
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        emb = embeddings.get(best["id"])
        if emb is not None:
            selected_vecs.append(emb)
    return selected


# ── Cycle 26 Fix 2: post-retrieval temporal dedup ─────────────────────────────
def _temporal_dedup(
    facts: list[dict],
    content_embs: dict[str, list[float]],
    entity_ids_by_fact: dict[str, set[str]],
    similarity_threshold: float = _TEMPORAL_DEDUP_SIMILARITY,
) -> list[dict]:
    """Collapse facts that are semantically near-identical about the same
    entity at different points in time. Keep the one with the latest
    `created_at` (ties: higher `confidence`, then stable by `id`).

    Two facts cluster iff BOTH conditions hold:
      1. they share at least one entity_id
      2. cosine similarity of their content embeddings >= threshold

    Either alone is wrong: similarity alone would collapse "Mary in Boston"
    with "Tom in Boston" (different subjects); entity overlap alone would
    collapse "Mary's condo in Beacon Hill" with "Mary's cabin in Vermont"
    (different places). The AND gate targets exactly "two facts about the
    same entity saying near-the-same thing at different times."

    Input is small (<=10 facts). O(n^2) single-linkage clustering is fine.
    """
    n = len(facts)
    if n <= 1:
        return list(facts)

    # Union-find
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        fi = facts[i]
        ei = entity_ids_by_fact.get(fi["id"], set())
        emb_i = content_embs.get(fi["id"])
        if emb_i is None or not ei:
            continue
        for j in range(i + 1, n):
            fj = facts[j]
            ej = entity_ids_by_fact.get(fj["id"], set())
            emb_j = content_embs.get(fj["id"])
            if emb_j is None or not ej:
                continue
            if not (ei & ej):
                continue
            if _cosine(emb_i, emb_j) >= similarity_threshold:
                union(i, j)

    # Group indices by cluster root
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    # Pick the winner per cluster: latest created_at, then highest confidence,
    # then lowest id (stable).
    def _pick_winner(indices: list[int]) -> dict:
        best = facts[indices[0]]
        for idx in indices[1:]:
            cand = facts[idx]
            best_ts = str(best.get("created_at") or "")
            cand_ts = str(cand.get("created_at") or "")
            if cand_ts > best_ts:
                best = cand
                continue
            if cand_ts < best_ts:
                continue
            # Timestamps tied → confidence
            best_conf = float(best.get("confidence") or 0.0)
            cand_conf = float(cand.get("confidence") or 0.0)
            if cand_conf > best_conf:
                best = cand
                continue
            if cand_conf < best_conf:
                continue
            # Confidence tied → stable by id (lexicographic smallest wins)
            if (cand.get("id") or "") < (best.get("id") or ""):
                best = cand
        return best

    winners = [_pick_winner(idxs) for idxs in clusters.values()]

    # Preserve original relative order by first-appearance index.
    order_by_id = {f["id"]: i for i, f in enumerate(facts)}
    winners.sort(key=lambda f: order_by_id[f["id"]])
    return winners

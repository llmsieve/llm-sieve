"""Learning & Feedback Loop — observation-based classifier/retrieval tuning.

Four feedback signals, collected per-interaction:
  1. Retrieval Relevance  — cosine(response embedding, pre-populated facts) > threshold → "used"
  2. Recall Tool Calls    — each LLM recall call = classifier miss signal
  3. Fact Confirmation    — usage_rate = usage_count / retrieval_count per fact
  4. Query Patterns       — temporal/topical query trends (weakest signal)

Tuning loop (every ~50 interactions):
  - Recalculate per-fact usage_rate
  - Identify core facts (top N by usage_rate)
  - Flag over-retrieved facts (usage_rate < 0.1)
  - Persist updated weights in preferences table
  - Pure arithmetic — no LLM needed

Usage::

    loop = LearningLoop(store, embed_fn, config.learning)
    # After each interaction:
    await loop.record_interaction(user_query, response_text, pre_populated_facts, recall_rounds)
    # Tuning runs automatically when interaction_count % tune_interval == 0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("recall.learning")


# ─── Signal 1: Retrieval Relevance ───────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def score_retrieval_relevance(
    response_text: str,
    pre_populated_facts: list[dict],
    embed_fn: Any,
    threshold: float = 0.7,
) -> list[dict]:
    """Compare response embedding against each pre-populated fact.

    Returns a list of {fact_id, similarity, used} dicts.
    """
    if not response_text or not pre_populated_facts or embed_fn is None:
        return []

    try:
        response_emb = await embed_fn(response_text)
    except Exception as exc:
        logger.warning("Relevance scoring embed failed: %s", exc)
        return []

    results = []
    for fact in pre_populated_facts:
        fact_id = fact.get("id")
        if not fact_id:
            continue

        # Get fact embedding — need to embed the content if not stored as vector
        try:
            fact_emb = await embed_fn(fact.get("content", ""))
        except Exception:
            continue

        sim = _cosine_similarity(response_emb, fact_emb)
        results.append({
            "fact_id": fact_id,
            "similarity": sim,
            "used": sim >= threshold,
        })

    return results


# ─── Signal 2: Recall Tool Call Tracking ──────────────────────────────────────

@dataclass
class RecallCallSignal:
    """Records a recall tool call as a classifier training signal."""
    user_query: str
    recall_query: str
    round_number: int


# ─── Signal 3: Fact Confirmation / Usage Rate ────────────────────────────────

@dataclass
class FactUsageStats:
    """Per-fact usage statistics."""
    fact_id: str
    content: str
    retrieval_count: int
    usage_count: int
    usage_rate: float        # usage_count / retrieval_count (0 if never retrieved)
    tier: str                # "core" | "normal" | "over_retrieved" | "cold"


def compute_fact_usage_stats(facts: list[dict]) -> list[FactUsageStats]:
    """Compute usage_rate and tier classification for all facts."""
    results = []
    for f in facts:
        retrieval_count = f.get("retrieval_count", 0)
        usage_count = f.get("usage_count", 0)

        if retrieval_count > 0:
            usage_rate = usage_count / retrieval_count
        else:
            usage_rate = 0.0

        # Tier classification
        if usage_rate >= 0.8 and retrieval_count >= 3:
            tier = "core"
        elif retrieval_count >= 5 and usage_rate < 0.1:
            tier = "over_retrieved"
        elif retrieval_count == 0 and usage_count == 0:
            tier = "cold"
        else:
            tier = "normal"

        results.append(FactUsageStats(
            fact_id=f["id"],
            content=f.get("content", ""),
            retrieval_count=retrieval_count,
            usage_count=usage_count,
            usage_rate=usage_rate,
            tier=tier,
        ))

    return results


# ─── Signal 4: Query Pattern Tracking ────────────────────────────────────────

def classify_query_category(query: str) -> str:
    """Classify a query into a broad category for pattern tracking."""
    query_lower = query.lower()

    categories = [
        ("location", ["where", "live", "city", "country", "address", "based"]),
        ("identity", ["who am i", "my name", "about me"]),
        ("family", ["family", "partner", "wife", "husband", "child", "kid", "parent", "mother", "father"]),
        ("work", ["work", "job", "career", "employer", "occupation", "salary", "earn"]),
        ("finance", ["money", "salary", "earn", "income", "net worth", "savings", "budget", "refinance"]),
        ("health", ["health", "doctor", "medical", "allergy", "medication"]),
        ("preference", ["like", "prefer", "favourite", "favorite", "enjoy", "love", "hate"]),
        ("temporal", ["when", "how long", "last time", "recently", "yesterday", "schedule"]),
    ]

    for cat, keywords in categories:
        if any(kw in query_lower for kw in keywords):
            return cat

    return "general"


# ─── Tuning Loop ──────────────────────────────────────────────────────────────

@dataclass
class TuningResult:
    """Result of a tuning loop run."""
    interaction_count: int
    total_facts: int
    core_facts: int
    over_retrieved_facts: int
    cold_facts: int
    avg_usage_rate: float
    avg_recall_calls_per_query: float
    category_distribution: dict[str, int] = field(default_factory=dict)


class LearningLoop:
    """Observation-based learning loop for classifier/retrieval tuning.

    Collects signals per-interaction and runs a tuning pass every tune_interval
    interactions. Pure arithmetic — no LLM involved.
    """

    def __init__(
        self,
        store: Any,
        embed_fn: Any | None = None,
        config: Any | None = None,
    ):
        self._store = store
        self._embed_fn = embed_fn

        # Config defaults
        if config is not None:
            self._tune_interval = config.tune_interval
            self._relevance_threshold = config.relevance_threshold
            self._core_facts_size = config.core_facts_size
        else:
            self._tune_interval = 50
            self._relevance_threshold = 0.7
            self._core_facts_size = 30

        # In-memory accumulators (reset each tuning cycle)
        self._recall_calls_total = 0
        self._interactions_since_tune = 0
        self._query_categories: list[str] = []

    async def record_interaction(
        self,
        user_query: str,
        response_text: str = "",
        pre_populated_facts: list[dict] | None = None,
        recall_rounds: int = 0,
    ) -> TuningResult | None:
        """Record a single interaction and its feedback signals.

        Returns a TuningResult if a tuning pass was triggered, else None.
        """
        if self._store._conn is None:
            return None

        # Log interaction
        self._store.log_interaction()
        self._interactions_since_tune += 1

        # Signal 1: Retrieval relevance
        if response_text and pre_populated_facts:
            relevance_scores = await score_retrieval_relevance(
                response_text, pre_populated_facts,
                self._embed_fn, self._relevance_threshold,
            )
            # Update usage_count on used facts
            for score in relevance_scores:
                if score["used"]:
                    try:
                        self._store.conn.execute(
                            "UPDATE facts SET usage_count = usage_count + 1 WHERE id = ?",
                            (score["fact_id"],),
                        )
                    except Exception:
                        pass
            if relevance_scores:
                self._store.conn.commit()

        # Signal 2: Recall tool calls
        self._recall_calls_total += recall_rounds

        # Signal 3: Fact confirmation is implicit (updated via usage_count above)

        # Signal 4: Query patterns
        if user_query:
            category = classify_query_category(user_query)
            self._query_categories.append(category)
            try:
                self._store.upsert_preference(
                    category="query_pattern",
                    content=category,
                    strength=0.5,
                )
            except Exception:
                pass

        # Check if tuning should run
        interaction_count = self._store.get_interaction_count()
        if self._interactions_since_tune >= self._tune_interval:
            return self._run_tuning(interaction_count)

        return None

    def _run_tuning(self, interaction_count: int) -> TuningResult:
        """Execute the tuning loop. Pure arithmetic — no LLM."""
        logger.info("Tuning loop triggered at interaction %d", interaction_count)

        # Get all facts with usage stats
        all_facts = self._store.get_all_facts_with_usage()
        usage_stats = compute_fact_usage_stats(all_facts)

        # Classify into tiers
        core = [s for s in usage_stats if s.tier == "core"]
        over_retrieved = [s for s in usage_stats if s.tier == "over_retrieved"]
        cold = [s for s in usage_stats if s.tier == "cold"]

        # Compute averages
        usage_rates = [s.usage_rate for s in usage_stats if s.retrieval_count > 0]
        avg_usage_rate = sum(usage_rates) / len(usage_rates) if usage_rates else 0.0

        avg_recall_per_query = (
            self._recall_calls_total / self._interactions_since_tune
            if self._interactions_since_tune > 0
            else 0.0
        )

        # Query category distribution
        cat_dist: dict[str, int] = {}
        for cat in self._query_categories:
            cat_dist[cat] = cat_dist.get(cat, 0) + 1

        # Persist core facts tier to preferences
        core_sorted = sorted(usage_stats, key=lambda s: s.usage_rate, reverse=True)
        core_ids = [s.fact_id for s in core_sorted[:self._core_facts_size]]
        try:
            self._store.upsert_preference(
                category="core_facts",
                content=",".join(core_ids),
                strength=avg_usage_rate,
            )
        except Exception as exc:
            logger.warning("Failed to persist core facts: %s", exc)

        # Persist metrics
        try:
            self._store.upsert_preference(
                category="learning_metrics",
                content=f"avg_recall_calls={avg_recall_per_query:.2f}",
                strength=avg_recall_per_query,
            )
            self._store.upsert_preference(
                category="learning_metrics",
                content=f"avg_usage_rate={avg_usage_rate:.2f}",
                strength=avg_usage_rate,
            )
        except Exception as exc:
            logger.warning("Failed to persist learning metrics: %s", exc)

        result = TuningResult(
            interaction_count=interaction_count,
            total_facts=len(usage_stats),
            core_facts=len(core),
            over_retrieved_facts=len(over_retrieved),
            cold_facts=len(cold),
            avg_usage_rate=avg_usage_rate,
            avg_recall_calls_per_query=avg_recall_per_query,
            category_distribution=cat_dist,
        )

        logger.info(
            "Tuning complete: %d facts (%d core, %d over-retrieved, %d cold), "
            "avg_usage_rate=%.2f, avg_recall_calls=%.2f/query",
            result.total_facts, result.core_facts,
            result.over_retrieved_facts, result.cold_facts,
            result.avg_usage_rate, result.avg_recall_calls_per_query,
        )

        # Reset accumulators
        self._recall_calls_total = 0
        self._interactions_since_tune = 0
        self._query_categories = []

        return result

    def get_metrics(self) -> dict[str, Any]:
        """Return current learning metrics without running a full tuning pass."""
        if self._store._conn is None:
            return {}

        all_facts = self._store.get_all_facts_with_usage()
        usage_stats = compute_fact_usage_stats(all_facts)

        core = [s for s in usage_stats if s.tier == "core"]
        over_retrieved = [s for s in usage_stats if s.tier == "over_retrieved"]
        cold = [s for s in usage_stats if s.tier == "cold"]

        usage_rates = [s.usage_rate for s in usage_stats if s.retrieval_count > 0]

        return {
            "total_facts": len(usage_stats),
            "core_facts": len(core),
            "over_retrieved_facts": len(over_retrieved),
            "cold_facts": len(cold),
            "avg_usage_rate": sum(usage_rates) / len(usage_rates) if usage_rates else 0.0,
            "interactions_since_tune": self._interactions_since_tune,
            "tune_interval": self._tune_interval,
            "interaction_count": self._store.get_interaction_count(),
        }

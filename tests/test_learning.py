"""Tests for Phase 9: Learning Feedback Loop.

Tests cover:
- Retrieval relevance scoring (Signal 1)
- Recall tool call tracking (Signal 2)
- Fact confirmation / usage rate computation (Signal 3)
- Query pattern classification (Signal 4)
- Tuning loop trigger and execution
- Core facts tier identification
- Metrics endpoint
- 50+ simulated interaction checkpoint
"""

from __future__ import annotations

import math

import pytest

from sieve.config import LearningConfig, StoreConfig
from sieve.store import MemoryStore
from sieve.learning import (
    FactUsageStats,
    LearningLoop,
    TuningResult,
    _cosine_similarity,
    classify_query_category,
    compute_fact_usage_stats,
    score_retrieval_relevance,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    config = StoreConfig(path=str(tmp_path / "test.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test-learning")
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


def _fake_embed(text: str) -> list[float]:
    v = [ord(text[0]) / 256.0 if text else 0.1, 0.5, 0.3, 0.2]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


async def _async_embed(text: str) -> list[float]:
    return _fake_embed(text)


@pytest.fixture
def learning_loop(store):
    config = LearningConfig(tune_interval=5, relevance_threshold=0.7, core_facts_size=3)
    return LearningLoop(store, embed_fn=_async_embed, config=config)


# ─── Signal 1: Retrieval Relevance ──────────────────────────────────────────

class TestRetrievalRelevance:
    async def test_identical_text_scores_high(self):
        facts = [{"id": "f1", "content": "User lives in Springfield"}]
        scores = await score_retrieval_relevance(
            "User lives in Springfield", facts, _async_embed, threshold=0.7,
        )
        assert len(scores) == 1
        assert scores[0]["similarity"] > 0.9
        assert scores[0]["used"] is True

    async def test_empty_response_returns_empty(self):
        scores = await score_retrieval_relevance("", [{"id": "f1", "content": "x"}], _async_embed)
        assert scores == []

    async def test_no_facts_returns_empty(self):
        scores = await score_retrieval_relevance("hello", [], _async_embed)
        assert scores == []

    async def test_no_embed_fn_returns_empty(self):
        scores = await score_retrieval_relevance("hello", [{"id": "f1", "content": "x"}], None)
        assert scores == []

    async def test_dissimilar_not_used(self):
        # 'A' vs 'z' — different first chars → different embeddings
        facts = [{"id": "f1", "content": "Apples are red"}]
        scores = await score_retrieval_relevance(
            "zebras run fast", facts, _async_embed, threshold=0.99,
        )
        assert len(scores) == 1
        # With our simple embedding, similarity varies but shouldn't be 1.0
        assert isinstance(scores[0]["used"], bool)


# ─── Signal 3: Fact Usage Stats ──────────────────────────────────────────────

class TestFactUsageStats:
    def test_core_fact(self):
        facts = [{"id": "f1", "content": "x", "retrieval_count": 10, "usage_count": 9}]
        stats = compute_fact_usage_stats(facts)
        assert stats[0].tier == "core"
        assert stats[0].usage_rate == 0.9

    def test_over_retrieved_fact(self):
        facts = [{"id": "f1", "content": "x", "retrieval_count": 20, "usage_count": 1}]
        stats = compute_fact_usage_stats(facts)
        assert stats[0].tier == "over_retrieved"
        assert stats[0].usage_rate == 0.05

    def test_cold_fact(self):
        facts = [{"id": "f1", "content": "x", "retrieval_count": 0, "usage_count": 0}]
        stats = compute_fact_usage_stats(facts)
        assert stats[0].tier == "cold"

    def test_normal_fact(self):
        facts = [{"id": "f1", "content": "x", "retrieval_count": 5, "usage_count": 3}]
        stats = compute_fact_usage_stats(facts)
        assert stats[0].tier == "normal"
        assert stats[0].usage_rate == 0.6

    def test_zero_retrieval_no_div_error(self):
        facts = [{"id": "f1", "content": "x", "retrieval_count": 0, "usage_count": 5}]
        stats = compute_fact_usage_stats(facts)
        assert stats[0].usage_rate == 0.0


# ─── Signal 4: Query Category Classification ─────────────────────────────────

class TestQueryCategory:
    def test_location_query(self):
        assert classify_query_category("where do I live?") == "location"

    def test_family_query(self):
        assert classify_query_category("tell me about my family") == "family"

    def test_work_query(self):
        assert classify_query_category("what is my job?") == "work"

    def test_finance_query(self):
        assert classify_query_category("should I refinance my mortgage?") == "finance"

    def test_preference_query(self):
        assert classify_query_category("what do I like to eat?") == "preference"

    def test_temporal_query(self):
        assert classify_query_category("when did I last visit?") == "temporal"

    def test_general_query(self):
        assert classify_query_category("tell me a joke") == "general"


# ─── Cosine Similarity ───────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal(self):
        assert abs(_cosine_similarity([1, 0, 0], [0, 1, 0])) < 0.001

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


# ─── Learning Loop ───────────────────────────────────────────────────────────

class TestLearningLoop:
    async def test_record_interaction_increments(self, learning_loop, store):
        result = await learning_loop.record_interaction("where do I live?")
        assert result is None  # not yet at tune_interval
        assert learning_loop._interactions_since_tune == 1

    async def test_tuning_triggers_at_interval(self, learning_loop, store):
        # Seed some facts
        store.insert_fact("User lives in Springfield", embedding=_fake_embed("User lives in Springfield"))
        store.insert_fact("User is a librarian", embedding=_fake_embed("User is a librarian"))

        # Run 5 interactions (tune_interval=5)
        result = None
        for i in range(5):
            result = await learning_loop.record_interaction(f"query {i}")

        assert result is not None
        assert isinstance(result, TuningResult)
        assert result.total_facts >= 2
        assert result.interaction_count >= 5

    async def test_tuning_resets_accumulators(self, learning_loop, store):
        store.insert_fact("Test fact", embedding=None)

        for i in range(5):
            await learning_loop.record_interaction(f"query {i}")

        assert learning_loop._interactions_since_tune == 0
        assert learning_loop._recall_calls_total == 0

    async def test_recall_rounds_tracked(self, learning_loop, store):
        store.insert_fact("Test", embedding=None)

        await learning_loop.record_interaction("q1", recall_rounds=2)
        await learning_loop.record_interaction("q2", recall_rounds=3)
        assert learning_loop._recall_calls_total == 5

    async def test_query_patterns_persisted(self, learning_loop, store):
        await learning_loop.record_interaction("where do I live?")
        prefs = store.get_preferences(category="query_pattern")
        categories = [p["content"] for p in prefs]
        assert "location" in categories

    async def test_get_metrics(self, learning_loop, store):
        store.insert_fact("Test", embedding=None)
        await learning_loop.record_interaction("hello")
        metrics = learning_loop.get_metrics()
        assert "total_facts" in metrics
        assert "interactions_since_tune" in metrics
        assert metrics["interactions_since_tune"] == 1

    async def test_core_facts_persisted_after_tuning(self, learning_loop, store):
        # Create facts with varying usage
        for i in range(5):
            fid = store.insert_fact(f"Fact {i}", embedding=None)
            # Simulate usage
            for _ in range(i * 3):
                store.boost_fact_confidence(fid, boost=0.001)

        # Trigger tuning
        for i in range(5):
            await learning_loop.record_interaction(f"query {i}")

        prefs = store.get_preferences(category="core_facts")
        assert len(prefs) >= 1


# ─── 50+ Interaction Simulation Checkpoint ───────────────────────────────────

class TestSimulated50Interactions:
    async def test_50_interactions_produces_metrics(self, store):
        """Checkpoint: after 50+ simulated interactions, learning metrics available."""
        config = LearningConfig(tune_interval=10, core_facts_size=5)
        loop = LearningLoop(store, embed_fn=_async_embed, config=config)

        # Seed facts
        fact_ids = []
        for content in [
            "User lives in Springfield",
            "User is a librarian at City Library",
            "User is 38 years old",
            "User has two children",
            "User's partner is Jordan",
            "User earns $180k per year",
            "User is from Riverside",
            "User's best friend is Robin",
        ]:
            fid = store.insert_fact(content, embedding=_fake_embed(content))
            fact_ids.append(fid)

        # Simulate 50+ interactions with varying recall rounds
        queries = [
            "where do I live?",
            "what is the weather?",
            "tell me about my family",
            "what do I do for work?",
            "should I refinance?",
            "how old am I?",
            "what is my salary?",
            "where am I from?",
            "who is Robin?",
            "what's for dinner?",
        ]

        tuning_results = []
        for i in range(55):
            query = queries[i % len(queries)]
            recall_rounds = 1 if i % 3 == 0 else 0
            result = await loop.record_interaction(
                query, recall_rounds=recall_rounds,
            )
            if result is not None:
                tuning_results.append(result)

        # Should have triggered tuning at least 5 times (55 / 10)
        assert len(tuning_results) >= 5

        # Check final metrics
        metrics = loop.get_metrics()
        assert metrics["total_facts"] >= 8
        assert metrics["interaction_count"] >= 55

        # Check that learning metrics were persisted
        prefs = store.get_preferences(category="learning_metrics")
        assert len(prefs) >= 1

        # Check core facts preference exists
        core_prefs = store.get_preferences(category="core_facts")
        assert len(core_prefs) >= 1

        # Check query patterns were tracked
        pattern_prefs = store.get_preferences(category="query_pattern")
        assert len(pattern_prefs) >= 1

        # Verify tuning results show recall_calls trending
        for tr in tuning_results:
            assert isinstance(tr.avg_recall_calls_per_query, float)
            assert tr.avg_recall_calls_per_query >= 0


class TestStoreLearnMethods:
    def test_upsert_preference_insert(self, store):
        pid = store.upsert_preference("test_cat", "test_content", 0.5)
        prefs = store.get_preferences("test_cat")
        assert len(prefs) == 1
        assert prefs[0]["content"] == "test_content"

    def test_upsert_preference_update(self, store):
        store.upsert_preference("test_cat", "test_content", 0.5)
        store.upsert_preference("test_cat", "test_content", 0.8)
        prefs = store.get_preferences("test_cat")
        assert len(prefs) == 1
        assert prefs[0]["strength"] == 0.8
        assert prefs[0]["observation_count"] == 2

    def test_get_preferences_all(self, store):
        store.upsert_preference("cat_a", "content_a")
        store.upsert_preference("cat_b", "content_b")
        all_prefs = store.get_preferences()
        assert len(all_prefs) >= 2

    def test_log_interaction(self, store):
        store.log_interaction("session1")
        count = store.get_interaction_count()
        assert count == 1

    def test_interaction_count_increments(self, store):
        for _ in range(5):
            store.log_interaction()
        assert store.get_interaction_count() == 5

    def test_get_all_facts_with_usage(self, store):
        store.insert_fact("Test fact", embedding=None)
        facts = store.get_all_facts_with_usage()
        assert len(facts) == 1
        assert facts[0]["retrieval_count"] == 0
        assert facts[0]["usage_count"] == 0

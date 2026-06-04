"""Regression tests for ValidationCollector.snapshot_store.

The collector reads MemoryStore.stats() keys that are suffixed with "_count"
(facts_count, entities_count, relationships_count, known_unknowns_count).
A prior version of the code read the un-suffixed names and silently wrote
zeros into validation_metrics.db — long validation runs were entirely
reported as facts_in_store=0 because of it.
"""

from __future__ import annotations

from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.validation_collector import RequestMetrics, ValidationCollector


def _seed_minimal_store(path: str) -> MemoryStore:
    store = MemoryStore(StoreConfig(path=path))
    store.open()
    store.init_schema()
    entity_id = store.insert_entity("Jamie Rivera", type="person")
    store.insert_fact(
        "User is Jamie Rivera",
        entity_ids=[entity_id],
        source="writer_s1",
        confidence=0.75,
    )
    return store


def test_snapshot_store_reads_count_suffixed_keys(tmp_path):
    store_path = tmp_path / "snapshot.db"
    store = _seed_minimal_store(str(store_path))

    collector = ValidationCollector(
        db_path=tmp_path / "metrics.db", enabled=True,
    )
    metrics = RequestMetrics()
    collector.snapshot_store(metrics, store)

    assert metrics.facts_in_store == 1, (
        "snapshot_store must read 'facts_count' from MemoryStore.stats(); "
        "reading the legacy 'facts' key silently yields zero."
    )
    assert metrics.entities_in_store == 1
    assert metrics.relationships_in_store == 0
    assert metrics.known_unknowns_count == 0
    assert metrics.facts_current == 1
    assert metrics.facts_superseded == 0
    store.close()


def test_stats_returns_count_suffixed_keys(tmp_path):
    """Guard the contract snapshot_store relies on: stats() keys end in _count."""
    store = _seed_minimal_store(str(tmp_path / "stats.db"))
    stats = store.stats()
    for required in (
        "facts_count", "entities_count",
        "relationships_count", "known_unknowns_count",
    ):
        assert required in stats, (
            f"MemoryStore.stats() must expose '{required}'; the validation "
            f"collector depends on this key shape."
        )
    store.close()


# ─── Writer stage-count recording ──────────────────────────────────────────


def test_record_writer_result_copies_stage_counts(tmp_path):
    """record_writer_result must populate metrics with per-stage counters
    from WriteResult. Before this was wired, writer_stage1_facts /
    writer_stage2_facts in validation_metrics.db were always 0 regardless
    of what the writer actually did."""
    from sieve.writer import WriteResult

    collector = ValidationCollector(
        db_path=tmp_path / "metrics.db", enabled=True,
    )
    metrics = RequestMetrics()
    result = WriteResult(
        facts_written=3,
        stage1_facts=2,
        stage2_facts=4,
        stage2_invoked=True,
        conflicts_detected=1,
        supersessions=2,
    )

    collector.record_writer_result(metrics, result)

    assert metrics.writer_stage1_facts == 2
    assert metrics.writer_stage2_facts == 4
    assert metrics.writer_stage2_invoked is True
    assert metrics.writer_conflicts_detected == 1
    assert metrics.writer_supersessions == 2


def test_record_writer_result_noop_on_none_metrics(tmp_path):
    """Must be safe to call when validation is disabled (metrics=None)."""
    from sieve.writer import WriteResult

    collector = ValidationCollector(
        db_path=tmp_path / "metrics.db", enabled=True,
    )
    # Should not raise
    collector.record_writer_result(None, WriteResult(stage1_facts=5))


def test_record_writer_result_noop_on_none_result(tmp_path):
    """Must be safe to call when the writer task didn't complete in time."""
    collector = ValidationCollector(
        db_path=tmp_path / "metrics.db", enabled=True,
    )
    metrics = RequestMetrics()
    collector.record_writer_result(metrics, None)
    # Defaults untouched
    assert metrics.writer_stage1_facts == 0
    assert metrics.writer_stage2_facts == 0
    assert metrics.writer_stage2_invoked is False

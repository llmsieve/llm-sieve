"""Tests for the writer → validation-metrics rendezvous helper.

The writer is fired as an async background task so production requests
are not blocked by fact extraction. For validation runs we still want
the per-request metrics row to carry the writer's stage-level stats.
``await_writer_and_record`` is the rendezvous point: it awaits the
writer task (with a bounded timeout so a slow writer can never wedge
the response-finalise path) and copies stats into metrics via the
collector.

These tests exercise the helper directly without a running proxy.
"""

from __future__ import annotations

import asyncio

import pytest

from sieve.main import _await_writer_and_record
from sieve.validation_collector import RequestMetrics, ValidationCollector
from sieve.writer import WriteResult


@pytest.fixture
def collector(tmp_path):
    return ValidationCollector(db_path=tmp_path / "metrics.db", enabled=True)


async def _completed_task(result):
    """Wrap a value in a completed asyncio.Task."""
    async def _coro():
        return result
    return asyncio.create_task(_coro())


@pytest.mark.asyncio
async def test_awaits_task_and_records_stats(collector):
    """Happy path: task completes quickly, metrics get populated."""
    task = await _completed_task(
        WriteResult(stage1_facts=3, stage2_facts=1, stage2_invoked=True,
                    conflicts_detected=2, supersessions=1)
    )
    metrics = RequestMetrics()

    await _await_writer_and_record(metrics, task, collector, timeout_s=1.0)

    assert metrics.writer_stage1_facts == 3
    assert metrics.writer_stage2_facts == 1
    assert metrics.writer_stage2_invoked is True
    assert metrics.writer_conflicts_detected == 2
    assert metrics.writer_supersessions == 1


@pytest.mark.asyncio
async def test_noop_when_metrics_none(collector):
    """Validation-disabled path: no metrics object means no work."""
    task = await _completed_task(WriteResult(stage1_facts=9))
    # Must not raise
    await _await_writer_and_record(None, task, collector, timeout_s=1.0)
    # Task still awaited so it doesn't leak
    assert task.done()


@pytest.mark.asyncio
async def test_noop_when_task_none(collector):
    """Fire-and-forget path (production): no task was attached."""
    metrics = RequestMetrics()
    await _await_writer_and_record(metrics, None, collector, timeout_s=1.0)
    assert metrics.writer_stage1_facts == 0
    assert metrics.writer_stage2_facts == 0


@pytest.mark.asyncio
async def test_timeout_leaves_metrics_at_defaults(collector):
    """A slow writer must not wedge response finalisation.

    If the task doesn't complete within timeout_s the helper returns
    without recording (metrics stay at their default zeros) — better to
    lose per-stage stats than to block the client's stream from closing.
    """
    async def _slow():
        await asyncio.sleep(5.0)
        return WriteResult(stage1_facts=99)

    task = asyncio.create_task(_slow())
    metrics = RequestMetrics()

    await _await_writer_and_record(metrics, task, collector, timeout_s=0.05)

    assert metrics.writer_stage1_facts == 0, (
        "on timeout the helper should NOT record partial/stale data"
    )
    # Clean up the background task so pytest doesn't warn
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_records_slow_but_under_timeout(collector):
    """Writer that takes real time but finishes under the timeout
    still gets its stats recorded. Guards against accidentally setting
    a timeout too tight for the actual S2/CPU-pinned writer latency."""
    async def _slow_success():
        await asyncio.sleep(0.2)  # realistic-ish fraction of writer latency
        return WriteResult(stage1_facts=7, stage2_facts=3, stage2_invoked=True)

    task = asyncio.create_task(_slow_success())
    metrics = RequestMetrics()

    # Use a generous timeout so sleep completes well under it
    await _await_writer_and_record(metrics, task, collector, timeout_s=5.0)

    assert metrics.writer_stage1_facts == 7
    assert metrics.writer_stage2_facts == 3
    assert metrics.writer_stage2_invoked is True


@pytest.mark.asyncio
async def test_default_timeout_is_generous_enough_for_s2(collector):
    """The default timeout must accommodate a realistic S2 writer run
    (regex + CPU-pinned LLM call + validation). A 5-second sleep
    loosely models this without requiring Ollama.

    If this test fails, tightening the default timeout has broken the
    core use case the helper exists for.
    """
    async def _realistic_s2():
        await asyncio.sleep(5.0)
        return WriteResult(stage1_facts=2, stage2_facts=4, stage2_invoked=True)

    task = asyncio.create_task(_realistic_s2())
    metrics = RequestMetrics()

    # Default timeout — no override. Must be generous enough for S2.
    await _await_writer_and_record(metrics, task, collector)

    assert metrics.writer_stage1_facts == 2, (
        "default timeout is too tight for a realistic S2 writer run"
    )
    assert metrics.writer_stage2_facts == 4


@pytest.mark.asyncio
async def test_swallows_writer_exception(collector):
    """A writer that raised must not propagate out of the rendezvous.

    The writer is best-effort background work. If it crashed, we still
    want to finalise the metrics row (with zeros for stage counters)
    rather than burning the whole response on a writer bug.
    """
    async def _crashing():
        raise RuntimeError("writer blew up")

    task = asyncio.create_task(_crashing())
    metrics = RequestMetrics()

    await _await_writer_and_record(metrics, task, collector, timeout_s=1.0)

    # No exception propagated, defaults preserved
    assert metrics.writer_stage1_facts == 0

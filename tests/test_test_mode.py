"""Tests for sieve.test_mode protocol surface.

Verifies schema mirror + endpoint mounting/unmounting per SIEVE_TEST_MODE.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ─── Schema tests ───────────────────────────────────────────────────────────────


def test_protocol_version_one():
    from sieve.test_mode import PROTOCOL_VERSION
    assert PROTOCOL_VERSION == 1


def test_is_test_mode_enabled():
    from sieve.test_mode import is_test_mode_enabled
    with patch.dict(os.environ, {"SIEVE_TEST_MODE": "on"}, clear=False):
        assert is_test_mode_enabled() is True
    with patch.dict(os.environ, {"SIEVE_TEST_MODE": "1"}, clear=False):
        assert is_test_mode_enabled() is True
    with patch.dict(os.environ, {"SIEVE_TEST_MODE": "TRUE"}, clear=False):
        assert is_test_mode_enabled() is True
    with patch.dict(os.environ, {"SIEVE_TEST_MODE": ""}, clear=False):
        assert is_test_mode_enabled() is False
    with patch.dict(os.environ, {"SIEVE_TEST_MODE": "off"}, clear=False):
        assert is_test_mode_enabled() is False


def test_turn_complete_schema():
    from sieve.test_mode.schemas import TurnComplete, ComponentBreakdown
    event = TurnComplete(
        schema_version=1,
        event_id="evt_001",
        event_type="turn_complete",
        ts_utc="2026-04-26T12:34:56.789Z",
        turn_idx=1,
        sieve_inbound_tokens=17064,
        sieve_outbound_tokens=1875,
        component_breakdown=ComponentBreakdown(sys=555, ctx=18, hist=218, user=4, tools=484),
        latency_ms=14200,
        request_body_hash="a" * 64,
        response_body_hash="b" * 64,
        phase_at_turn="ACCUMULATE",
        facts_in_store_at_turn=49,
    )
    assert event.turn_idx == 1
    # Round trip JSON
    j = event.model_dump_json()
    again = TurnComplete.model_validate_json(j)
    assert again == event


def test_extra_field_rejected():
    from sieve.test_mode.schemas import TurnComplete, ComponentBreakdown
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TurnComplete(
            schema_version=1,
            event_id="evt_001",
            event_type="turn_complete",
            ts_utc="2026-04-26T12:34:56.789Z",
            turn_idx=1,
            sieve_inbound_tokens=17064,
            sieve_outbound_tokens=1875,
            component_breakdown=ComponentBreakdown(sys=555, ctx=18, hist=218, user=4, tools=484),
            latency_ms=14200,
            request_body_hash="a" * 64,
            response_body_hash="b" * 64,
            phase_at_turn="ACCUMULATE",
            facts_in_store_at_turn=49,
            future_field="drift",  # type: ignore[call-arg]
        )


# ─── Event bus tests ────────────────────────────────────────────────────────────


def test_event_bus_emits_and_subscribers_receive():
    import asyncio
    from sieve.test_mode.event_bus import EventBus

    bus = EventBus()
    queue = bus.subscribe()

    bus.emit_phase_change(
        from_phase="ACCUMULATE", to_phase="ACTIVATE",
        fact_count=50, turn_idx=85,
    )

    async def get_one():
        return await asyncio.wait_for(queue.get(), timeout=1.0)

    event = asyncio.run(get_one())
    assert event.event_type == "phase_change"
    assert event.from_phase == "ACCUMULATE"
    assert event.to_phase == "ACTIVATE"


def test_event_bus_buffer_replay_on_reconnect():
    import asyncio
    from sieve.test_mode.event_bus import EventBus

    bus = EventBus()
    # Emit 3 events before any subscriber.
    bus.emit_phase_change(from_phase="OBSERVE", to_phase="ACCUMULATE", fact_count=20, turn_idx=10)
    bus.emit_phase_change(from_phase="ACCUMULATE", to_phase="ACTIVATE", fact_count=50, turn_idx=85)
    bus.emit_store_state(facts_total=64)

    # Subscribe with last_event_id pointing at the first event; we should
    # get the next 2 replayed.
    queue = bus.subscribe(last_event_id="evt_00000001")

    async def drain():
        out = []
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
                out.append(event)
        except asyncio.TimeoutError:
            return out

    events = asyncio.run(drain())
    assert len(events) == 2
    assert events[0].event_type == "phase_change"
    assert events[1].event_type == "store_state"


# ─── Test-mode gate test ────────────────────────────────────────────────────────


def test_test_mode_off_means_no_router(monkeypatch):
    """When SIEVE_TEST_MODE is unset, /test/* must not be mounted.

    We can't easily import + boot the full FastAPI app in a unit test, so
    we just verify the gating helper. End-to-end gate is tested via the
    shell-based smoke (commit message references /tmp/test_tm_prod.sh).
    """
    monkeypatch.delenv("SIEVE_TEST_MODE", raising=False)
    from sieve.test_mode import is_test_mode_enabled
    assert is_test_mode_enabled() is False

"""In-process event bus for sieve test-mode telemetry.

A single ``EventBus`` instance lives on the FastAPI app state when
``SIEVE_TEST_MODE=on``. It maintains:

- A monotonic event counter (assigns ``event_id`` strings)
- A ring buffer (last 200 events) for SSE Last-Event-ID reconnect-resume
- A list of active subscribers (asyncio queues, one per connected SSE
  client; usually exactly one — sieve-test connects once per run)

Pipeline call sites (main.py turn-handler) call ``bus.emit(...)`` after
each turn / writer completion / phase transition. The bus enqueues to
all active subscribers and appends to the ring buffer. SSE consumers
get a typed JSON line per event.

CARDINAL RULE: no test logic in this module. It is plumbing for the
protocol surface only. Scenarios, hypotheses, grading happen in
sieve-test.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from .schemas import (
    ComponentBreakdown, ErrorEvent, PhaseChange, SieveTestEvent,
    StoreState, TurnComplete, WriterDone,
)

logger = logging.getLogger("recall.test_mode")


class EventBus:
    """In-process event bus. Single instance per Sieve daemon."""

    BUFFER_SIZE = 200

    def __init__(self) -> None:
        self._counter = 0
        self._buffer: deque[SieveTestEvent] = deque(maxlen=self.BUFFER_SIZE)
        self._subscribers: list[asyncio.Queue[SieveTestEvent]] = []

    # ─── lifecycle ─────────────────────────────────────────────────────────────

    def next_event_id(self) -> str:
        self._counter += 1
        return f"evt_{self._counter:08d}"

    def n_events_emitted(self) -> int:
        return self._counter

    # ─── emission API (called from pipeline hooks) ─────────────────────────────

    def emit_turn_complete(
        self,
        *,
        turn_idx: int,
        sieve_inbound_tokens: int,
        sieve_outbound_tokens: int,
        sys: int, ctx: int, hist: int, user: int, tools: int,
        latency_ms: int,
        request_body_hash: str,
        response_body_hash: str,
        phase_at_turn: str,
        facts_in_store_at_turn: int,
    ) -> None:
        try:
            event = TurnComplete(
                schema_version=1,
                event_id=self.next_event_id(),
                event_type="turn_complete",
                ts_utc=_now_iso(),
                turn_idx=turn_idx,
                sieve_inbound_tokens=sieve_inbound_tokens,
                sieve_outbound_tokens=sieve_outbound_tokens,
                component_breakdown=ComponentBreakdown(
                    sys=sys, ctx=ctx, hist=hist, user=user, tools=tools,
                ),
                latency_ms=latency_ms,
                request_body_hash=request_body_hash,
                response_body_hash=response_body_hash,
                phase_at_turn=phase_at_turn,  # type: ignore[arg-type]
                facts_in_store_at_turn=facts_in_store_at_turn,
            )
        except Exception as exc:
            logger.warning("test_mode: failed to construct turn_complete: %s", exc)
            return
        self._publish(event)

    def emit_writer_done(
        self, *, turn_idx: int, s2_invoked: bool,
        candidates_extracted: int, candidates_kept: int, store_delta: int,
    ) -> None:
        try:
            event = WriterDone(
                schema_version=1,
                event_id=self.next_event_id(),
                event_type="writer_done",
                ts_utc=_now_iso(),
                turn_idx=turn_idx,
                s2_invoked=s2_invoked,
                candidates_extracted=candidates_extracted,
                candidates_kept=candidates_kept,
                store_delta=store_delta,
            )
        except Exception as exc:
            logger.warning("test_mode: failed to construct writer_done: %s", exc)
            return
        self._publish(event)

    def emit_phase_change(
        self, *, from_phase: str, to_phase: str,
        fact_count: int, turn_idx: int,
    ) -> None:
        try:
            event = PhaseChange(
                schema_version=1,
                event_id=self.next_event_id(),
                event_type="phase_change",
                ts_utc=_now_iso(),
                from_phase=from_phase,  # type: ignore[arg-type]
                to_phase=to_phase,  # type: ignore[arg-type]
                fact_count=fact_count,
                turn_idx=turn_idx,
            )
        except Exception as exc:
            logger.warning("test_mode: failed to construct phase_change: %s", exc)
            return
        self._publish(event)

    def emit_store_state(
        self, *, facts_total: int,
        facts_by_kind: dict[str, int] | None = None,
        last_writer_event_id: str | None = None,
    ) -> None:
        try:
            event = StoreState(
                schema_version=1,
                event_id=self.next_event_id(),
                event_type="store_state",
                ts_utc=_now_iso(),
                facts_total=facts_total,
                facts_by_kind=facts_by_kind or {},
                last_writer_event_id=last_writer_event_id,
            )
        except Exception as exc:
            logger.warning("test_mode: failed to construct store_state: %s", exc)
            return
        self._publish(event)

    def emit_error(
        self, *, error_class: str, error_message: str,
        turn_idx: int | None = None,
    ) -> None:
        try:
            event = ErrorEvent(
                schema_version=1,
                event_id=self.next_event_id(),
                event_type="error",
                ts_utc=_now_iso(),
                turn_idx=turn_idx,
                error_class=error_class,
                error_message=error_message[:1900],
            )
        except Exception as exc:
            logger.warning("test_mode: failed to construct error event: %s", exc)
            return
        self._publish(event)

    # ─── subscriber management ─────────────────────────────────────────────────

    def subscribe(self, *, last_event_id: str | None = None) -> asyncio.Queue[SieveTestEvent]:
        """Register a new SSE subscriber. Returns its queue.

        If ``last_event_id`` is provided, replay any buffered events newer
        than that id BEFORE the queue starts receiving live events. If the
        id is older than the buffer's tail (overflow), emit a synthetic
        error event noting the gap.
        """
        queue: asyncio.Queue[SieveTestEvent] = asyncio.Queue()

        if last_event_id is not None:
            self._replay_from(last_event_id, queue)

        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[SieveTestEvent]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    async def stream(
        self, queue: asyncio.Queue[SieveTestEvent],
    ) -> AsyncIterator[SieveTestEvent]:
        """Yield events from a subscriber's queue until cancellation."""
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self.unsubscribe(queue)

    # ─── internals ─────────────────────────────────────────────────────────────

    def _publish(self, event: SieveTestEvent) -> None:
        self._buffer.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer; drop and emit an in-band error.
                logger.warning("test_mode: subscriber queue full; dropping event")

    def _replay_from(
        self, last_event_id: str, queue: asyncio.Queue[SieveTestEvent],
    ) -> None:
        """Push buffered events newer than last_event_id into the queue."""
        # Linear scan; buffer is bounded.
        replay: list[SieveTestEvent] = []
        found = False
        for event in self._buffer:
            if found:
                replay.append(event)
            elif event.event_id == last_event_id:
                found = True
        if not found and self._buffer:
            # Buffer overflow — sieve-test missed events between
            # last_event_id and the buffer's tail. Surface as in-band error.
            err = ErrorEvent(
                schema_version=1,
                event_id=self.next_event_id(),
                event_type="error",
                ts_utc=_now_iso(),
                turn_idx=None,
                error_class="TelemetryGap",
                error_message=(
                    f"buffer overflow on reconnect: requested last_event_id={last_event_id} "
                    f"older than buffer tail. Possible event loss."
                ),
            )
            self._buffer.append(err)
            queue.put_nowait(err)
            # Then replay everything currently in buffer.
            for event in self._buffer:
                if event is not err:
                    queue.put_nowait(event)
            return

        for event in replay:
            queue.put_nowait(event)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ─── module-level singleton (None when test_mode is off) ───────────────────────

_BUS: EventBus | None = None


def get_bus() -> EventBus | None:
    """Return the active EventBus, or None if test_mode is disabled."""
    return _BUS


def init_bus() -> EventBus:
    """Create the singleton EventBus. Idempotent."""
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS


def shutdown_bus() -> None:
    global _BUS
    _BUS = None


__all__ = ["EventBus", "get_bus", "init_bus", "shutdown_bus"]

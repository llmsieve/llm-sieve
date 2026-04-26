"""FastAPI router for test-mode endpoints.

Mounted on the app ONLY when ``SIEVE_TEST_MODE=on``. Provides:

- ``POST /test/control/set-config``
- ``POST /test/control/wipe-store``
- ``POST /test/control/set-clock``
- ``POST /test/control/start-run``
- ``POST /test/control/end-run``
- ``GET  /test/state``
- ``GET  /test/events``  (SSE stream)

CARDINAL RULE: this module touches Sieve runtime via narrow interfaces
(state shared with main.py via ``app.state``). It does NOT contain test
logic — that lives in sieve-test.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from . import PROTOCOL_VERSION, is_test_mode_enabled
from .event_bus import get_bus, init_bus
from .schemas import (
    EndRunRequest, EndRunResponse,
    ErrorResponse,
    SetClockRequest, SetClockResponse,
    SetConfigRequest, SetConfigResponse,
    StartRunRequest, StartRunResponse,
    StateResponse,
    WipeStoreRequest, WipeStoreResponse,
)

logger = logging.getLogger("recall.test_mode")


def build_router() -> APIRouter:
    """Build the /test/* router. Caller mounts onto the FastAPI app."""
    router = APIRouter(prefix="/test", tags=["test_mode"])

    # ─── /test/state ────────────────────────────────────────────────────────────

    @router.get("/state")
    async def state(request: Request) -> JSONResponse:
        try:
            payload = _build_state_response(request)
            return JSONResponse(payload.model_dump())
        except Exception as exc:
            return _error_500(f"state introspection failed: {exc}")

    # ─── /test/control/* ────────────────────────────────────────────────────────

    @router.post("/control/set-config")
    async def set_config(req: SetConfigRequest, request: Request) -> JSONResponse:
        if req.require_test_mode and not is_test_mode_enabled():
            return _error_400("not in test mode (SIEVE_TEST_MODE unset)")
        applied: list[str] = []
        rejected: dict[str, str] = {}
        config = getattr(request.app.state, "sieve_config", None)
        if config is None:
            return _error_500("sieve config not on app.state")
        for key, value in req.config.items():
            try:
                _apply_dotted_key(config, key, value)
                applied.append(key)
            except Exception as exc:
                rejected[key] = str(exc)
        resp = SetConfigResponse(applied_keys=applied, rejected_keys=rejected)
        return JSONResponse(resp.model_dump())

    @router.post("/control/wipe-store")
    async def wipe_store(req: WipeStoreRequest, request: Request) -> JSONResponse:
        store = getattr(request.app.state, "memory_store", None)
        if store is None:
            return _error_500("memory_store not on app.state")
        try:
            facts_before = _count_facts(store)
            _wipe_store_in_place(store)
            facts_after = _count_facts(store)
        except Exception as exc:
            return _error_500(f"wipe failed: {exc}")
        resp = WipeStoreResponse(facts_before=facts_before, facts_after=facts_after)
        return JSONResponse(resp.model_dump())

    @router.post("/control/set-clock")
    async def set_clock(req: SetClockRequest) -> JSONResponse:
        try:
            from sieve import clock as sieve_clock
            previous = None
            try:
                previous = sieve_clock.get_clock().now().isoformat()
            except Exception:
                pass
            # Write to the file-backed clock if SIEVE_CLOCK_SOURCE is file:...
            import os
            source = os.environ.get("SIEVE_CLOCK_SOURCE", "")
            if source.startswith("file:"):
                path = source[len("file:"):]
                with open(path, "w") as f:
                    f.write(req.iso_utc + "\n")
            resp = SetClockResponse(
                previous_iso_utc=previous,
                current_iso_utc=req.iso_utc,
            )
            return JSONResponse(resp.model_dump())
        except Exception as exc:
            return _error_500(f"set-clock failed: {exc}")

    @router.post("/control/start-run")
    async def start_run(req: StartRunRequest, request: Request) -> JSONResponse:
        existing = getattr(request.app.state, "active_run_uuid", None)
        if existing is not None and existing != req.run_uuid:
            return _error_409(f"another run active: {existing}")
        request.app.state.active_run_uuid = req.run_uuid
        # Ensure event bus is up.
        init_bus()
        try:
            sieve_sha = _get_sieve_commit_sha()
            subj_digest = _get_model_digest(_subject_model_id(request))
            resp = StartRunResponse(
                run_uuid=req.run_uuid,
                sieve_commit_sha=sieve_sha,
                subject_model_digest=subj_digest,
                grader_model_digest=None,
                sieve_test_mode_version=PROTOCOL_VERSION,
            )
            return JSONResponse(resp.model_dump())
        except Exception as exc:
            return _error_500(f"start-run failed: {exc}")

    @router.post("/control/end-run")
    async def end_run(req: EndRunRequest, request: Request) -> JSONResponse:
        bus = get_bus()
        n_emitted = bus.n_events_emitted() if bus is not None else 0
        try:
            facts = _count_facts(getattr(request.app.state, "memory_store", None))
        except Exception:
            facts = 0
        request.app.state.active_run_uuid = None
        resp = EndRunResponse(
            run_uuid=req.run_uuid,
            n_events_emitted=n_emitted,
            final_facts_count=facts,
        )
        return JSONResponse(resp.model_dump())

    # ─── /test/events SSE ───────────────────────────────────────────────────────

    @router.get("/events")
    async def events(
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> EventSourceResponse:
        bus = get_bus()
        if bus is None:
            init_bus()
            bus = get_bus()
        assert bus is not None

        queue = bus.subscribe(last_event_id=last_event_id)

        async def event_generator():
            try:
                async for event in bus.stream(queue):
                    yield {
                        "id": event.event_id,
                        "data": event.model_dump_json(),
                    }
            except asyncio.CancelledError:
                logger.debug("test_mode SSE cancelled (client disconnect)")
                raise

        return EventSourceResponse(event_generator())

    return router


# ─── helpers ────────────────────────────────────────────────────────────────────


def _build_state_response(request: Request) -> StateResponse:
    config = getattr(request.app.state, "sieve_config", None)
    store = getattr(request.app.state, "memory_store", None)
    started = getattr(request.app.state, "started_at", time.time())

    facts = 0
    try:
        facts = _count_facts(store)
    except Exception:
        pass

    subject_model_id = None
    writer_model_id = None
    if config is not None:
        try:
            subject_model_id = config.provider.default_model
        except Exception:
            pass
        try:
            writer_model_id = config.writer.model
        except Exception:
            pass

    sieve_mode = "test" if is_test_mode_enabled() else "production"
    active_run = getattr(request.app.state, "active_run_uuid", None)

    return StateResponse(
        sieve_test_mode_version=PROTOCOL_VERSION,
        sieve_mode=sieve_mode,  # type: ignore[arg-type]
        daemon_uptime_s=int(time.time() - started),
        current_phase=None,
        facts_in_store=facts,
        subject_model_id=subject_model_id,
        writer_model_id=writer_model_id,
        active_run_uuid=active_run,
    )


def _count_facts(store: Any) -> int:
    if store is None:
        return 0
    conn = getattr(store, "_conn", None) or getattr(store, "conn", None)
    if conn is None:
        return 0
    try:
        cur = conn.execute("SELECT COUNT(*) FROM facts")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _wipe_store_in_place(store: Any) -> None:
    """Truncate facts/entities/relationships tables; preserve schema."""
    conn = getattr(store, "_conn", None) or getattr(store, "conn", None)
    if conn is None:
        raise RuntimeError("memory_store has no conn attribute")
    # Best-effort: list of tables we know Sieve uses. Missing tables are ignored.
    for table in ("facts", "entities", "relationships", "episodes",
                  "session_coherence", "tool_calls"):
        try:
            conn.execute(f"DELETE FROM {table}")
        except Exception:
            pass
    conn.commit()


def _apply_dotted_key(config: Any, dotted: str, value: Any) -> None:
    """Apply ``dotted`` (e.g. 'provider.default_model') = value on config."""
    parts = dotted.split(".")
    obj = config
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _subject_model_id(request: Request) -> str | None:
    config = getattr(request.app.state, "sieve_config", None)
    if config is None:
        return None
    try:
        return config.provider.default_model
    except Exception:
        return None


def _get_sieve_commit_sha() -> str:
    """Return git HEAD SHA of the sieve install. Best-effort."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd="/home/ath/Dev_Projects/llm-sieve",
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _get_model_digest(model_id: str | None) -> str:
    """Return sha256 of `ollama show <model> --modelfile`. Returns 64 zeros on failure."""
    if not model_id:
        return "0" * 64
    import hashlib
    import subprocess
    try:
        out = subprocess.check_output(
            ["ollama", "show", model_id, "--modelfile"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return hashlib.sha256(out.encode("utf-8")).hexdigest()
    except Exception:
        return "0" * 64


# ─── error helpers ──────────────────────────────────────────────────────────────


def _error_400(message: str, error_class: str = "BadRequest") -> JSONResponse:
    err = ErrorResponse(error_class=error_class, error_message=message)
    return JSONResponse(err.model_dump(), status_code=400)


def _error_409(message: str, error_class: str = "Conflict") -> JSONResponse:
    err = ErrorResponse(error_class=error_class, error_message=message)
    return JSONResponse(err.model_dump(), status_code=409)


def _error_500(message: str, error_class: str = "InternalError") -> JSONResponse:
    err = ErrorResponse(error_class=error_class, error_message=message)
    return JSONResponse(err.model_dump(), status_code=500)


__all__ = ["build_router"]

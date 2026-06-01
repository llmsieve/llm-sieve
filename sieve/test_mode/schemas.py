"""Pydantic schemas for test-mode protocol.

These schemas define the wire contract for ``/test/control/*`` requests/
responses and ``/test/events`` SSE payloads. The schemas are versioned
(``PROTOCOL_VERSION``); breaking changes bump the major version. An
test harness on the other end consumes these schemas and fails fast
on Pydantic validation errors if drift occurs — the Sieve runtime
keeps working regardless.

This module defines the runtime side of the contract authoritatively.
No harness code is imported here.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from . import PROTOCOL_VERSION

SchemaVersion = Literal[1]


# ─── Event schemas ─────────────────────────────────────────────────────────────


class ComponentBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sys: int = Field(ge=0)
    ctx: int = Field(ge=0)
    hist: int = Field(ge=0)
    user: int = Field(ge=0)
    tools: int = Field(ge=0)


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: SchemaVersion
    event_id: str = Field(min_length=1)
    event_type: str
    ts_utc: str


class TurnComplete(_EventBase):
    event_type: Literal["turn_complete"]
    turn_idx: int = Field(ge=1, le=120)
    sieve_inbound_tokens: int = Field(ge=0)
    sieve_outbound_tokens: int = Field(ge=0)
    component_breakdown: ComponentBreakdown
    latency_ms: int = Field(ge=0)
    request_body_hash: str = Field(min_length=64, max_length=64)
    response_body_hash: str = Field(min_length=64, max_length=64)
    phase_at_turn: Literal["OBSERVE", "ACCUMULATE", "ACTIVATE"]
    facts_in_store_at_turn: int = Field(ge=0)


class WriterDone(_EventBase):
    event_type: Literal["writer_done"]
    turn_idx: int = Field(ge=1, le=120)
    s2_invoked: bool
    candidates_extracted: int = Field(ge=0)
    candidates_kept: int = Field(ge=0)
    store_delta: int


class PhaseChange(_EventBase):
    event_type: Literal["phase_change"]
    from_phase: Literal["OBSERVE", "ACCUMULATE", "ACTIVATE"]
    to_phase: Literal["OBSERVE", "ACCUMULATE", "ACTIVATE"]
    fact_count: int = Field(ge=0)
    turn_idx: int = Field(ge=1, le=120)


class StoreState(_EventBase):
    event_type: Literal["store_state"]
    facts_total: int = Field(ge=0)
    facts_by_kind: dict[str, int] = Field(default_factory=dict)
    last_writer_event_id: str | None = None


class ErrorEvent(_EventBase):
    event_type: Literal["error"]
    turn_idx: int | None = None
    error_class: str = Field(min_length=1)
    error_message: str = Field(max_length=2000)


SieveTestEvent = Annotated[
    Union[TurnComplete, WriterDone, PhaseChange, StoreState, ErrorEvent],
    Field(discriminator="event_type"),
]


# ─── Control schemas ───────────────────────────────────────────────────────────


class _ControlBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(default=PROTOCOL_VERSION, ge=1)


class ErrorResponse(_ControlBase):
    ok: Literal[False] = False
    error_class: str = Field(min_length=1)
    error_message: str = Field(max_length=2000)


class SetConfigRequest(_ControlBase):
    config: dict[str, Any]
    require_test_mode: bool = True


class SetConfigResponse(_ControlBase):
    ok: Literal[True] = True
    applied_keys: list[str] = Field(default_factory=list)
    rejected_keys: dict[str, str] = Field(default_factory=dict)


class WipeStoreRequest(_ControlBase):
    confirm: Literal["WIPE"]


class WipeStoreResponse(_ControlBase):
    ok: Literal[True] = True
    facts_before: int = Field(ge=0)
    facts_after: int = Field(ge=0)


class SetClockRequest(_ControlBase):
    iso_utc: str


class SetClockResponse(_ControlBase):
    ok: Literal[True] = True
    previous_iso_utc: str | None = None
    current_iso_utc: str


class StartRunRequest(_ControlBase):
    run_uuid: str = Field(min_length=36, max_length=36)
    scenario_id: str = Field(min_length=1)
    scenario_version: int = Field(ge=1)
    seed: int = Field(ge=0)


class StartRunResponse(_ControlBase):
    ok: Literal[True] = True
    run_uuid: str
    sieve_commit_sha: str = Field(min_length=7)
    subject_model_digest: str = Field(min_length=64, max_length=64)
    grader_model_digest: str | None = None
    sieve_test_mode_version: int = Field(ge=1)


class EndRunRequest(_ControlBase):
    run_uuid: str = Field(min_length=36, max_length=36)
    flush_telemetry: bool = True


class EndRunResponse(_ControlBase):
    ok: Literal[True] = True
    run_uuid: str
    n_events_emitted: int = Field(ge=0)
    final_facts_count: int = Field(ge=0)


class StateResponse(_ControlBase):
    ok: Literal[True] = True
    sieve_test_mode_version: int = Field(ge=1)
    sieve_mode: Literal["production", "test"]
    daemon_uptime_s: int = Field(ge=0)
    current_phase: Literal["OBSERVE", "ACCUMULATE", "ACTIVATE"] | None = None
    facts_in_store: int = Field(ge=0)
    subject_model_id: str | None = None
    writer_model_id: str | None = None
    active_run_uuid: str | None = None


__all__ = [
    "PROTOCOL_VERSION",
    "ComponentBreakdown",
    "TurnComplete", "WriterDone", "PhaseChange", "StoreState", "ErrorEvent",
    "SieveTestEvent",
    "ErrorResponse",
    "SetConfigRequest", "SetConfigResponse",
    "WipeStoreRequest", "WipeStoreResponse",
    "SetClockRequest", "SetClockResponse",
    "StartRunRequest", "StartRunResponse",
    "EndRunRequest", "EndRunResponse",
    "StateResponse",
]

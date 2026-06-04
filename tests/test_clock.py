"""Tests for the injectable clock in sieve.clock.

Why this exists: Sieve's temporal logic (fact stamping, supersession
markers, last-confirmed timestamps) was pinned to wall-clock. That
made reproducible multi-day evaluation impossible without waiting
real days. The clock abstraction lets the test harness inject a
virtual "now" so a 90-day simulation runs in minutes. Production
code is unaffected: if SIEVE_CLOCK_SOURCE is unset, the WallClock
is used — same behaviour as before.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest


def test_wallclock_returns_current_utc_time():
    from sieve.clock import WallClock

    clock = WallClock()
    before = datetime.now(timezone.utc)
    got = clock.now()
    after = datetime.now(timezone.utc)

    assert got.tzinfo is not None, "WallClock must return tz-aware datetimes"
    assert before <= got <= after


def test_injected_clock_reads_iso_datetime_from_file(tmp_path: Path):
    from sieve.clock import InjectedClock

    clock_file = tmp_path / "sieve_clock"
    clock_file.write_text("2030-06-15T12:30:45+00:00")

    clock = InjectedClock(f"file:{clock_file}")
    got = clock.now()

    assert got == datetime(2030, 6, 15, 12, 30, 45, tzinfo=timezone.utc)


def test_injected_clock_file_is_reread_on_every_call(tmp_path: Path):
    """The runner advances the clock mid-run by rewriting the file;
    Sieve must pick that up without re-instantiating the clock."""
    from sieve.clock import InjectedClock

    clock_file = tmp_path / "sieve_clock"
    clock_file.write_text("2030-01-01T00:00:00+00:00")
    clock = InjectedClock(f"file:{clock_file}")

    first = clock.now()
    clock_file.write_text("2030-06-15T00:00:00+00:00")
    second = clock.now()

    assert first == datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert second == datetime(2030, 6, 15, tzinfo=timezone.utc)


def test_injected_clock_reads_iso_datetime_from_env(monkeypatch):
    from sieve.clock import InjectedClock

    monkeypatch.setenv("SIEVE_TEST_NOW", "2031-02-28T08:00:00+00:00")
    clock = InjectedClock("env:SIEVE_TEST_NOW")

    assert clock.now() == datetime(2031, 2, 28, 8, 0, 0, tzinfo=timezone.utc)


def test_injected_clock_rejects_unknown_source_prefix():
    from sieve.clock import InjectedClock

    with pytest.raises(ValueError, match="file:|env:"):
        InjectedClock("wallclock")


def test_injected_clock_assumes_utc_if_no_timezone(tmp_path: Path):
    """A naive ISO string gets UTC tacked on — matches the production
    convention that all Sieve timestamps are UTC."""
    from sieve.clock import InjectedClock

    clock_file = tmp_path / "sieve_clock"
    clock_file.write_text("2030-06-15T12:00:00")
    clock = InjectedClock(f"file:{clock_file}")

    got = clock.now()
    assert got.tzinfo is not None
    assert got == datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_get_clock_defaults_to_wallclock_when_env_unset(monkeypatch):
    """Critical production-behaviour-preservation test: with no env
    var set, get_clock() returns a WallClock. This is the
    zero-production-change guarantee."""
    from sieve.clock import WallClock, get_clock

    monkeypatch.delenv("SIEVE_CLOCK_SOURCE", raising=False)
    clock = get_clock()

    assert isinstance(clock, WallClock)


def test_get_clock_returns_wallclock_for_explicit_wallclock_value(monkeypatch):
    from sieve.clock import WallClock, get_clock

    monkeypatch.setenv("SIEVE_CLOCK_SOURCE", "wallclock")
    assert isinstance(get_clock(), WallClock)


def test_get_clock_returns_injected_clock_for_file_source(monkeypatch, tmp_path: Path):
    from sieve.clock import InjectedClock, get_clock

    clock_file = tmp_path / "sieve_clock"
    clock_file.write_text("2030-01-01T00:00:00+00:00")
    monkeypatch.setenv("SIEVE_CLOCK_SOURCE", f"file:{clock_file}")

    clock = get_clock()
    assert isinstance(clock, InjectedClock)
    assert clock.now() == datetime(2030, 1, 1, tzinfo=timezone.utc)


# --- Integration: injection points honour the clock --------------------


def test_store_now_iso_respects_injected_clock(monkeypatch):
    """Writes stamped via store._now_iso must reflect the injected
    clock, not the wall clock. This is the contract the validation
    harness relies on."""
    monkeypatch.setenv("SIEVE_NOW", "2030-06-15T12:00:00+00:00")
    monkeypatch.setenv("SIEVE_CLOCK_SOURCE", "env:SIEVE_NOW")

    from sieve import store

    assert store._now_iso() == "2030-06-15T12:00:00+00:00"


def test_validation_collector_now_iso_respects_injected_clock(monkeypatch):
    monkeypatch.setenv("SIEVE_NOW", "2030-06-15T12:00:00+00:00")
    monkeypatch.setenv("SIEVE_CLOCK_SOURCE", "env:SIEVE_NOW")

    from sieve import validation_collector

    assert validation_collector._now_iso() == "2030-06-15T12:00:00+00:00"


def test_tool_registry_now_iso_respects_injected_clock(monkeypatch):
    monkeypatch.setenv("SIEVE_NOW", "2030-06-15T12:00:00+00:00")
    monkeypatch.setenv("SIEVE_CLOCK_SOURCE", "env:SIEVE_NOW")

    from sieve import tool_registry

    assert tool_registry._now_iso() == "2030-06-15T12:00:00+00:00"


def test_store_now_iso_falls_back_to_wallclock_when_env_unset(monkeypatch):
    """Production-behaviour preservation: with no env set,
    store._now_iso() still returns a live UTC ISO string close to
    the real current time."""
    monkeypatch.delenv("SIEVE_CLOCK_SOURCE", raising=False)

    from sieve import store

    before = datetime.now(timezone.utc)
    got = datetime.fromisoformat(store._now_iso())
    after = datetime.now(timezone.utc)

    assert before <= got <= after


def test_progression_phases_are_reproducible_under_frozen_clock(monkeypatch):
    """Progression is fact-count driven, not time-driven.  Under a
    frozen clock the OBSERVE → ACCUMULATE → ACTIVATE transitions must
    still fire at the configured thresholds — which is what makes the
    validation harness's 90-day compressed run produce the same
    phase trajectory as a wall-clock 90-day run.
    """
    monkeypatch.setenv("SIEVE_NOW", "2030-01-01T00:00:00+00:00")
    monkeypatch.setenv("SIEVE_CLOCK_SOURCE", "env:SIEVE_NOW")

    from sieve.config import ProgressionConfig
    from sieve.progression import Phase, detect_phase

    cfg = ProgressionConfig(
        phase_1_threshold=20,
        phase_2_threshold=60,
        observe_turns=30,
        accumulate_turns=15,
        activate_turns=5,
    )

    # Walk the fact count up and assert transitions happen at the
    # configured thresholds regardless of the frozen clock.
    assert detect_phase(0, cfg).phase is Phase.OBSERVE
    assert detect_phase(19, cfg).phase is Phase.OBSERVE
    assert detect_phase(20, cfg).phase is Phase.ACCUMULATE
    assert detect_phase(59, cfg).phase is Phase.ACCUMULATE
    assert detect_phase(60, cfg).phase is Phase.ACTIVATE
    assert detect_phase(999, cfg).phase is Phase.ACTIVATE

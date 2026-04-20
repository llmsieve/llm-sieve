"""Injectable clock for reproducible testing.

Production code reads the current time through :func:`get_clock`,
which returns a :class:`WallClock` by default. Tests and offline
evaluation harnesses can set ``SIEVE_CLOCK_SOURCE`` to feed Sieve
a controlled "now" — enabling multi-day simulations to complete
in real-time seconds instead of real-time days.

When ``SIEVE_CLOCK_SOURCE`` is unset, behaviour is byte-identical
to the pre-clock-abstraction code path.

Source formats:

* unset / ``"wallclock"`` → :class:`WallClock`
* ``"file:/some/path"`` → :class:`InjectedClock` reading ISO-8601
  from that file on every call
* ``"env:VAR_NAME"`` → :class:`InjectedClock` reading ISO-8601
  from that environment variable on every call
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    """Returns the real UTC time. The production default."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class InjectedClock:
    """Reads "now" from an external source on every call.

    ``source`` is one of:

    * ``"file:/path/to/clock"`` — the file contains a single
      ISO-8601 datetime string.
    * ``"env:VAR_NAME"`` — the named environment variable holds
      the ISO-8601 string.

    The source is re-read on every call so an external runner can
    advance the clock without Sieve restarting.
    """

    def __init__(self, source: str):
        if source.startswith("file:"):
            self._kind = "file"
            self._ref = Path(source[len("file:"):])
        elif source.startswith("env:"):
            self._kind = "env"
            self._ref = source[len("env:"):]
        else:
            raise ValueError(
                f"InjectedClock source must start with 'file:' or 'env:'; got {source!r}"
            )

    def now(self) -> datetime:
        if self._kind == "file":
            raw = Path(self._ref).read_text().strip()
        else:
            raw = os.environ[self._ref].strip()
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed


def get_clock() -> Clock:
    """Return the clock dictated by ``SIEVE_CLOCK_SOURCE``.

    Unset or ``"wallclock"`` → :class:`WallClock` (production default).
    Any ``file:`` or ``env:`` source → :class:`InjectedClock`.
    """
    source = os.environ.get("SIEVE_CLOCK_SOURCE", "wallclock")
    if source == "wallclock":
        return WallClock()
    return InjectedClock(source)

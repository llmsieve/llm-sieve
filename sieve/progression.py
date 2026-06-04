"""Progressive activation — pick a phase from the current fact count.

Sieve runs in one of three phases per request:

- **OBSERVE** — the store is thin; keep lots of recent conversation
  history so the model can answer from the raw turns while the writer
  catches up on fact extraction.
- **ACCUMULATE** — the store is growing; halve the conversation window
  and lean more on retrieval.
- **ACTIVATE** — the store is mature; minimal history, retrieval-driven.

This module owns only the decision ``(fact_count, config) -> decision``.
Counting facts is the store's job; applying the decision is the
composer's job (``compose_lean_payload`` accepts a ``progression``
override). Keeping the decision pure makes the logic trivial to unit
test and lets callers log ``decision.render_tag()`` without having to
re-derive the phase name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sieve.config import ProgressionConfig


class Phase(str, Enum):
    OBSERVE = "OBSERVE"
    ACCUMULATE = "ACCUMULATE"
    ACTIVATE = "ACTIVATE"


@dataclass(frozen=True)
class PhaseDecision:
    """One phase-detection result.

    Attributes
    ----------
    phase : Phase
        The chosen phase.
    turns : int
        How many prior user/assistant pairs the composer should retain.
    fact_count : int
        The fact count this decision was derived from. Logged alongside
        the phase tag so operators can see what drove the phase choice.
    """
    phase: Phase
    turns: int
    fact_count: int

    @property
    def label(self) -> str:
        return self.phase.value

    def render_tag(self) -> str:
        """Render the phase as a log/status tag: ``[ACCUMULATE: 35 facts]``."""
        return f"[{self.phase.value}: {self.fact_count} facts]"


def detect_phase(fact_count: int, config: ProgressionConfig) -> PhaseDecision:
    """Map a current-fact count to the active phase + turn budget.

    Boundaries are inclusive on entry: ``fact_count >= phase_1_threshold``
    enters ACCUMULATE, ``fact_count >= phase_2_threshold`` enters
    ACTIVATE. This matches the spec in the Progressive Activation Notion
    doc and means a single fact tips OBSERVE→ACCUMULATE at exactly the
    configured threshold.
    """
    if fact_count < 0:
        raise ValueError(f"fact_count must be non-negative, got {fact_count}")

    if fact_count < config.phase_1_threshold:
        return PhaseDecision(
            phase=Phase.OBSERVE,
            turns=config.observe_turns,
            fact_count=fact_count,
        )
    if fact_count < config.phase_2_threshold:
        return PhaseDecision(
            phase=Phase.ACCUMULATE,
            turns=config.accumulate_turns,
            fact_count=fact_count,
        )
    return PhaseDecision(
        phase=Phase.ACTIVATE,
        turns=config.activate_turns,
        fact_count=fact_count,
    )

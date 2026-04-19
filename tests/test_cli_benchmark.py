"""Tests for ``sieve benchmark``.

The benchmark exercises a live proxy, so we inject an httpx.MockTransport
that fakes the proxy's chat endpoint. That lets the tests verify:

- the 15-message script runs in order, phases line up
- inbound/outbound tokens are pulled from X-Sieve-* headers when present
- absence-signal heuristic fires correctly on a refusal response and
  correctly does NOT fire on a fabricated response
- the summary panel totals match what each turn reported
- the store_fact_count hook is called before/after each turn and
  drives the "facts_learned" total
"""

from __future__ import annotations

import json
from io import StringIO

import httpx
import pytest
from rich.console import Console

from sieve.cli_benchmark import (
    ABSENCE_PATTERNS,
    BENCHMARK_MESSAGES,
    BenchmarkSummary,
    TurnResult,
    looks_like_absence_signal,
    render_summary,
    run_benchmark,
)


# ── Script shape ─────────────────────────────────────────────────────────


def test_script_is_15_messages():
    assert len(BENCHMARK_MESSAGES) == 15


def test_script_phase_layout():
    """Matches the spec: 1-3 intro, 4-8 retrieve, 9-12 deep, 13-14 update, 15 trap."""
    phases = [m["phase"] for m in BENCHMARK_MESSAGES]
    assert phases[0:3] == ["introduce"] * 3
    assert phases[3:8] == ["retrieve"] * 5
    assert phases[8:12] == ["deep"] * 4
    assert phases[12:14] == ["update"] * 2
    assert phases[14] == "trap"


def test_script_messages_have_content():
    for m in BENCHMARK_MESSAGES:
        assert m.get("content"), f"missing content for {m!r}"
        assert isinstance(m["content"], str)


def test_trap_message_references_name_not_seen_earlier():
    """The trap must ask about an entity not mentioned in messages 1–14.

    If the fictional sibling's name appears in the setup, the model can
    answer truthfully from prior context and the trap is invalidated.
    """
    setup_text = " ".join(m["content"].lower() for m in BENCHMARK_MESSAGES[:-1])
    trap_content = BENCHMARK_MESSAGES[-1]["content"].lower()
    # Extract the proper noun(s) from the trap — here, "Jordan".
    # Cheap check: any word longer than 4 chars in the trap that also
    # appears in the setup would be suspicious.
    # But "what", "does", "work" etc. are fine. The specific trap uses
    # "Jordan" and "sibling" — neither should appear in setup.
    assert "jordan" not in setup_text
    assert "sibling" not in setup_text


# ── Absence-signal heuristic ─────────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "I don't know who Jordan is.",
    "I do not have any information about Jordan.",
    "You haven't mentioned a sibling to me.",
    "No record of Jordan in my memory.",
    "I'm not sure — you haven't told me about Jordan.",
    "I cannot recall any mention of a sibling called Jordan.",
    "No mention of Jordan was made.",
])
def test_absence_signal_detects_refusals(text):
    assert looks_like_absence_signal(text) is True


@pytest.mark.parametrize("text", [
    "Jordan is a software engineer at Google.",
    "Your sibling Jordan works as a dentist.",
    "Based on what you've told me, Jordan is a marine biologist too.",
    "",  # empty — ambiguous, treated as not-a-signal
])
def test_absence_signal_rejects_fabrications(text):
    assert looks_like_absence_signal(text) is False


def test_absence_patterns_are_lowercase():
    """Sanity: the matcher lowercases text before comparison."""
    for p in ABSENCE_PATTERNS:
        assert p == p.lower()


# ── Mock transport harness ───────────────────────────────────────────────


class FakeProxy:
    """Simulates a Sieve proxy's /api/chat endpoint.

    - ``inbound_by_turn`` and ``outbound_by_turn`` control the header values.
    - ``responses_by_phase`` controls what the model "says" per phase.
    - Increments ``facts`` on every "introduce" and "update" turn so the
      benchmark sees non-zero learning (mirrors real behaviour on those
      phases).
    """

    def __init__(
        self,
        *,
        inbound_by_turn: list[int],
        outbound_by_turn: list[int],
        responses_by_phase: dict[str, str],
    ):
        self.inbound = list(inbound_by_turn)
        self.outbound = list(outbound_by_turn)
        self.responses = responses_by_phase
        self.calls: list[dict] = []
        self.facts = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        turn_idx = len(self.calls)
        self.calls.append(body)
        user_content = body["messages"][0]["content"]

        # Work out phase from BENCHMARK_MESSAGES at this position so the fake
        # mimics the real script.
        phase = BENCHMARK_MESSAGES[turn_idx]["phase"]
        if phase in ("introduce", "update"):
            self.facts += 1

        text = self.responses.get(phase, "ok")
        resp_json = {
            "model": body.get("model"),
            "message": {"role": "assistant", "content": text},
            "done": True,
        }
        return httpx.Response(
            200,
            json=resp_json,
            headers={
                "X-Sieve-Inbound-Tokens": str(self.inbound[turn_idx]),
                "X-Sieve-Outbound-Tokens": str(self.outbound[turn_idx]),
                "X-Sieve-Rounds": "0",
            },
        )


def _run_with_fake(fake: FakeProxy) -> BenchmarkSummary:
    transport = httpx.MockTransport(fake.handler)
    return run_benchmark(
        base_url="http://fake-proxy",
        model="test-model",
        store_fact_count=lambda: fake.facts,
        transport=transport,
        timeout=5.0,
    )


# ── Core run_benchmark behaviour ─────────────────────────────────────────


def test_run_benchmark_executes_all_15_messages_in_order():
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.",
            "retrieve": "Answer.",
            "deep": "Answer.",
            "update": "Understood.",
            "trap": "I don't know who Jordan is.",
        },
    )
    summary = _run_with_fake(fake)

    # 15 calls in order, prompts match the script
    assert len(summary.turns) == 15
    for turn, msg in zip(summary.turns, BENCHMARK_MESSAGES):
        assert turn.prompt == msg["content"]
        assert turn.phase == msg["phase"]

    # Every call reached the proxy
    assert len(fake.calls) == 15


def test_run_benchmark_reads_token_headers():
    fake = FakeProxy(
        inbound_by_turn=list(range(1000, 16000, 1000))[:15],
        outbound_by_turn=[300] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.", "trap": "I don't know.",
        },
    )
    summary = _run_with_fake(fake)
    for i, turn in enumerate(summary.turns):
        assert turn.inbound_tokens == 1000 * (i + 1)
        assert turn.outbound_tokens == 300


def test_run_benchmark_reduction_percentage():
    fake = FakeProxy(
        inbound_by_turn=[1000] * 15,
        outbound_by_turn=[100] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.", "trap": "I don't know.",
        },
    )
    summary = _run_with_fake(fake)
    assert summary.total_inbound == 15_000
    assert summary.total_outbound == 1_500
    assert summary.reduction_pct == pytest.approx(90.0)


def test_run_benchmark_detects_absence_signal_on_trap():
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.",
            "trap": "I don't have any information about your sibling Jordan.",
        },
    )
    summary = _run_with_fake(fake)
    assert summary.trap_absence_signal is True
    # Only the trap row should carry an absence-signal value.
    non_trap = [t for t in summary.turns if t.phase != "trap"]
    assert all(t.absence_signal is None for t in non_trap)


def test_run_benchmark_flags_fabrication_on_trap():
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.",
            "trap": "Your sibling Jordan works as a software engineer at Apple.",
        },
    )
    summary = _run_with_fake(fake)
    assert summary.trap_absence_signal is False


def test_run_benchmark_counts_facts_learned():
    """Facts should grow on introduce + update phases in our fake."""
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.", "trap": "I don't know.",
        },
    )
    summary = _run_with_fake(fake)
    # 3 intros + 2 updates = 5 facts in our fake
    assert summary.facts_learned == 5


def test_run_benchmark_records_per_turn_fact_deltas():
    """facts_before/after per row should let us identify which message taught something."""
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.", "trap": "I don't know.",
        },
    )
    summary = _run_with_fake(fake)
    intros = [t for t in summary.turns if t.phase == "introduce"]
    assert all(t.facts_after - t.facts_before == 1 for t in intros)
    retrieves = [t for t in summary.turns if t.phase == "retrieve"]
    assert all(t.facts_after == t.facts_before for t in retrieves)


def test_run_benchmark_falls_back_to_char_count_when_header_missing():
    """If the proxy predates X-Sieve-Inbound-Tokens, the client approximates."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"content": "ok"}, "done": True},
            # No X-Sieve-* headers
        )

    transport = httpx.MockTransport(handler)
    summary = run_benchmark(
        base_url="http://fake-proxy",
        model="test-model",
        store_fact_count=lambda: 0,
        transport=transport,
        timeout=5.0,
    )
    # Every turn should have a non-zero approximation
    assert all(t.inbound_tokens > 0 for t in summary.turns)
    assert all(t.outbound_tokens == 0 for t in summary.turns)


# ── Rendering ────────────────────────────────────────────────────────────


def test_render_summary_includes_key_lines():
    summary = BenchmarkSummary(
        total_inbound=15_000,
        total_outbound=1_500,
        reduction_pct=90.0,
        facts_learned=5,
        trap_absence_signal=True,
        turns=[
            TurnResult(
                index=i,
                phase=BENCHMARK_MESSAGES[i - 1]["phase"],
                prompt=BENCHMARK_MESSAGES[i - 1]["content"],
                response="ok",
                inbound_tokens=1000,
                outbound_tokens=100,
                facts_before=0,
                facts_after=0,
                elapsed_s=0.5,
                absence_signal=(True if BENCHMARK_MESSAGES[i - 1]["phase"] == "trap" else None),
            )
            for i in range(1, 16)
        ],
    )
    buf = StringIO()
    console = Console(file=buf, width=200, force_terminal=False, no_color=True)
    render_summary(summary, model="m", base_url="http://p", console=console)
    out = buf.getvalue()

    assert "Per-message breakdown" in out
    assert "15,000" in out or "15000" in out
    assert "1,500" in out or "1500" in out
    assert "90.0%" in out
    assert "Facts learned" in out
    assert "5" in out
    assert "absence signal fired" in out
    # The self-documenting footer
    assert "sieve benchmark" in out


def test_render_summary_flags_fabrication():
    summary = BenchmarkSummary(
        total_inbound=100,
        total_outbound=80,
        reduction_pct=20.0,
        facts_learned=0,
        trap_absence_signal=False,
        turns=[],
    )
    buf = StringIO()
    console = Console(file=buf, width=160, force_terminal=False, no_color=True)
    render_summary(summary, model="m", base_url="http://p", console=console)
    out = buf.getvalue()
    assert "no absence signal detected" in out

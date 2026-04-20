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
        # The current user message is always the LAST user message in
        # the messages array (history from prior turns precedes it).
        user_content = ""
        for m in reversed(body["messages"]):
            if m.get("role") == "user":
                user_content = m.get("content", "")
                break

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
        stream=False,
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
        stream=False,
    )
    # Every turn should have a non-zero approximation
    assert all(t.inbound_tokens > 0 for t in summary.turns)
    assert all(t.outbound_tokens == 0 for t in summary.turns)


# ── Rendering ────────────────────────────────────────────────────────────


def test_render_summary_includes_key_lines():
    summary = BenchmarkSummary(
        total_inbound=15_000,
        total_outbound=1_500,
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
        facts_learned=0,
        trap_absence_signal=False,
        turns=[],
    )
    buf = StringIO()
    console = Console(file=buf, width=160, force_terminal=False, no_color=True)
    render_summary(summary, model="m", base_url="http://p", console=console)
    out = buf.getvalue()
    assert "no absence signal detected" in out


# ── History threading ───────────────────────────────────────────────────


def test_run_benchmark_threads_history_across_turns():
    """Turn N's request should contain N-1 prior user+assistant exchanges.

    Real agents ship growing history; the baseline pass is supposed to
    demonstrate that bloat.
    """
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Answer.", "deep": "Answer.",
            "update": "Understood.", "trap": "I don't know.",
        },
    )
    _run_with_fake(fake)
    # Turn 1: just the current user message.
    first = fake.calls[0]["messages"]
    assert len([m for m in first if m.get("role") == "user"]) == 1
    # Turn 5: 5 user messages (turns 1-5) and 4 assistant replies (turns 1-4).
    fifth = fake.calls[4]["messages"]
    user_count = sum(1 for m in fifth if m.get("role") == "user")
    asst_count = sum(1 for m in fifth if m.get("role") == "assistant")
    assert user_count == 5
    assert asst_count == 4
    # Last message should always be the new user turn.
    assert fifth[-1]["role"] == "user"


# ── LLM-grader pluggability ─────────────────────────────────────────────


def test_run_benchmark_respects_injected_recall_grader():
    """grade_recall hook replaces the default keyword heuristic."""
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            "introduce": "Got it.", "retrieve": "Nonsense answer.",
            "deep": "Answer.", "update": "Understood.", "trap": "I don't know.",
        },
    )
    transport = httpx.MockTransport(fake.handler)
    # Grader always returns True; should override the keyword heuristic
    # that would have failed on "Nonsense answer."
    always_yes = lambda i, q, r, h: True if h else None
    summary = run_benchmark(
        base_url="http://fake-proxy",
        model="test-model",
        store_fact_count=lambda: fake.facts,
        transport=transport,
        timeout=5.0,
        stream=False,
        grade_recall=always_yes,
    )
    # 6 gradable turns (4,5,6,7,8,14), all marked correct by injected grader.
    assert summary.gradable_recalls == 6
    assert summary.correct_recalls == 6


def test_run_benchmark_respects_injected_trap_grader():
    fake = FakeProxy(
        inbound_by_turn=[100] * 15,
        outbound_by_turn=[50] * 15,
        responses_by_phase={
            # Response LOOKS like fabrication (would fail the heuristic)…
            "introduce": "ok", "retrieve": "ok", "deep": "ok", "update": "ok",
            "trap": "Jordan works at Apple.",
        },
    )
    transport = httpx.MockTransport(fake.handler)
    # …but injected grader overrides.
    summary = run_benchmark(
        base_url="http://fake-proxy",
        model="test-model",
        store_fact_count=lambda: fake.facts,
        transport=transport,
        timeout=5.0,
        stream=False,
        grade_trap=lambda q, r: True,
    )
    assert summary.trap_absence_signal is True


# ── Pricing / headline ──────────────────────────────────────────────────


def test_pricing_table_has_expected_tiers():
    from sieve._pricing import PRICING_TABLE, dollars_saved, price_for
    for key in ("claude-opus", "claude-sonnet", "claude-haiku",
                "gpt-4o", "gpt-4o-mini", "local"):
        assert key in PRICING_TABLE
    assert price_for("local") == 0.0
    # Saving 1M tokens on sonnet ($3/M) should be $3.
    assert dollars_saved(1_000_000, "claude-sonnet") == pytest.approx(3.0)
    # Unknown tier returns 0 (safe default).
    assert price_for("made-up-model") == 0.0


def test_headline_mentions_reduction_and_baseline_sieve_tokens():
    from sieve.cli_benchmark import build_headline, CompareSummary
    baseline = BenchmarkSummary(
        total_inbound=24000, total_outbound=24000,
        facts_learned=0, trap_absence_signal=None, turns=[],
    )
    sieve = BenchmarkSummary(
        total_inbound=1000, total_outbound=8000,
        facts_learned=11, trap_absence_signal=True, turns=[],
        correct_recalls=6, gradable_recalls=6,
    )
    compare = CompareSummary(
        baseline_tokens=24000,
        sieve_outbound_tokens=8000,
        sieve_inbound_tokens=1000,
        baseline=baseline,
        sieve=sieve,
    )
    headline = build_headline(summary=compare, pricing_tier="claude-sonnet")
    # Reduction percentage reported
    assert "67%" in headline or "66%" in headline
    # Both raw token counts reported (the "X → Y" pattern)
    assert "24,000" in headline and "8,000" in headline
    assert "6/6" in headline
    assert "trap refused" in headline


def test_headline_single_mode_mentions_recalls():
    from sieve.cli_benchmark import build_headline
    summary = BenchmarkSummary(
        total_inbound=160, total_outbound=4000,
        facts_learned=11, trap_absence_signal=True, turns=[1, 2, 3],  # count, not real
        correct_recalls=5, gradable_recalls=6,
    )
    # Turns count is what matters for the render; we stub with fake list.
    headline = build_headline(single=summary)
    assert "5/6" in headline
    assert "refused on the trap" in headline


# ── Machine-readable output ─────────────────────────────────────────────


def test_summary_to_dict_round_trips_compare_summary():
    from sieve.cli_benchmark import summary_to_dict, CompareSummary
    baseline = BenchmarkSummary(
        total_inbound=24000, total_outbound=24000,
        facts_learned=0, trap_absence_signal=None, turns=[],
    )
    sieve = BenchmarkSummary(
        total_inbound=1000, total_outbound=8000,
        facts_learned=11, trap_absence_signal=True, turns=[],
        correct_recalls=6, gradable_recalls=6,
    )
    compare = CompareSummary(
        baseline_tokens=24000,
        sieve_outbound_tokens=8000,
        sieve_inbound_tokens=1000,
        baseline=baseline,
        sieve=sieve,
    )
    d = summary_to_dict(compare, model="qwen3.5:9b", pricing_tier="claude-sonnet")
    assert d["mode"] == "compare"
    assert d["tokens_saved"] == 16000
    assert d["reduction_pct"] == pytest.approx(66.67, rel=0.01)
    assert d["dollars_saved_per_run"] > 0
    assert d["dollars_saved_per_1k_runs"] > d["dollars_saved_per_run"]
    # JSON-serialisable.
    json.dumps(d)


def test_render_markdown_produces_paste_able_report():
    from sieve.cli_benchmark import render_markdown, CompareSummary
    baseline = BenchmarkSummary(
        total_inbound=24000, total_outbound=24000,
        facts_learned=0, trap_absence_signal=None,
        turns=[
            TurnResult(
                index=i, phase=BENCHMARK_MESSAGES[i-1]["phase"],
                prompt=BENCHMARK_MESSAGES[i-1]["content"],
                response="baseline response", inbound_tokens=1000, outbound_tokens=1000,
                facts_before=0, facts_after=0, elapsed_s=1.0, absence_signal=None,
            )
            for i in range(1, 16)
        ],
    )
    sieve = BenchmarkSummary(
        total_inbound=1000, total_outbound=8000,
        facts_learned=11, trap_absence_signal=True,
        correct_recalls=6, gradable_recalls=6,
        turns=[
            TurnResult(
                index=i, phase=BENCHMARK_MESSAGES[i-1]["phase"],
                prompt=BENCHMARK_MESSAGES[i-1]["content"],
                response="sieve response", inbound_tokens=300, outbound_tokens=500,
                facts_before=0, facts_after=0, elapsed_s=2.0, absence_signal=None,
            )
            for i in range(1, 16)
        ],
    )
    compare = CompareSummary(
        baseline_tokens=24000, sieve_outbound_tokens=8000,
        sieve_inbound_tokens=1000,
        baseline=baseline, sieve=sieve,
    )
    md = render_markdown(
        compare, model="qwen3.5:9b",
        sieve_base_url="http://x", direct_base_url="http://y",
        pricing_tier="claude-sonnet",
    )
    assert "## Sieve benchmark" in md
    assert "| Tokens sent to LLM" in md
    assert "**Savings" in md
    assert "$" in md
    assert "### Per-turn" in md


# ── AggregatedStat ──────────────────────────────────────────────────────


def test_aggregated_stat_empty():
    from sieve.cli_benchmark import AggregatedStat
    s = AggregatedStat.from_values([])
    assert s.n == 0
    assert s.mean == 0.0
    assert s.stddev == 0.0
    assert s.render() == "0"


def test_aggregated_stat_single_value():
    from sieve.cli_benchmark import AggregatedStat
    s = AggregatedStat.from_values([42.0])
    assert s.n == 1
    assert s.mean == 42.0
    assert s.stddev == 0.0
    # No stddev on a single value — render omits the ±.
    assert "±" not in s.render()


def test_aggregated_stat_multi_values():
    from sieve.cli_benchmark import AggregatedStat
    s = AggregatedStat.from_values([10.0, 12.0, 14.0])
    assert s.n == 3
    assert s.mean == pytest.approx(12.0)
    # Sample stddev of [10, 12, 14] = 2.0
    assert s.stddev == pytest.approx(2.0)
    assert "±" in s.render()


# ── Multi-run compare ───────────────────────────────────────────────────


def test_run_benchmark_compare_multi_aggregates():
    """Three compare runs → AggregatedCompareSummary with mean/stddev populated."""
    from sieve.cli_benchmark import run_benchmark_compare_multi

    # Build two fakes: one for baseline (high tokens), one for sieve
    # (low tokens). We cheat by using a single FakeProxy and flipping
    # an internal counter to return different token headers per pass.
    class TwoPassFake:
        def __init__(self):
            self.facts = 0
            self.calls = 0
            self.pass_index = 0  # 0 = baseline, 1 = sieve

        def handler(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            # Each "run" in the multi is 15 turns baseline + 15 turns sieve
            # = 30 calls. Baseline first (high tokens), then sieve (low).
            calls_in_this_run = (self.calls - 1) % 30
            is_baseline = calls_in_this_run < 15
            phase_idx = calls_in_this_run % 15
            if phase_idx == 15 - 1:
                # approximately scripting the end of a run
                pass
            phase = BENCHMARK_MESSAGES[phase_idx]["phase"]
            if phase == "introduce" or phase == "update":
                self.facts += 1

            # Token counts differ between baseline and sieve passes.
            if is_baseline:
                inbound = 2000
                outbound = 2000
            else:
                inbound = 200
                outbound = 300
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                },
                headers={
                    "X-Sieve-Inbound-Tokens": str(inbound),
                    "X-Sieve-Outbound-Tokens": str(outbound),
                },
            )

    fake = TwoPassFake()
    transport = httpx.MockTransport(fake.handler)

    resets = []
    def _reset():
        resets.append(1)
        fake.facts = 0

    agg = run_benchmark_compare_multi(
        runs=3,
        sieve_base_url="http://sieve",
        direct_base_url="http://direct",
        model="test-model",
        store_fact_count=lambda: fake.facts,
        reset_store=_reset,
        transport=transport,
        stream=False,
    )

    assert len(agg.runs) == 3
    assert agg.baseline_tokens.n == 3
    # Baseline tokens come from the actual agent-shaped payload (computed
    # client-side when use_sieve_headers=False). Sieve outbound comes
    # from our fake's X-Sieve-Outbound header (300 × 15 = 4500).
    assert agg.sieve_outbound_tokens.mean == pytest.approx(4_500)
    # Baseline >> Sieve outbound — the whole point of the benchmark.
    assert agg.baseline_tokens.mean > agg.sieve_outbound_tokens.mean * 5
    # All three runs are deterministic → zero stddev on the reduction
    # percentage.
    assert agg.reduction_pct.stddev == pytest.approx(0.0)
    # reset_store called before each of the 3 runs, plus once between
    # baseline and sieve passes of each run.
    assert len(resets) >= 3


def test_aggregated_compare_headline_format():
    """The headline should include both raw token counts and the %± stddev."""
    from sieve.cli_benchmark import (
        AggregatedStat, AggregatedCompareSummary, CompareSummary,
        BenchmarkSummary, build_headline,
    )
    # Three identical runs — stddev = 0
    def _make_cs():
        baseline = BenchmarkSummary(
            total_inbound=30000, total_outbound=30000,
            facts_learned=0, trap_absence_signal=None, turns=[],
        )
        sieve = BenchmarkSummary(
            total_inbound=300, total_outbound=4500,
            facts_learned=11, trap_absence_signal=True,
            turns=[None] * 15,
            correct_recalls=6, gradable_recalls=6,
        )
        return CompareSummary(
            baseline_tokens=30000,
            sieve_outbound_tokens=4500,
            sieve_inbound_tokens=300,
            baseline=baseline, sieve=sieve,
        )
    runs = [_make_cs() for _ in range(3)]
    agg = AggregatedCompareSummary(
        runs=runs,
        baseline_tokens=AggregatedStat.from_values([30000.0, 30000.0, 30000.0]),
        sieve_outbound_tokens=AggregatedStat.from_values([4500.0, 4500.0, 4500.0]),
        tokens_saved=AggregatedStat.from_values([25500.0, 25500.0, 25500.0]),
        reduction_pct=AggregatedStat.from_values([85.0, 85.0, 85.0]),
        baseline_wall_clock_s=AggregatedStat.from_values([1.0, 1.0, 1.0]),
        sieve_wall_clock_s=AggregatedStat.from_values([2.0, 2.0, 2.0]),
        correct_recalls_per_run=[6, 6, 6],
        gradable_recalls=6,
        trap_absence_per_run=[True, True, True],
        facts_learned_per_run=[11, 11, 11],
    )
    h = build_headline(aggregated=agg, fixture="medium", model="qwen3.5:9b")
    assert "85%" in h
    assert "30,000" in h and "4,500" in h
    assert "6/6" in h
    assert "trap refused" in h
    assert "medium" in h
    assert "qwen3.5:9b" in h


def test_aggregated_compare_headline_with_stddev():
    """Unequal runs should produce a ± stddev in the headline."""
    from sieve.cli_benchmark import (
        AggregatedStat, AggregatedCompareSummary, CompareSummary,
        BenchmarkSummary, build_headline,
    )
    def _cs(sieve_out):
        baseline = BenchmarkSummary(
            total_inbound=30000, total_outbound=30000,
            facts_learned=0, trap_absence_signal=None, turns=[],
        )
        sieve = BenchmarkSummary(
            total_inbound=300, total_outbound=sieve_out,
            facts_learned=11, trap_absence_signal=True,
            turns=[None] * 15,
            correct_recalls=6, gradable_recalls=6,
        )
        return CompareSummary(
            baseline_tokens=30000, sieve_outbound_tokens=sieve_out,
            sieve_inbound_tokens=300, baseline=baseline, sieve=sieve,
        )
    runs = [_cs(4000), _cs(5000), _cs(4500)]
    pct_vals = [(30000 - s) / 30000 * 100 for s in (4000, 5000, 4500)]
    agg = AggregatedCompareSummary(
        runs=runs,
        baseline_tokens=AggregatedStat.from_values([30000.0] * 3),
        sieve_outbound_tokens=AggregatedStat.from_values([4000.0, 5000.0, 4500.0]),
        tokens_saved=AggregatedStat.from_values([26000.0, 25000.0, 25500.0]),
        reduction_pct=AggregatedStat.from_values(pct_vals),
        baseline_wall_clock_s=AggregatedStat.from_values([1.0] * 3),
        sieve_wall_clock_s=AggregatedStat.from_values([2.0] * 3),
        correct_recalls_per_run=[6, 6, 6],
        gradable_recalls=6,
        trap_absence_per_run=[True, True, True],
        facts_learned_per_run=[11, 11, 11],
    )
    h = build_headline(aggregated=agg, fixture="medium", model="test")
    assert "±" in h  # stddev present


# ── Fixtures ────────────────────────────────────────────────────────────


def test_fixture_registry_has_four_sizes():
    from sieve._agent_fixture import (
        fixture_names, fixture_description, fixture_approx_tokens, fixture_for,
    )
    assert fixture_names() == ["small", "medium", "large", "xlarge"]
    # Each fixture is usable and has a nonzero base size.
    prev = -1
    for name in fixture_names():
        base = fixture_approx_tokens(name)
        assert base >= 0
        # Fixtures grow monotonically (small < medium < large < xlarge).
        assert base >= prev
        prev = base
        desc = fixture_description(name)
        assert isinstance(desc, str) and desc
        builder = fixture_for(name)
        payload = builder("hello", "test-model", history=[])
        assert payload["model"] == "test-model"
        assert payload["messages"][-1]["content"] == "hello"


def test_fixture_sizes_are_meaningfully_different():
    """medium should be ~10× small; xlarge should be ~100× small."""
    from sieve._agent_fixture import fixture_approx_tokens
    small = fixture_approx_tokens("small")
    medium = fixture_approx_tokens("medium")
    large = fixture_approx_tokens("large")
    xlarge = fixture_approx_tokens("xlarge")
    assert medium > small * 3   # real lift, not a rounding difference
    assert large > medium * 3
    assert xlarge > large * 2


def test_fixture_for_unknown_name_raises():
    from sieve._agent_fixture import fixture_for
    with pytest.raises(KeyError):
        fixture_for("jumbo")


# ── Recall range helper ────────────────────────────────────────────────


def test_recall_range_consistent_and_mixed():
    from sieve.cli_benchmark import (
        AggregatedStat, AggregatedCompareSummary, CompareSummary,
        BenchmarkSummary, _recall_range,
    )
    def _mk(recalls: list[int]):
        # minimal shell for _recall_range
        fake_bs = BenchmarkSummary(
            total_inbound=0, total_outbound=0, facts_learned=0,
            trap_absence_signal=None, turns=[],
        )
        fake_cs = CompareSummary(
            baseline_tokens=0, sieve_outbound_tokens=0, sieve_inbound_tokens=0,
            baseline=fake_bs, sieve=fake_bs,
        )
        zero = AggregatedStat.from_values([])
        return AggregatedCompareSummary(
            runs=[fake_cs] * len(recalls),
            baseline_tokens=zero, sieve_outbound_tokens=zero,
            tokens_saved=zero, reduction_pct=zero,
            baseline_wall_clock_s=zero, sieve_wall_clock_s=zero,
            correct_recalls_per_run=recalls,
            gradable_recalls=6,
            trap_absence_per_run=[True] * len(recalls),
            facts_learned_per_run=[0] * len(recalls),
        )
    assert _recall_range(_mk([6, 6, 6])).startswith("6/6")
    # Mixed — should show a range
    r = _recall_range(_mk([4, 6, 5]))
    assert "4/6" in r and "6/6" in r


# ── Markdown report on aggregated summary ──────────────────────────────


def test_render_aggregated_markdown_has_methodology_and_limitations():
    from sieve.cli_benchmark import (
        AggregatedStat, AggregatedCompareSummary, CompareSummary,
        BenchmarkSummary, render_aggregated_markdown,
    )
    base = BenchmarkSummary(
        total_inbound=30000, total_outbound=30000,
        facts_learned=0, trap_absence_signal=None,
        turns=[
            TurnResult(
                index=i, phase=BENCHMARK_MESSAGES[i-1]["phase"],
                prompt=BENCHMARK_MESSAGES[i-1]["content"],
                response="r", inbound_tokens=2000, outbound_tokens=2000,
                facts_before=0, facts_after=0, elapsed_s=1.0, absence_signal=None,
            ) for i in range(1, 16)
        ],
    )
    sie = BenchmarkSummary(
        total_inbound=300, total_outbound=4500,
        facts_learned=11, trap_absence_signal=True,
        correct_recalls=6, gradable_recalls=6,
        turns=[
            TurnResult(
                index=i, phase=BENCHMARK_MESSAGES[i-1]["phase"],
                prompt=BENCHMARK_MESSAGES[i-1]["content"],
                response="You live in Porto." if i == 4 else "response",
                inbound_tokens=200, outbound_tokens=300,
                facts_before=0, facts_after=0, elapsed_s=2.0,
                absence_signal=True if i == 15 else None,
            ) for i in range(1, 16)
        ],
    )
    cs = CompareSummary(
        baseline_tokens=30000, sieve_outbound_tokens=4500,
        sieve_inbound_tokens=300, baseline=base, sieve=sie,
    )
    runs = [cs, cs, cs]
    agg = AggregatedCompareSummary(
        runs=runs,
        baseline_tokens=AggregatedStat.from_values([30000.0] * 3),
        sieve_outbound_tokens=AggregatedStat.from_values([4500.0] * 3),
        tokens_saved=AggregatedStat.from_values([25500.0] * 3),
        reduction_pct=AggregatedStat.from_values([85.0] * 3),
        baseline_wall_clock_s=AggregatedStat.from_values([15.0] * 3),
        sieve_wall_clock_s=AggregatedStat.from_values([30.0] * 3),
        correct_recalls_per_run=[6, 6, 6], gradable_recalls=6,
        trap_absence_per_run=[True, True, True],
        facts_learned_per_run=[11, 11, 11],
    )
    md = render_aggregated_markdown(
        agg, model="qwen3.5:9b", grader_model="qwen3.5:9b",
        fixture="medium", sieve_base_url="http://sieve",
        direct_base_url="http://direct", pricing_tier="claude-sonnet",
        turns_per_run=15,
    )
    # Methodology leads, adjective-free.
    assert "## Methodology" in md
    # Limitations must have the exact heading so skeptics find it.
    assert "## Known limitations" in md
    # Self-grading disclosed loudly when grader == model.
    assert "self-grading" in md.lower()
    # Reproduce command and config signature both present.
    assert "## Reproduce" in md
    assert "Config signature" in md
    # Cost panel fires when pricing tier is paid.
    assert "$" in md
    # No marketing adjectives in the generated text.
    forbidden = ["blazing", "lightning", "industry-leading", "massive"]
    low = md.lower()
    for adj in forbidden:
        assert adj not in low, f"marketing adjective leaked: {adj!r}"

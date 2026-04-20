"""`sieve benchmark` — reproducible proof of token reduction + memory learning.

Runs a 15-message scripted conversation and measures:
  - inbound vs outbound tokens (how much bloat the proxy strips)
  - facts learned per turn (memory is actually being written)
  - correct recalls (the store surfaces the right facts on query turns)
  - absence signal on the trap (no hallucination on unmentioned entities)

Two run modes:

  - **Single-mode** (default): sends each user turn as a bare chat
    message. Cheap to run and demonstrates the memory + recall
    features; the token-reduction number will be negative because
    Sieve is adding context to an empty baseline.
  - **Compare mode** (`--compare`): wraps each turn inside a realistic
    agent-shaped payload (~3K-token system prompt + tool schemas +
    prior history), runs the full script twice — once directly against
    the LLM (baseline), once through Sieve — and reports side-by-side
    token counts. This is the one that demonstrates Sieve's token
    reduction claim honestly.

Designed to work with any OpenAI-compatible or Ollama-compatible LLM
that sieve.yaml is pointed at — the messages do not depend on the model
knowing specific facts, only on the proxy's observable behaviour.

Public entry points:

- ``run_benchmark(...)`` — single-mode runner. Returns a BenchmarkSummary.
- ``run_benchmark_compare(...)`` — two-pass runner. Returns a CompareSummary.
- ``render_summary(...)`` / ``render_compare_summary(...)`` — rich tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import httpx


# ── The 15-message script ─────────────────────────────────────────────────
# Designed to be generic enough for any model:
#   1–3  introduce facts (name, location/work, family)
#   4–8  retrieval queries (questions that need the seeded facts)
#   9–12 deeper follow-ups (build on prior answers)
#   13–14 temporal update (change a fact, then verify the new value)
#   15    trap — ask about a family member that was never mentioned
#
# The trap wording avoids the name "Jordan" appearing earlier in the script
# so any reference in the response is unambiguously a hallucination.

BENCHMARK_MESSAGES: list[dict] = [
    # 1–3: introduce facts
    {"phase": "introduce", "content": "Hi, I'm Sam. I live in Porto, Portugal."},
    {"phase": "introduce", "content": "I work as a marine biologist studying octopus cognition."},
    {"phase": "introduce", "content": "I have a partner named Alex and a rescue cat called Luna."},
    # 4–8: retrieval
    {"phase": "retrieve", "content": "Where do I live again?"},
    {"phase": "retrieve", "content": "What do I do for a living?"},
    {"phase": "retrieve", "content": "What's my cat's name?"},
    {"phase": "retrieve", "content": "Can you remind me what field I research?"},
    {"phase": "retrieve", "content": "Who is my partner?"},
    # 9–12: deeper questions (build on prior answers, no new facts required)
    {"phase": "deep", "content": "Given my job, what's a typical research challenge I might face?"},
    {"phase": "deep", "content": "What kind of climate should I expect year-round where I live?"},
    {"phase": "deep", "content": "If I wanted to take Luna on a walk safely, what should I keep in mind?"},
    {"phase": "deep", "content": "What are some things Alex and I might enjoy doing together nearby?"},
    # 13–14: temporal update
    {"phase": "update", "content": "Actually, I moved last month. I now live in Lisbon, not Porto."},
    {"phase": "update", "content": "Where do I live now?"},
    # 15: trap — never mentioned "Jordan" before
    {"phase": "trap", "content": "What does my sibling Jordan do for work?"},
]


# ── Recall-correctness heuristics ─────────────────────────────────────────
# Per-turn keyword checks for the retrieve/update phases. A response is
# "correct" when at least one of the expected keywords appears (case
# insensitive). Deliberately permissive — models phrase things
# differently and we don't want to fail on "Porto, Portugal" vs "in Porto".
RECALL_EXPECTATIONS: dict[int, tuple[str, ...]] = {
    4: ("porto",),
    5: ("marine biolog", "octopus"),
    6: ("luna",),
    7: ("marine biolog", "octopus"),
    8: ("alex",),
    14: ("lisbon",),
}


def response_recalls(index: int, text: str) -> bool | None:
    """Was the expected fact present in the model's response?

    Returns None for turns we don't grade (introduce / deep / trap).
    """
    expected = RECALL_EXPECTATIONS.get(index)
    if expected is None:
        return None
    low = (text or "").lower()
    return any(kw in low for kw in expected)


# ── Heuristics for absence-signal detection ──────────────────────────────

# Phrases we accept as evidence the model refused / signalled absence rather
# than fabricated. Lowercased; any match counts as a signal.
ABSENCE_PATTERNS: tuple[str, ...] = (
    "don't know",
    "do not know",
    "don't have",
    "do not have",
    "haven't mentioned",
    "have not mentioned",
    "haven't told",
    "have not told",
    "no information",
    "no record",
    "not in my memory",
    "not sure",
    "cannot recall",
    "can't recall",
    "didn't mention",
    "did not mention",
    "you have not",
    "you haven't",
    "i don't see",
    "i do not see",
    "no mention",
    "not present",
)


def looks_like_absence_signal(text: str) -> bool:
    """Fuzzy heuristic: did the model refuse/abstain on the trap?"""
    if not text:
        return False
    low = text.lower()
    return any(p in low for p in ABSENCE_PATTERNS)


# ── Data shapes ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TurnResult:
    """One row of benchmark output — what happened on one message."""

    index: int             # 1-based
    phase: str             # "introduce" / "retrieve" / "deep" / "update" / "trap"
    prompt: str
    response: str
    inbound_tokens: int    # what the proxy received (after payload wrap)
    outbound_tokens: int   # what the proxy sent to the LLM
    facts_before: int
    facts_after: int
    elapsed_s: float
    absence_signal: bool | None  # None except on trap
    correct_recall: bool | None = None  # None except on retrieve/update
    activation_phase: str = ""   # X-Sieve-Phase header, empty if missing


@dataclass(frozen=True)
class BenchmarkSummary:
    total_inbound: int
    total_outbound: int
    facts_learned: int
    trap_absence_signal: bool | None
    turns: list[TurnResult]
    correct_recalls: int = 0
    gradable_recalls: int = 0

    @property
    def reduction_pct(self) -> float:
        """Legacy percentage — honest: zero when outbound > inbound."""
        if self.total_inbound <= 0:
            return 0.0
        if self.total_outbound > self.total_inbound:
            return 0.0
        return (self.total_inbound - self.total_outbound) / self.total_inbound * 100.0


@dataclass(frozen=True)
class CompareSummary:
    """Two-pass summary: baseline (direct) vs Sieve (proxy)."""

    baseline_tokens: int       # tokens sent directly to the LLM (no Sieve)
    sieve_outbound_tokens: int # tokens Sieve forwarded to the LLM
    sieve_inbound_tokens: int  # tokens Sieve received from the "agent"
    baseline: BenchmarkSummary
    sieve: BenchmarkSummary

    @property
    def tokens_saved(self) -> int:
        return self.baseline_tokens - self.sieve_outbound_tokens

    @property
    def reduction_pct(self) -> float:
        if self.baseline_tokens <= 0:
            return 0.0
        return (self.tokens_saved / self.baseline_tokens) * 100.0


# ── Pure core ────────────────────────────────────────────────────────────


def _approx(text: str) -> int:
    """chars/4 fallback — matches sieve's internal approximation."""
    return max(1, len(text or "") // 4)


def _payload_tokens(payload: dict) -> int:
    """Estimate total tokens in an outgoing payload."""
    import json
    return _approx(json.dumps(payload))


def _default_wrap(user_message: str, model: str) -> dict:
    """Single-mode payload: just the user turn."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
    }


def run_benchmark(
    *,
    base_url: str,
    model: str,
    store_fact_count: Callable[[], int],
    transport: httpx.BaseTransport | None = None,
    timeout: float = 90.0,
    messages: Iterable[dict] = BENCHMARK_MESSAGES,
    wrap_payload: Callable[[str, str], dict] | None = None,
    use_sieve_headers: bool = True,
) -> BenchmarkSummary:
    """Drive the scripted conversation through a live endpoint.

    Parameters
    ----------
    base_url
        Target endpoint root. For single-mode / Sieve pass this is the
        Sieve proxy (e.g. http://127.0.0.1:11435). For the baseline
        pass this is the direct LLM endpoint.
    model
        Model name to pass in the payload.
    store_fact_count
        Zero-arg callable returning the current current-fact count.
        Polled before/after each turn.
    transport
        Optional httpx transport override for tests.
    timeout
        Per-request timeout in seconds.
    messages
        Override the script. Default is BENCHMARK_MESSAGES.
    wrap_payload
        Callable ``(user_message, model) -> dict`` that builds the
        outgoing payload. Default is a bare user-turn (single mode).
        For compare mode pass ``build_agent_payload`` from
        ``sieve._agent_fixture``.
    use_sieve_headers
        When True, inbound/outbound token counts are read from
        ``X-Sieve-Inbound-Tokens`` / ``X-Sieve-Outbound-Tokens``
        response headers (the authoritative source when the endpoint
        is the Sieve proxy). When False (baseline runs hitting the
        LLM directly), both counts are computed client-side from the
        outgoing payload, and outbound == inbound.
    """
    results: list[TurnResult] = []
    messages = list(messages)
    wrap = wrap_payload or _default_wrap

    initial_facts = store_fact_count()

    client_kwargs = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport

    with httpx.Client(**client_kwargs) as client:
        for i, msg in enumerate(messages, start=1):
            facts_before = store_fact_count()
            payload = wrap(msg["content"], model)
            import time
            t0 = time.perf_counter()
            r = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()

            data = r.json() if r.content else {}
            response_text = ((data.get("message") or {}).get("content") or "").strip()

            if use_sieve_headers:
                inbound = int(r.headers.get("X-Sieve-Inbound-Tokens", "0")) or _payload_tokens(payload)
                outbound = int(r.headers.get("X-Sieve-Outbound-Tokens", "0"))
            else:
                # Direct-to-LLM baseline — no Sieve headers.
                tokens = _payload_tokens(payload)
                inbound = tokens
                outbound = tokens
            activation_phase = r.headers.get("X-Sieve-Phase", "")

            facts_after = store_fact_count()

            absence = None
            if msg["phase"] == "trap":
                absence = looks_like_absence_signal(response_text)
            correct = response_recalls(i, response_text)

            results.append(TurnResult(
                index=i,
                phase=msg["phase"],
                prompt=msg["content"],
                response=response_text,
                inbound_tokens=inbound,
                outbound_tokens=outbound,
                facts_before=facts_before,
                facts_after=facts_after,
                elapsed_s=elapsed,
                absence_signal=absence,
                correct_recall=correct,
                activation_phase=activation_phase,
            ))

    total_inbound = sum(t.inbound_tokens for t in results)
    total_outbound = sum(t.outbound_tokens for t in results)
    final_facts = store_fact_count()
    trap_signal = next(
        (t.absence_signal for t in results if t.phase == "trap"),
        None,
    )
    gradable = [t for t in results if t.correct_recall is not None]
    correct = sum(1 for t in gradable if t.correct_recall)

    return BenchmarkSummary(
        total_inbound=total_inbound,
        total_outbound=total_outbound,
        facts_learned=max(0, final_facts - initial_facts),
        correct_recalls=correct,
        gradable_recalls=len(gradable),
        trap_absence_signal=trap_signal,
        turns=results,
    )


def run_benchmark_compare(
    *,
    sieve_base_url: str,
    direct_base_url: str,
    model: str,
    store_fact_count: Callable[[], int],
    reset_store: Callable[[], None],
    transport: httpx.BaseTransport | None = None,
    timeout: float = 120.0,
    messages: Iterable[dict] = BENCHMARK_MESSAGES,
) -> CompareSummary:
    """Two-pass benchmark: baseline (direct) then Sieve (proxy).

    Both passes send the same agent-shaped payload per turn. The
    baseline pass shows what an uninstrumented agent costs per turn;
    the Sieve pass shows what Sieve compresses it to. The delta is
    the honest "tokens saved" number.

    Parameters
    ----------
    sieve_base_url
        The Sieve proxy root.
    direct_base_url
        The LLM endpoint the proxy forwards to (e.g. Ollama at 11434).
    reset_store
        Called between the two passes to clear any facts written
        during the baseline so the Sieve pass starts from the same
        state. Required — the store is a side channel that would
        otherwise pollute the Sieve pass's "facts learned" count.
    """
    from sieve._agent_fixture import build_agent_payload

    # Pass 1 — baseline, direct to LLM.
    baseline_summary = run_benchmark(
        base_url=direct_base_url,
        model=model,
        store_fact_count=store_fact_count,
        transport=transport,
        timeout=timeout,
        messages=messages,
        wrap_payload=build_agent_payload,
        use_sieve_headers=False,
    )

    # Clear any bleed-through between passes. The baseline pass
    # doesn't go through Sieve, so no facts should have been written
    # — but reset_store is cheap insurance against misconfiguration.
    reset_store()

    # Pass 2 — through Sieve proxy.
    sieve_summary = run_benchmark(
        base_url=sieve_base_url,
        model=model,
        store_fact_count=store_fact_count,
        transport=transport,
        timeout=timeout,
        messages=messages,
        wrap_payload=build_agent_payload,
        use_sieve_headers=True,
    )

    return CompareSummary(
        baseline_tokens=baseline_summary.total_inbound,
        sieve_outbound_tokens=sieve_summary.total_outbound,
        sieve_inbound_tokens=sieve_summary.total_inbound,
        baseline=baseline_summary,
        sieve=sieve_summary,
    )


# ── Rendering (rich) ─────────────────────────────────────────────────────


def _format_cut(inbound: int, outbound: int) -> str:
    """Render the per-row token-delta column honestly.

    When outbound < inbound: show the positive reduction percentage.
    When outbound > inbound: show the overhead in tokens, NOT a
        negative percentage (misleading).
    When inbound == 0 or missing: show em-dash.
    """
    if inbound <= 0:
        return "—"
    if outbound <= inbound:
        pct = (inbound - outbound) / inbound * 100.0
        return f"{pct:.0f}%"
    overhead = outbound - inbound
    return f"[dim]+{overhead}t[/]"


def render_summary(
    summary: BenchmarkSummary,
    *,
    model: str,
    base_url: str,
    console,
) -> None:
    """Pretty-print single-mode benchmark output.

    Leads with the memory-learning story (facts, correct recalls,
    trap). Token reduction is reported honestly — no negative
    percentages — but demoted to a secondary panel since single-mode
    doesn't showcase Sieve's compression claim.
    """
    from rich.table import Table
    from rich.panel import Panel

    table = Table(title="Per-message breakdown", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Phase")
    table.add_column("Sieve")
    table.add_column("Prompt", overflow="fold", max_width=38)
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("Facts", justify="right")
    table.add_column("Recall", justify="center")
    table.add_column("Time", justify="right")

    prev_activation = ""
    for t in summary.turns:
        cut = _format_cut(t.inbound_tokens, t.outbound_tokens)
        facts_delta = t.facts_after - t.facts_before
        if facts_delta > 0:
            facts = f"{t.facts_after} [green](+{facts_delta})[/]"
        else:
            facts = f"{t.facts_after}"
        phase_label = {
            "introduce": "intro",
            "retrieve": "recall",
            "deep": "deep",
            "update": "update",
            "trap": "[bold yellow]trap[/]",
        }.get(t.phase, t.phase)
        ap = (t.activation_phase or "").upper()
        activation_badge = {
            "OBSERVE": "[cyan]obs[/]",
            "ACCUMULATE": "[yellow]acc[/]",
            "ACTIVATE": "[green]act[/]",
        }.get(ap, "[dim]—[/]")
        if ap and ap != prev_activation and prev_activation:
            activation_badge = f"{activation_badge} [bold]↗[/]"
        if ap:
            prev_activation = ap
        # Recall correctness cell: ✓ / ✗ / blank.
        if t.correct_recall is True:
            recall_cell = "[green]✓[/]"
        elif t.correct_recall is False:
            recall_cell = "[red]✗[/]"
        elif t.phase == "trap":
            recall_cell = (
                "[green]✓[/]" if t.absence_signal else "[red]✗[/]"
            )
        else:
            recall_cell = ""
        table.add_row(
            str(t.index),
            phase_label,
            activation_badge,
            t.prompt,
            str(t.inbound_tokens),
            str(t.outbound_tokens),
            cut,
            facts,
            recall_cell,
            f"{t.elapsed_s:.1f}s",
        )

    console.print(table)

    transitions: list[tuple[int, str, str]] = []
    last = ""
    for t in summary.turns:
        ap = (t.activation_phase or "").upper()
        if not ap:
            continue
        if last and ap != last:
            transitions.append((t.index, last, ap))
        last = ap
    if transitions:
        console.print(
            "[bold]Phase transitions:[/] "
            + "  ".join(
                f"turn {idx}: [dim]{src}[/] → [bold]{dst}[/]"
                for idx, src, dst in transitions
            )
        )

    # Lead with the memory-learning story.
    lines = [
        f"[bold]Model:[/]  [cyan]{model}[/]",
        f"[bold]Proxy:[/]  [cyan]{base_url}[/]",
        "",
        f"[bold]Facts learned:[/]       {summary.facts_learned}",
    ]
    if summary.gradable_recalls:
        lines.append(
            f"[bold]Correct recalls:[/]     "
            f"[green]{summary.correct_recalls}/{summary.gradable_recalls}[/]"
        )
    if summary.trap_absence_signal is True:
        lines.append("[bold]Trap query:[/]          [green]absence signal fired ✓[/]")
    elif summary.trap_absence_signal is False:
        lines.append("[bold]Trap query:[/]          [red]no absence signal detected[/]")
    else:
        lines.append("[bold]Trap query:[/]          (not reached)")
    console.print(
        Panel("\n".join(lines), title="Benchmark summary", border_style="cyan")
    )

    # Token panel — demoted, honest framing. In single-mode the
    # inbound is tiny (bare user turns) so we report the proxy overhead
    # per turn rather than a misleading reduction percentage.
    avg_overhead = (
        (summary.total_outbound - summary.total_inbound) / len(summary.turns)
        if summary.turns else 0.0
    )
    if avg_overhead > 0:
        token_lines = [
            f"[dim]Inbound (bare chat):[/]   {summary.total_inbound:,} tokens",
            f"[dim]Outbound (with Sieve):[/] {summary.total_outbound:,} tokens",
            f"[dim]Proxy overhead:[/]        ~{avg_overhead:.0f} tokens / turn",
            "",
            "[dim]Single-mode sends bare user turns; Sieve adds its lean prompt",
            "and recall tool. To see reduction vs a real agent payload, run:[/]",
            "  [cyan]sieve benchmark --compare[/]",
        ]
    else:
        reduction = (
            (summary.total_inbound - summary.total_outbound)
            / summary.total_inbound * 100.0
            if summary.total_inbound else 0.0
        )
        token_lines = [
            f"Inbound:   {summary.total_inbound:,} tokens",
            f"Outbound:  {summary.total_outbound:,} tokens",
            f"Reduction: [green]{reduction:.1f}%[/]",
        ]
    console.print(
        Panel(
            "\n".join(token_lines),
            title="Tokens",
            border_style="dim",
        )
    )
    console.print("\n[dim]For token-reduction proof: [cyan]sieve benchmark --compare[/][/]")


def render_compare_summary(
    summary: CompareSummary,
    *,
    model: str,
    sieve_base_url: str,
    direct_base_url: str,
    console,
) -> None:
    """Side-by-side rendering for --compare runs.

    The headline is tokens-saved. The memory-features story lives in
    a secondary panel since it's primarily a property of the Sieve
    pass.
    """
    from rich.table import Table
    from rich.panel import Panel

    # Side-by-side totals.
    saved = summary.tokens_saved
    pct = summary.reduction_pct

    token_table = Table(title="Baseline vs Sieve", show_lines=False)
    token_table.add_column("Metric")
    token_table.add_column("Baseline (direct)", justify="right")
    token_table.add_column("With Sieve", justify="right")
    token_table.add_column("Δ", justify="right")
    token_table.add_row(
        "Tokens sent to LLM",
        f"{summary.baseline_tokens:,}",
        f"{summary.sieve_outbound_tokens:,}",
        (
            f"[green]−{saved:,} ({pct:.1f}%)[/]" if saved > 0
            else f"[red]+{-saved:,}[/]"
        ),
    )
    baseline_avg_latency = (
        sum(t.elapsed_s for t in summary.baseline.turns) / len(summary.baseline.turns)
        if summary.baseline.turns else 0.0
    )
    sieve_avg_latency = (
        sum(t.elapsed_s for t in summary.sieve.turns) / len(summary.sieve.turns)
        if summary.sieve.turns else 0.0
    )
    token_table.add_row(
        "Avg latency / turn",
        f"{baseline_avg_latency:.1f}s",
        f"{sieve_avg_latency:.1f}s",
        f"{(sieve_avg_latency - baseline_avg_latency):+.1f}s",
    )
    console.print(token_table)

    # Memory features — Sieve-only (baseline has no store).
    recall_line = (
        f"[green]{summary.sieve.correct_recalls}/{summary.sieve.gradable_recalls}[/]"
        if summary.sieve.gradable_recalls else "—"
    )
    trap_line = (
        "[green]absence signal fired ✓[/]"
        if summary.sieve.trap_absence_signal is True
        else "[red]no absence signal[/]"
        if summary.sieve.trap_absence_signal is False
        else "(not reached)"
    )
    console.print(
        Panel(
            "\n".join([
                f"[bold]Facts learned:[/]       {summary.sieve.facts_learned}",
                f"[bold]Correct recalls:[/]     {recall_line}",
                f"[bold]Trap query:[/]          {trap_line}",
            ]),
            title="Memory features (Sieve pass)",
            border_style="cyan",
        )
    )

    # Summary panel.
    headline_lines = [
        f"[bold]Model:[/]     [cyan]{model}[/]",
        f"[bold]Baseline:[/]  [cyan]{direct_base_url}[/] (no Sieve)",
        f"[bold]Sieve:[/]     [cyan]{sieve_base_url}[/]",
        "",
    ]
    if saved > 0:
        headline_lines.extend([
            f"[bold green]Sieve cut {saved:,} tokens per run ({pct:.1f}%)[/]",
            f"[dim]That's {saved / len(summary.sieve.turns):.0f} fewer tokens per turn[/]"
            f"[dim] on a 15-turn coding conversation.[/]",
        ])
    else:
        headline_lines.append(
            f"[bold red]Sieve added {-saved:,} tokens this run[/] "
            "— check fixture sizing."
        )
    console.print(
        Panel(
            "\n".join(headline_lines),
            title="Benchmark summary",
            border_style="green" if saved > 0 else "red",
        )
    )
    console.print(
        "\n[dim]Run this yourself: [cyan]sieve benchmark --compare[/][/]"
    )

"""`sieve benchmark` — reproducible proof of token reduction + memory learning.

Runs a 15-message scripted conversation against a running Sieve proxy and
measures inbound vs outbound tokens, facts learned per turn, response time,
and whether the absence-signal layer fires on a trap query.

Designed to work with any OpenAI-compatible or Ollama-compatible LLM that
sieve.yaml is pointed at — the messages do not depend on the model knowing
specific facts, only on the proxy's observable behaviour.

Public entry points:

- ``run_benchmark(base_url, model, store_fact_count, transport=None)`` —
  pure-ish core used by the CLI and the tests. Yields ``TurnResult`` rows.
- ``render_summary(results, owner_name, fact_count_before, fact_count_after)``
  — builds the rich summary table.

The CLI wrapper in ``cli.py`` handles config loading, proxy discovery, and
printing.
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
    """Fuzzy heuristic: did the model refuse/abstain on the trap?

    This is conservative — a clean refusal matches, anything that *might*
    be fabrication (names a job, describes Jordan, etc.) does not.
    """
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
    inbound_tokens: int    # what the proxy received
    outbound_tokens: int   # what the proxy sent to the LLM
    facts_before: int
    facts_after: int
    elapsed_s: float
    absence_signal: bool | None  # None except on trap; True/False on trap
    # Progressive-activation phase reported by the proxy for this request.
    # Empty string on older proxies that don't emit the X-Sieve-Phase header.
    activation_phase: str = ""


@dataclass(frozen=True)
class BenchmarkSummary:
    total_inbound: int
    total_outbound: int
    reduction_pct: float
    facts_learned: int
    trap_absence_signal: bool | None
    turns: list[TurnResult]


# ── Pure core ────────────────────────────────────────────────────────────


def run_benchmark(
    *,
    base_url: str,
    model: str,
    store_fact_count: Callable[[], int],
    transport: httpx.BaseTransport | None = None,
    timeout: float = 90.0,
    messages: Iterable[dict] = BENCHMARK_MESSAGES,
) -> BenchmarkSummary:
    """Drive the scripted conversation through a live proxy.

    Parameters
    ----------
    base_url : str
        The Sieve proxy root (e.g. ``http://127.0.0.1:11435``).
    model : str
        Model name to pass in the payload. Whatever the proxy is pointed at.
    store_fact_count : Callable[[], int]
        Zero-arg callable returning the current number of current facts in
        the store. Queried before and after each turn so we can attribute
        learning to messages. In the CLI this reads ``MemoryStore.stats``;
        tests inject a counter.
    transport : httpx.BaseTransport, optional
        Override httpx transport for tests (MockTransport). Default: real HTTP.
    timeout : float
        Per-request timeout in seconds.
    messages : iterable of {"phase": str, "content": str}
        Override the script. Default is BENCHMARK_MESSAGES.
    """
    results: list[TurnResult] = []
    messages = list(messages)

    initial_facts = store_fact_count()

    client_kwargs = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport

    with httpx.Client(**client_kwargs) as client:
        for i, msg in enumerate(messages, start=1):
            facts_before = store_fact_count()
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": msg["content"]}],
                "stream": False,
            }
            import time
            t0 = time.perf_counter()
            r = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
            elapsed = time.perf_counter() - t0
            r.raise_for_status()

            data = r.json() if r.content else {}
            response_text = ((data.get("message") or {}).get("content") or "").strip()

            # Server-computed counts when available; fall back to client-side
            # approximation so the benchmark still works if the user's proxy
            # predates the header.
            inbound = int(r.headers.get("X-Sieve-Inbound-Tokens", "0")) or _approx(msg["content"])
            outbound = int(r.headers.get("X-Sieve-Outbound-Tokens", "0"))
            activation_phase = r.headers.get("X-Sieve-Phase", "")

            facts_after = store_fact_count()

            absence = None
            if msg["phase"] == "trap":
                absence = looks_like_absence_signal(response_text)

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
                activation_phase=activation_phase,
            ))

    total_inbound = sum(t.inbound_tokens for t in results)
    total_outbound = sum(t.outbound_tokens for t in results)
    reduction_pct = (
        (total_inbound - total_outbound) / total_inbound * 100.0
        if total_inbound else 0.0
    )
    final_facts = store_fact_count()
    trap_signal = next(
        (t.absence_signal for t in results if t.phase == "trap"),
        None,
    )

    return BenchmarkSummary(
        total_inbound=total_inbound,
        total_outbound=total_outbound,
        reduction_pct=reduction_pct,
        facts_learned=max(0, final_facts - initial_facts),
        trap_absence_signal=trap_signal,
        turns=results,
    )


def _approx(text: str) -> int:
    """Fallback client-side token count (chars/4) — matches sieve's own approximation."""
    return max(1, len(text or "") // 4)


# ── Rendering (rich) ─────────────────────────────────────────────────────


def render_summary(
    summary: BenchmarkSummary,
    *,
    model: str,
    base_url: str,
    console,
) -> None:
    """Pretty-print the per-turn table + totals to a rich Console.

    Kept separate from run_benchmark so tests can assert on summary values
    without needing to parse rendered output.
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
    table.add_column("Cut", justify="right")
    table.add_column("Facts", justify="right")
    table.add_column("Time", justify="right")

    # Track phase transitions so we can annotate where OBSERVE →
    # ACCUMULATE → ACTIVATE happen in the conversation. The transition
    # column is the empty string on all rows except the first one of a
    # new phase; this keeps the table readable while making the shift
    # obvious.
    prev_activation = ""
    for t in summary.turns:
        cut = "—"
        if t.inbound_tokens:
            pct = (t.inbound_tokens - t.outbound_tokens) / t.inbound_tokens * 100
            cut = f"{pct:+.0f}%" if pct < 0 else f"{pct:.0f}%"
        facts_delta = t.facts_after - t.facts_before
        # Always render the fact count so readers can see the store
        # growing message-by-message, not only on gain turns. A plain
        # number reads cleanly next to a coloured delta.
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
        # Short, colour-coded activation-phase badge. Older proxies
        # without progressive activation return an empty header; we
        # render "—" so the column stays visually aligned.
        ap = (t.activation_phase or "").upper()
        activation_badge = {
            "OBSERVE": "[cyan]obs[/]",
            "ACCUMULATE": "[yellow]acc[/]",
            "ACTIVATE": "[green]act[/]",
        }.get(ap, "[dim]—[/]")
        # Decorate the badge with an → marker on the first message of a
        # new phase so the transition is visually obvious without a
        # dedicated transitions panel.
        if ap and ap != prev_activation and prev_activation:
            activation_badge = f"{activation_badge} [bold]↗[/]"
        if ap:
            prev_activation = ap
        table.add_row(
            str(t.index),
            phase_label,
            activation_badge,
            t.prompt,
            str(t.inbound_tokens),
            str(t.outbound_tokens),
            cut,
            facts,
            f"{t.elapsed_s:.1f}s",
        )

    console.print(table)

    # Phase-transition summary — a dedicated one-liner list so the
    # transitions are discoverable at a glance even if the table badge
    # is missed. Only emitted when the proxy reported at least one
    # phase transition during the run.
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

    # Per-phase reduction breakdown — makes the phase transitions
    # visible at a glance. Skipped when the proxy doesn't emit the
    # X-Sieve-Phase header (older builds).
    phase_rows: dict[str, tuple[int, int, int]] = {}
    for t in summary.turns:
        if not t.activation_phase:
            continue
        key = t.activation_phase.upper()
        count, inb, outb = phase_rows.get(key, (0, 0, 0))
        phase_rows[key] = (count + 1, inb + t.inbound_tokens, outb + t.outbound_tokens)

    if phase_rows:
        # Render in phase order: OBSERVE → ACCUMULATE → ACTIVATE, then any others.
        order = ["OBSERVE", "ACCUMULATE", "ACTIVATE"]
        ordered_keys = [k for k in order if k in phase_rows] + \
            [k for k in phase_rows if k not in order]
        phase_table = Table(title="Per-phase reduction", show_lines=False)
        phase_table.add_column("Phase")
        phase_table.add_column("Messages", justify="right")
        phase_table.add_column("Inbound", justify="right")
        phase_table.add_column("Outbound", justify="right")
        phase_table.add_column("Reduction", justify="right")
        for key in ordered_keys:
            count, inb, outb = phase_rows[key]
            reduction = (inb - outb) / inb * 100.0 if inb else 0.0
            phase_table.add_row(
                key,
                str(count),
                f"{inb:,}",
                f"{outb:,}",
                f"{reduction:.1f}%",
            )
        console.print(phase_table)

    # Overall summary panel
    lines = [
        f"[bold]Model:[/]  [cyan]{model}[/]",
        f"[bold]Proxy:[/]  [cyan]{base_url}[/]",
        "",
        f"[bold]Total inbound tokens:[/]   {summary.total_inbound:,}",
        f"[bold]Total outbound tokens:[/]  {summary.total_outbound:,}",
        f"[bold]Overall reduction:[/]      [green]{summary.reduction_pct:.1f}%[/]",
        f"[bold]Facts learned:[/]          {summary.facts_learned}",
    ]
    if summary.trap_absence_signal is True:
        lines.append("[bold]Trap query:[/]             [green]absence signal fired[/] ✓")
    elif summary.trap_absence_signal is False:
        lines.append("[bold]Trap query:[/]             [red]no absence signal detected[/]")
    else:
        lines.append("[bold]Trap query:[/]             (not reached)")

    console.print(Panel("\n".join(lines), title="Benchmark summary", border_style="cyan"))
    console.print(
        "\n[dim]Run this benchmark yourself: [cyan]sieve benchmark[/][/]"
    )

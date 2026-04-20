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
    ttft_s: float = 0.0          # time to first streamed content token
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
class AggregatedStat:
    """Mean ± sample stddev over N observations.

    Skeptics read single-point benchmarks as marketing material. The
    ±stddev notation is a tribal marker of seriousness — we report
    both so a reader who cares about statistical significance can
    judge for themselves whether the delta cleared the noise floor.
    """
    mean: float
    stddev: float
    n: int

    @classmethod
    def from_values(cls, values: list[float]) -> "AggregatedStat":
        if not values:
            return cls(mean=0.0, stddev=0.0, n=0)
        n = len(values)
        mean = sum(values) / n
        if n < 2:
            return cls(mean=mean, stddev=0.0, n=n)
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return cls(mean=mean, stddev=variance ** 0.5, n=n)

    def render(self, fmt: str = "{:,.0f}") -> str:
        """Human-readable rendering, e.g. '14,581 ± 389' or '14,581'."""
        if self.n <= 1 or self.stddev == 0:
            return fmt.format(self.mean)
        return f"{fmt.format(self.mean)} ± {fmt.format(self.stddev)}"


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


def _default_wrap(user_message: str, model: str, history: list[dict], stream: bool) -> dict:
    """Single-mode payload: history + user turn (no system prompt, no tools)."""
    messages: list[dict] = list(history or [])
    messages.append({"role": "user", "content": user_message})
    return {
        "model": model,
        "messages": messages,
        "stream": stream,
    }


def _read_streamed_response(
    r: httpx.Response, t_start: float
) -> tuple[str, float]:
    """Consume an Ollama NDJSON stream. Returns (full_text, ttft_s).

    ``ttft_s`` is the wall-clock from request-send to first non-empty
    content chunk. When the upstream never emits content (unlikely on
    an Ollama proxy), ttft_s == total_elapsed.
    """
    import json
    import time

    pieces: list[str] = []
    ttft: float | None = None
    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Ollama stream shape: {"message": {"role":"assistant","content": "…"}, "done": false}
        msg = obj.get("message") or {}
        chunk = (msg.get("content") or "")
        if chunk:
            if ttft is None:
                ttft = time.perf_counter() - t_start
            pieces.append(chunk)
        if obj.get("done"):
            break
    total = time.perf_counter() - t_start
    return ("".join(pieces)).strip(), (ttft if ttft is not None else total)


def run_benchmark(
    *,
    base_url: str,
    model: str,
    store_fact_count: Callable[[], int],
    transport: httpx.BaseTransport | None = None,
    timeout: float = 120.0,
    messages: Iterable[dict] = BENCHMARK_MESSAGES,
    wrap_payload: Callable[[str, str, list[dict], bool], dict] | None = None,
    use_sieve_headers: bool = True,
    stream: bool = True,
    grade_recall: Callable[[int, str, str, str], bool | None] | None = None,
    grade_trap: Callable[[str, str], bool] | None = None,
) -> BenchmarkSummary:
    """Drive the scripted conversation through a live endpoint.

    Threads history across turns: turn N's request includes all prior
    user+assistant messages from this run. This is what a real
    conversational agent does. On the baseline pass the context grows
    linearly with turn count; the Sieve pass relies on the proxy to
    compress it back down.

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
        Callable ``(user_message, model, history, stream) -> dict``
        that builds the outgoing payload. ``history`` is the
        accumulated user+assistant messages from prior turns in this
        run. Default is a bare message-list (single mode). For compare
        mode pass ``build_agent_payload`` from ``sieve._agent_fixture``.
    use_sieve_headers
        When True, inbound/outbound token counts are read from
        ``X-Sieve-Inbound-Tokens`` / ``X-Sieve-Outbound-Tokens``
        response headers (the authoritative source when the endpoint
        is the Sieve proxy). When False (baseline runs hitting the
        LLM directly), both counts are computed client-side from the
        outgoing payload, and outbound == inbound.
    stream
        When True, requests streamed responses (Ollama NDJSON /
        OpenAI SSE) and measures time-to-first-token. Default True.
        Tests may pass False with MockTransport.
    grade_recall
        Optional ``(turn_index, prompt, response, expected_hint) ->
        bool|None`` grader. Returns None for non-gradable turns
        (introduce / deep). Default is the keyword heuristic
        ``response_recalls``.
    grade_trap
        Optional ``(prompt, response) -> bool`` grader for the trap
        turn. Default is the keyword heuristic
        ``looks_like_absence_signal``.
    """
    results: list[TurnResult] = []
    messages = list(messages)
    wrap = wrap_payload or _default_wrap

    initial_facts = store_fact_count()

    client_kwargs = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport

    # Accumulated conversation history for this run — grows after
    # every completed turn. Passed to ``wrap`` so each request sees
    # the full prior exchange (the realistic agent shape).
    history: list[dict] = []

    with httpx.Client(**client_kwargs) as client:
        for i, msg in enumerate(messages, start=1):
            facts_before = store_fact_count()
            payload = wrap(msg["content"], model, history, stream)
            import time
            t0 = time.perf_counter()
            if stream:
                req = client.build_request(
                    "POST", f"{base_url.rstrip('/')}/api/chat", json=payload,
                )
                r = client.send(req, stream=True)
                try:
                    r.raise_for_status()
                    response_text, ttft = _read_streamed_response(r, t0)
                finally:
                    r.close()
                elapsed = time.perf_counter() - t0
            else:
                r = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
                elapsed = time.perf_counter() - t0
                r.raise_for_status()
                data = r.json() if r.content else {}
                response_text = ((data.get("message") or {}).get("content") or "").strip()
                ttft = elapsed

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

            # Grade trap / recall.
            absence = None
            if msg["phase"] == "trap":
                grader = grade_trap or (lambda _p, resp: looks_like_absence_signal(resp))
                absence = grader(msg["content"], response_text)

            # Recall grader: for retrieve/update turns. Default is the
            # keyword heuristic; LLM-based grader is injected by the CLI.
            if grade_recall is not None:
                # Hint is a short description of the expected answer,
                # built from RECALL_EXPECTATIONS when available.
                hint = " / ".join(RECALL_EXPECTATIONS.get(i, ()))
                correct = grade_recall(i, msg["content"], response_text, hint)
            else:
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
                ttft_s=ttft,
                absence_signal=absence,
                correct_recall=correct,
                activation_phase=activation_phase,
            ))

            # Append this turn to history for the NEXT request. Real
            # agents do this; it's what drives baseline bloat.
            history.append({"role": "user", "content": msg["content"]})
            if response_text:
                history.append({"role": "assistant", "content": response_text})

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
    stream: bool = True,
    grade_recall: Callable[[int, str, str, str], bool | None] | None = None,
    grade_trap: Callable[[str, str], bool] | None = None,
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

    def _agent_wrap(user: str, mdl: str, history: list[dict], strm: bool) -> dict:
        return build_agent_payload(user, mdl, history=history, stream=strm)

    # Pass 1 — baseline, direct to LLM.
    baseline_summary = run_benchmark(
        base_url=direct_base_url,
        model=model,
        store_fact_count=store_fact_count,
        transport=transport,
        timeout=timeout,
        messages=messages,
        wrap_payload=_agent_wrap,
        use_sieve_headers=False,
        stream=stream,
        grade_recall=grade_recall,
        grade_trap=grade_trap,
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
        wrap_payload=_agent_wrap,
        use_sieve_headers=True,
        stream=stream,
        grade_recall=grade_recall,
        grade_trap=grade_trap,
    )

    return CompareSummary(
        baseline_tokens=baseline_summary.total_inbound,
        sieve_outbound_tokens=sieve_summary.total_outbound,
        sieve_inbound_tokens=sieve_summary.total_inbound,
        baseline=baseline_summary,
        sieve=sieve_summary,
    )


@dataclass(frozen=True)
class AggregatedCompareSummary:
    """N runs of --compare aggregated with mean ± stddev per metric.

    Surfaces the same metrics as CompareSummary but as AggregatedStat
    bundles. The underlying per-run results are kept in ``runs`` so
    downstream tooling can drill in if needed.
    """

    runs: list[CompareSummary]
    baseline_tokens: AggregatedStat
    sieve_outbound_tokens: AggregatedStat
    tokens_saved: AggregatedStat
    reduction_pct: AggregatedStat
    baseline_wall_clock_s: AggregatedStat
    sieve_wall_clock_s: AggregatedStat
    # Correctness metrics — per-run values then consensus.
    correct_recalls_per_run: list[int]
    gradable_recalls: int
    trap_absence_per_run: list[bool | None]
    facts_learned_per_run: list[int]
    # Context-window truncation finding. Populated by the CLI when a
    # baseline inbound approaches or exceeds a known local-model
    # ceiling; surfaced in the report's methodology section.
    baseline_truncation_observed: bool = False

    @property
    def most_recent(self) -> CompareSummary:
        """The last run — used where only point-in-time data is needed."""
        return self.runs[-1]

    @property
    def all_recalls_perfect(self) -> bool:
        if not self.correct_recalls_per_run or self.gradable_recalls == 0:
            return False
        return all(c == self.gradable_recalls for c in self.correct_recalls_per_run)

    @property
    def all_traps_refused(self) -> bool:
        return bool(self.trap_absence_per_run) and all(
            x is True for x in self.trap_absence_per_run
        )


def run_benchmark_compare_multi(
    *,
    runs: int = 3,
    sieve_base_url: str,
    direct_base_url: str,
    model: str,
    store_fact_count: Callable[[], int],
    reset_store: Callable[[], None],
    transport: httpx.BaseTransport | None = None,
    timeout: float = 120.0,
    messages: Iterable[dict] = BENCHMARK_MESSAGES,
    stream: bool = True,
    grade_recall: Callable[[int, str, str, str], bool | None] | None = None,
    grade_trap: Callable[[str, str], bool] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> AggregatedCompareSummary:
    """Run ``--compare`` ``runs`` times and aggregate with mean ± stddev.

    ``progress`` is an optional ``(run_index, total_runs, phase_label) ->
    None`` callback invoked before each of the 2N passes so the CLI
    can show progress. ``phase_label`` is 'baseline' or 'sieve'.
    """
    messages = list(messages)
    per_run: list[CompareSummary] = []
    for i in range(runs):
        reset_store()
        if progress:
            progress(i + 1, runs, "baseline")
        compare = run_benchmark_compare(
            sieve_base_url=sieve_base_url,
            direct_base_url=direct_base_url,
            model=model,
            store_fact_count=store_fact_count,
            reset_store=reset_store,
            transport=transport,
            timeout=timeout,
            messages=messages,
            stream=stream,
            grade_recall=grade_recall,
            grade_trap=grade_trap,
        )
        per_run.append(compare)
        if progress:
            progress(i + 1, runs, "done")

    baseline_vals = [float(r.baseline_tokens) for r in per_run]
    sieve_out_vals = [float(r.sieve_outbound_tokens) for r in per_run]
    saved_vals = [float(r.tokens_saved) for r in per_run]
    pct_vals = [r.reduction_pct for r in per_run]
    baseline_wall = [
        sum(t.elapsed_s for t in r.baseline.turns) for r in per_run
    ]
    sieve_wall = [sum(t.elapsed_s for t in r.sieve.turns) for r in per_run]

    return AggregatedCompareSummary(
        runs=per_run,
        baseline_tokens=AggregatedStat.from_values(baseline_vals),
        sieve_outbound_tokens=AggregatedStat.from_values(sieve_out_vals),
        tokens_saved=AggregatedStat.from_values(saved_vals),
        reduction_pct=AggregatedStat.from_values(pct_vals),
        baseline_wall_clock_s=AggregatedStat.from_values(baseline_wall),
        sieve_wall_clock_s=AggregatedStat.from_values(sieve_wall),
        correct_recalls_per_run=[r.sieve.correct_recalls for r in per_run],
        gradable_recalls=per_run[0].sieve.gradable_recalls if per_run else 0,
        trap_absence_per_run=[r.sieve.trap_absence_signal for r in per_run],
        facts_learned_per_run=[r.sieve.facts_learned for r in per_run],
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
    avg_ttft = _avg([t.ttft_s for t in summary.turns if t.ttft_s > 0])
    if avg_overhead > 0:
        token_lines = [
            f"[dim]Inbound (bare chat):[/]   {summary.total_inbound:,} tokens",
            f"[dim]Outbound (with Sieve):[/] {summary.total_outbound:,} tokens",
            f"[dim]Proxy overhead:[/]        ~{avg_overhead:.0f} tokens / turn",
        ]
        if avg_ttft > 0:
            token_lines.append(f"[dim]Avg TTFT:[/]              {avg_ttft:.1f}s")
        token_lines += [
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
        if avg_ttft > 0:
            token_lines.append(f"Avg TTFT:  {avg_ttft:.1f}s")
    console.print(
        Panel(
            "\n".join(token_lines),
            title="Tokens",
            border_style="dim",
        )
    )

    # Headline — plain text, paste-able into READMEs / reports.
    console.print(
        f"\n[bold]{build_headline(single=summary)}[/]"
    )
    console.print(
        "[dim]For token-reduction proof: [cyan]sieve benchmark --compare[/][/]"
    )


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_headline(
    *,
    aggregated: "AggregatedCompareSummary | None" = None,
    summary: CompareSummary | None = None,
    single: BenchmarkSummary | None = None,
    pricing_tier: str = "local",
    model: str = "",
    fixture: str = "",
) -> str:
    """One-line, paste-ready summary — the shareable hyperfine-equivalent.

    Reports what was measured, avoids adjectives. Safe for direct
    copy-paste into a README, commit message, tweet, or Slack. No
    rich markup. The aggregated variant includes ± stddev.

    Aggregated (preferred):
        "Sieve sent 52% ± 3% fewer tokens (30,390 → 14,581) over
         3 × 15-turn qwen3.5:9b conversations on the 'medium' fixture;
         correct recalls 6/6; trap refused."
    Compare (single run):
        "Sieve sent 52% fewer tokens (30,390 → 14,581) over a 15-turn
         qwen3.5:9b conversation on the 'medium' fixture; correct
         recalls 6/6; trap refused."
    Single-mode:
        "Sieve answered 6/6 recall questions correctly and refused on
         the trap over 15 turns."
    """
    if aggregated is not None:
        pct = aggregated.reduction_pct
        base = aggregated.baseline_tokens
        sie = aggregated.sieve_outbound_tokens
        runs_n = len(aggregated.runs)
        turns_n = len(aggregated.runs[0].sieve.turns) if aggregated.runs else 0
        recall_range = _recall_range(aggregated)
        if aggregated.all_traps_refused:
            trap_str = "trap refused"
        elif all(x is None for x in aggregated.trap_absence_per_run):
            trap_str = "trap not reached"
        elif any(x is True for x in aggregated.trap_absence_per_run):
            trap_str = "trap mixed"
        else:
            trap_str = "trap failed"
        fixture_str = f" on the '{fixture}' fixture" if fixture else ""
        model_str = f" {model}" if model else ""
        # Positive reduction (Sieve cut tokens) vs overhead (Sieve
        # added tokens on a small baseline). Honest framing either
        # way — never say "−-N% fewer".
        if pct.mean >= 0:
            pct_str = (
                f"{pct.mean:.0f}%" if pct.n <= 1 or pct.stddev == 0
                else f"{pct.mean:.0f}% ± {pct.stddev:.0f}%"
            )
            verb = f"Sieve sent {pct_str} fewer tokens"
        else:
            overhead = abs(pct.mean)
            pct_str = (
                f"{overhead:.0f}%" if pct.n <= 1 or pct.stddev == 0
                else f"{overhead:.0f}% ± {pct.stddev:.0f}%"
            )
            verb = f"Sieve added {pct_str} more tokens than the baseline"
        return (
            f"{verb} "
            f"({base.mean:,.0f} → {sie.mean:,.0f}) over "
            f"{runs_n} × {turns_n}-turn{model_str} conversations"
            f"{fixture_str}; "
            f"correct recalls {recall_range}; {trap_str}."
        )

    if summary is not None:
        saved = summary.tokens_saved
        pct = summary.reduction_pct
        recalls = (
            f"{summary.sieve.correct_recalls}/{summary.sieve.gradable_recalls}"
            if summary.sieve.gradable_recalls else "—"
        )
        trap_str = (
            "trap refused" if summary.sieve.trap_absence_signal is True
            else "trap failed" if summary.sieve.trap_absence_signal is False
            else "trap not reached"
        )
        fixture_str = f" on the '{fixture}' fixture" if fixture else ""
        model_str = f" {model}" if model else ""
        turns_n = len(summary.sieve.turns)
        return (
            f"Sieve sent {pct:.0f}% fewer tokens "
            f"({summary.baseline_tokens:,} → {summary.sieve_outbound_tokens:,}) "
            f"over a {turns_n}-turn{model_str} conversation{fixture_str}; "
            f"correct recalls {recalls}; {trap_str}."
        )

    if single is not None:
        recalls = (
            f"{single.correct_recalls}/{single.gradable_recalls}"
            if single.gradable_recalls else "—"
        )
        trap_mark = (
            " and refused on the trap"
            if single.trap_absence_signal is True else ""
        )
        return (
            f"Sieve answered {recalls} recall questions correctly"
            f"{trap_mark} over {len(single.turns)} turns."
        )

    return ""


def _recall_range(agg: "AggregatedCompareSummary") -> str:
    """Render per-run recall counts as a compact range label."""
    if not agg.correct_recalls_per_run or agg.gradable_recalls == 0:
        return "—"
    lo = min(agg.correct_recalls_per_run)
    hi = max(agg.correct_recalls_per_run)
    denom = agg.gradable_recalls
    if lo == hi:
        return f"{hi}/{denom}" + (" (all runs)" if len(agg.runs) > 1 else "")
    return f"{lo}/{denom}–{hi}/{denom} across runs"


def render_compare_summary(
    summary: CompareSummary,
    *,
    model: str,
    sieve_base_url: str,
    direct_base_url: str,
    console,
    pricing_tier: str = "local",
) -> None:
    """Rich rendering for --compare runs.

    The headline is tokens-saved + correct-recalls. Surfaces:
      * side-by-side token, TTFT, and total-latency totals
      * per-turn baseline context-growth sparkline (the bloat curve)
      * cost panel when a paid pricing tier is selected
      * side-by-side model responses on illustrative turns (the wow)
      * one-line headline at the bottom
    """
    from rich.table import Table
    from rich.panel import Panel
    from sieve._pricing import dollars_saved, price_for, tier_label
    from sieve._sparkline import sparkline

    saved = summary.tokens_saved
    pct = summary.reduction_pct

    # ── Baseline vs Sieve headline table ───────────────────────────
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

    # Time to first token — perceived-speed metric. Sieve sends a
    # smaller payload so the LLM starts generating sooner.
    baseline_avg_ttft = _avg([t.ttft_s for t in summary.baseline.turns if t.ttft_s > 0])
    sieve_avg_ttft = _avg([t.ttft_s for t in summary.sieve.turns if t.ttft_s > 0])
    if baseline_avg_ttft > 0 or sieve_avg_ttft > 0:
        ttft_delta = sieve_avg_ttft - baseline_avg_ttft
        ttft_color = "green" if ttft_delta < 0 else "red"
        token_table.add_row(
            "Time to first token",
            f"{baseline_avg_ttft:.1f}s",
            f"{sieve_avg_ttft:.1f}s",
            f"[{ttft_color}]{ttft_delta:+.1f}s[/]",
        )

    baseline_avg_total = _avg([t.elapsed_s for t in summary.baseline.turns])
    sieve_avg_total = _avg([t.elapsed_s for t in summary.sieve.turns])
    token_table.add_row(
        "Avg wall-clock / turn",
        f"{baseline_avg_total:.1f}s",
        f"{sieve_avg_total:.1f}s",
        f"{(sieve_avg_total - baseline_avg_total):+.1f}s",
    )
    console.print(token_table)

    # ── Context-growth sparkline ───────────────────────────────────
    # Baseline's per-turn inbound token count shows how conversation
    # history accumulates. Sieve's stays roughly constant because the
    # proxy trims history. Rendered as a one-line sparkline so the
    # shape is visible without a chart.
    baseline_inb = [t.inbound_tokens for t in summary.baseline.turns]
    sieve_out = [t.outbound_tokens for t in summary.sieve.turns]
    if baseline_inb and sieve_out:
        base_bar = sparkline(baseline_inb)
        sieve_bar = sparkline(sieve_out)
        console.print(
            Panel(
                "\n".join([
                    f"[red]baseline[/]  {base_bar}  "
                    f"[dim]turn 1: {baseline_inb[0]:,} → turn {len(baseline_inb)}: "
                    f"{baseline_inb[-1]:,} tokens[/]",
                    f"[green]sieve   [/]  {sieve_bar}  "
                    f"[dim]turn 1: {sieve_out[0]:,} → turn {len(sieve_out)}: "
                    f"{sieve_out[-1]:,} tokens[/]",
                ]),
                title="Per-turn context size (the bloat curve)",
                border_style="dim",
            )
        )

    # ── Side-by-side illustrative responses ────────────────────────
    # Pick 3 turns that contrast well: first recall (turn 4), update
    # recall (turn 14), trap (turn 15). Skip silently if a turn is
    # missing from either pass (defensive; shouldn't happen).
    illustrative_indices = [4, 14, 15]
    resp_table = Table(
        title="Model responses (selected turns)", show_lines=True,
    )
    resp_table.add_column("#", justify="right", style="dim", width=3)
    resp_table.add_column("Question", max_width=30)
    resp_table.add_column("Baseline (no Sieve)", max_width=50)
    resp_table.add_column("With Sieve", max_width=50)
    rows_added = 0
    for idx in illustrative_indices:
        b = next((t for t in summary.baseline.turns if t.index == idx), None)
        s = next((t for t in summary.sieve.turns if t.index == idx), None)
        if b is None or s is None:
            continue
        baseline_verdict = _turn_verdict_mark(b)
        sieve_verdict = _turn_verdict_mark(s)
        resp_table.add_row(
            str(idx),
            b.prompt,
            f"{baseline_verdict} {_clip(b.response, 280)}",
            f"{sieve_verdict} {_clip(s.response, 280)}",
        )
        rows_added += 1
    if rows_added:
        console.print(resp_table)

    # ── Memory-features panel (Sieve pass only) ────────────────────
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

    # ── Cost panel (only when a paid tier was selected) ────────────
    if price_for(pricing_tier) > 0 and saved > 0:
        saved_one = dollars_saved(saved, pricing_tier)
        saved_1k = saved_one * 1_000
        console.print(
            Panel(
                "\n".join([
                    f"[bold]Pricing model:[/]  {tier_label(pricing_tier)}",
                    f"[bold]This run:[/]       [green]${saved_one:.4f}[/] saved",
                    f"[bold]Per 1K runs:[/]    [green]${saved_1k:,.2f}[/] saved",
                    "",
                    "[dim]Based on input-token pricing only; response "
                    "tokens are identical in both passes.[/]",
                ]),
                title="Estimated savings",
                border_style="green",
            )
        )

    # ── Summary panel + headline ───────────────────────────────────
    sub_lines = [
        f"[bold]Model:[/]     [cyan]{model}[/]",
        f"[bold]Baseline:[/]  [cyan]{direct_base_url}[/] (no Sieve)",
        f"[bold]Sieve:[/]     [cyan]{sieve_base_url}[/]",
        "",
        build_headline(summary=summary, pricing_tier=pricing_tier),
    ]
    console.print(
        Panel(
            "\n".join(sub_lines),
            title="Benchmark summary",
            border_style="green" if saved > 0 else "red",
        )
    )
    console.print(
        "\n[dim]Run this yourself: [cyan]sieve benchmark --compare[/][/]"
    )


def render_aggregated_compare_summary(
    agg: AggregatedCompareSummary,
    *,
    model: str,
    grader_model: str,
    fixture: str,
    sieve_base_url: str,
    direct_base_url: str,
    console,
    pricing_tier: str = "local",
    turns_per_run: int | None = None,
    context_window_warning: str | None = None,
    skipped_fixtures: list[str] | None = None,
) -> None:
    """Skeptic-tuned render: methodology first, results second, limitations
    visible, repro command at the end, shareable closing line.

    Hierarchy mirrors the research findings: Cloudflare "Benchmarks are
    hard" + hyperfine "Summary" + ripgrep "biased author" disclosure.
    Nothing here is marketing copy. Adjectives are barred.
    """
    from rich.table import Table
    from rich.panel import Panel
    from sieve._pricing import dollars_saved, price_for, tier_label
    from sieve._agent_fixture import fixture_description

    if not agg.runs:
        console.print("[red]No runs completed.[/]")
        return

    runs_n = len(agg.runs)
    turns_n = turns_per_run or len(agg.runs[0].sieve.turns)

    # ── Run banner — self-identifying for a screenshot ──────────────
    banner = (
        f"[bold]Sieve benchmark[/]  "
        f"{runs_n} × {turns_n} turns  ·  fixture: [cyan]{fixture}[/] "
        f"({fixture_description(fixture)})  ·  model: [cyan]{model}[/]  "
        f"·  grader: [cyan]{grader_model}[/]"
    )
    console.print(banner)
    if pricing_tier and pricing_tier != "local":
        console.print(
            f"[dim]Pricing tier for cost panel: {tier_label(pricing_tier)}[/]"
        )
    console.print()

    # ── Methodology ────────────────────────────────────────────────
    grader_note = (
        " [yellow](self-grading)[/]"
        if grader_model == model
        else " (independent)"
    )
    meth_lines = [
        "• [bold]Script:[/] 15-turn conversation "
        "(3 intros → 5 recalls → 4 deep follow-ups → 2 updates → 1 trap)",
        f"• [bold]Baseline pass:[/] each turn POSTed directly to "
        f"[cyan]{direct_base_url}[/] with the '{fixture}' agent payload",
        f"• [bold]Sieve pass:[/] same payload → Sieve proxy at "
        f"[cyan]{sieve_base_url}[/] → same endpoint",
        f"• [bold]Grading:[/] recall + trap answered by "
        f"[cyan]{grader_model}[/] at temperature 0{grader_note}",
        f"• [bold]Runs:[/] {runs_n} full scripts, mean ± sample stddev reported",
        "• [bold]Latency:[/] not reported as a headline metric — depends on "
        "model, hardware, and network, none of which Sieve controls",
    ]
    console.print(
        Panel("\n".join(meth_lines), title="Methodology", border_style="dim")
    )

    # ── Results table ──────────────────────────────────────────────
    results = Table(title="Results  (mean ± stddev)", show_lines=False)
    results.add_column("Metric")
    results.add_column("Baseline (direct)", justify="right")
    results.add_column("With Sieve", justify="right")
    results.add_column("Δ", justify="right")

    base_tok = agg.baseline_tokens.render()
    sie_tok = agg.sieve_outbound_tokens.render()
    saved_mean = agg.tokens_saved.mean
    pct_mean = agg.reduction_pct.mean
    pct_std = agg.reduction_pct.stddev
    if pct_mean >= 0:
        # Reduction (the common case on medium / large / xlarge).
        if runs_n <= 1 or pct_std == 0:
            delta_cell = (
                f"[bold green]−{pct_mean:.0f}%[/] "
                f"([green]−{saved_mean:,.0f}[/])"
            )
        else:
            delta_cell = (
                f"[bold green]−{pct_mean:.0f}% ± {pct_std:.0f}%[/] "
                f"([green]−{saved_mean:,.0f}[/])"
            )
    else:
        # Sieve added tokens — only plausible on small/bare fixtures.
        overhead_pct = abs(pct_mean)
        overhead_tokens = -saved_mean
        if runs_n <= 1 or pct_std == 0:
            delta_cell = (
                f"[yellow]+{overhead_pct:.0f}%[/] "
                f"([yellow]+{overhead_tokens:,.0f}[/])"
            )
        else:
            delta_cell = (
                f"[yellow]+{overhead_pct:.0f}% ± {pct_std:.0f}%[/] "
                f"([yellow]+{overhead_tokens:,.0f}[/])"
            )
    results.add_row(
        "Tokens sent to LLM / run",
        base_tok,
        sie_tok,
        delta_cell,
    )
    results.add_row(
        "Correct recalls  (Sieve)",
        "—",
        _recall_range(agg),
        "",
    )
    trap_cell = (
        "[green]refused ✓[/]" if agg.all_traps_refused
        else "[red]failed ✗[/]" if any(
            x is False for x in agg.trap_absence_per_run
        )
        else "[dim]not reached[/]" if all(
            x is None for x in agg.trap_absence_per_run
        )
        else "[yellow]mixed[/]"
    )
    results.add_row(
        "Trap question  (Sieve)",
        "—",
        trap_cell,
        "",
    )
    facts_lo = min(agg.facts_learned_per_run)
    facts_hi = max(agg.facts_learned_per_run)
    facts_cell = (
        f"{facts_hi}" if facts_lo == facts_hi
        else f"{facts_lo}–{facts_hi}"
    )
    results.add_row(
        "Facts learned  (Sieve)",
        "—",
        facts_cell,
        "",
    )
    console.print(results)

    # ── Cost panel (paid pricing tier only) ────────────────────────
    if price_for(pricing_tier) > 0 and saved_mean > 0:
        saved_usd = dollars_saved(saved_mean, pricing_tier)
        saved_1k = saved_usd * 1000
        console.print(
            Panel(
                "\n".join([
                    f"[bold]Per run:[/]     "
                    f"saved [green]${saved_usd:,.4f}[/] @ {tier_label(pricing_tier)}",
                    f"[bold]Per 1K runs:[/] [green]${saved_1k:,.2f}[/]",
                    "",
                    "[dim]Based on input-token pricing only. Response-token "
                    "costs are identical in both passes. Multiply by your "
                    "own conversation count to extrapolate.[/]",
                ]),
                title="What this means per run",
                border_style="dim",
            )
        )

    # ── Trade-offs — named explicitly to inoculate against rebuttal ──
    sieve_wall = agg.sieve_wall_clock_s.mean
    base_wall = agg.baseline_wall_clock_s.mean
    tradeoff_lines: list[str] = []
    if sieve_wall > base_wall:
        delta = sieve_wall - base_wall
        tradeoff_lines.append(
            f"• Wall-clock overhead on this run: [yellow]+{delta:.1f}s total[/] "
            f"(baseline {base_wall:.1f}s, Sieve {sieve_wall:.1f}s)"
        )
        tradeoff_lines.append(
            "  This is retrieval + recall-tool overhead. On cloud models "
            "(Sonnet, GPT-4o) the smaller payload is generation-bound, "
            "typically making Sieve faster end-to-end. Not measured here."
        )
    if agg.baseline_truncation_observed:
        tradeoff_lines.append(
            "• [yellow]Baseline truncation observed[/] — the baseline "
            "payload exceeded the model's context window on one or more "
            "turns. Sieve compressed it to fit."
        )
    if tradeoff_lines:
        console.print(
            Panel(
                "\n".join(tradeoff_lines),
                title="Trade-offs",
                border_style="yellow",
            )
        )

    # ── Known limitations ─────────────────────────────────────────
    lim_lines = [
        f"• Single fixture size: '{fixture}'. Re-run with "
        f"[cyan]--fixture {{small|medium|large|xlarge}}[/] to test other "
        f"payload shapes.",
        f"• Recall grading by [cyan]{grader_model}[/]"
        + (
            " — same model as the one being tested. Skeptics flag this as "
            "self-grading. Re-run with [cyan]--grader-model[/] for "
            "independent scoring."
            if grader_model == model
            else "."
        ),
        f"• Script is fixed and {turns_n} turns long. Longer conversations "
        f"test different behaviour; see [cyan]--turns[/].",
        f"• {runs_n} runs gives {'point-in-time results' if runs_n == 1 else 'a stddev estimate'}"
        f"; increase [cyan]--runs[/] for tighter confidence.",
    ]
    if skipped_fixtures:
        lim_lines.append(
            f"• User-excluded fixtures this run: "
            f"{', '.join(skipped_fixtures)} (context-window concern)."
        )
    if context_window_warning:
        lim_lines.append(f"• {context_window_warning}")
    console.print(
        Panel(
            "\n".join(lim_lines),
            title="Known limitations",
            border_style="dim",
        )
    )

    # ── Reproduce line ─────────────────────────────────────────────
    import hashlib
    from sieve._agent_fixture import fixture_approx_tokens
    sig = hashlib.sha256(
        f"{fixture}|{model}|{grader_model}|{turns_n}|{runs_n}|{pricing_tier}".encode()
    ).hexdigest()[:12]
    repro_cmd = (
        f"sieve benchmark --fixture {fixture} --model {model} "
        f"--grader-model {grader_model} --turns {turns_n} --runs {runs_n} "
        f"--pricing {pricing_tier}"
    )
    repro_lines = [
        f"[bold]Command:[/] [cyan]{repro_cmd}[/]",
        f"[bold]Fixture baseline tokens:[/] ~{fixture_approx_tokens(fixture):,} "
        f"(+ tool schemas + growing history)",
        f"[bold]Config signature:[/] {sig}",
    ]
    try:
        from importlib.metadata import version as _pkgver
        repro_lines.append(f"[bold]Sieve version:[/] {_pkgver('llm-sieve')}")
    except Exception:
        pass
    console.print(
        Panel(
            "\n".join(repro_lines),
            title="Reproduce",
            border_style="dim",
        )
    )

    # ── Shareable closing line (the hyperfine pattern) ─────────────
    headline = build_headline(
        aggregated=agg, pricing_tier=pricing_tier, model=model, fixture=fixture,
    )
    sep = "─" * min(console.width if hasattr(console, "width") else 80, 80)
    console.print()
    console.print(f"[dim]{sep}[/]")
    console.print(f"[bold]{headline}[/]")
    console.print(f"[dim]{sep}[/]")


def _turn_verdict_mark(t: TurnResult) -> str:
    """Compact ✓/✗ marker for a turn's grading outcome."""
    if t.phase == "trap":
        if t.absence_signal is True:
            return "[green]✓[/]"
        if t.absence_signal is False:
            return "[red]✗[/]"
        return ""
    if t.correct_recall is True:
        return "[green]✓[/]"
    if t.correct_recall is False:
        return "[red]✗[/]"
    return ""


def _clip(text: str, max_len: int) -> str:
    """Truncate with ellipsis so the response cell stays readable."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# ── Machine-readable output ──────────────────────────────────────────────


def summary_to_dict(
    summary: CompareSummary | BenchmarkSummary,
    *,
    model: str,
    pricing_tier: str = "local",
) -> dict:
    """Convert a summary to a JSON-serialisable dict.

    Used by ``--format json`` and as the data source for the
    markdown formatter. Callers merge in their own context (run
    timestamp, base URLs) as needed.
    """
    from sieve._pricing import dollars_saved, price_for

    def _turn_dict(t: TurnResult) -> dict:
        return {
            "index": t.index,
            "phase": t.phase,
            "prompt": t.prompt,
            "response": t.response,
            "inbound_tokens": t.inbound_tokens,
            "outbound_tokens": t.outbound_tokens,
            "facts_before": t.facts_before,
            "facts_after": t.facts_after,
            "elapsed_s": round(t.elapsed_s, 3),
            "ttft_s": round(t.ttft_s, 3),
            "absence_signal": t.absence_signal,
            "correct_recall": t.correct_recall,
            "activation_phase": t.activation_phase,
        }

    if isinstance(summary, CompareSummary):
        saved = summary.tokens_saved
        return {
            "mode": "compare",
            "model": model,
            "pricing_tier": pricing_tier,
            "baseline_tokens": summary.baseline_tokens,
            "sieve_outbound_tokens": summary.sieve_outbound_tokens,
            "sieve_inbound_tokens": summary.sieve_inbound_tokens,
            "tokens_saved": saved,
            "reduction_pct": round(summary.reduction_pct, 2),
            "dollars_saved_per_run": (
                round(dollars_saved(saved, pricing_tier), 6)
                if price_for(pricing_tier) > 0 else 0.0
            ),
            "dollars_saved_per_1k_runs": (
                round(dollars_saved(saved, pricing_tier) * 1000, 4)
                if price_for(pricing_tier) > 0 else 0.0
            ),
            "headline": build_headline(summary=summary, pricing_tier=pricing_tier),
            "baseline": {
                "facts_learned": summary.baseline.facts_learned,
                "correct_recalls": summary.baseline.correct_recalls,
                "gradable_recalls": summary.baseline.gradable_recalls,
                "trap_absence_signal": summary.baseline.trap_absence_signal,
                "turns": [_turn_dict(t) for t in summary.baseline.turns],
            },
            "sieve": {
                "facts_learned": summary.sieve.facts_learned,
                "correct_recalls": summary.sieve.correct_recalls,
                "gradable_recalls": summary.sieve.gradable_recalls,
                "trap_absence_signal": summary.sieve.trap_absence_signal,
                "turns": [_turn_dict(t) for t in summary.sieve.turns],
            },
        }

    # Single-mode BenchmarkSummary.
    return {
        "mode": "single",
        "model": model,
        "total_inbound": summary.total_inbound,
        "total_outbound": summary.total_outbound,
        "facts_learned": summary.facts_learned,
        "correct_recalls": summary.correct_recalls,
        "gradable_recalls": summary.gradable_recalls,
        "trap_absence_signal": summary.trap_absence_signal,
        "headline": build_headline(single=summary),
        "turns": [_turn_dict(t) for t in summary.turns],
    }


def render_aggregated_markdown(
    agg: AggregatedCompareSummary,
    *,
    model: str,
    grader_model: str,
    fixture: str,
    sieve_base_url: str,
    direct_base_url: str,
    pricing_tier: str = "local",
    turns_per_run: int | None = None,
    skipped_fixtures: list[str] | None = None,
    context_window_warning: str | None = None,
) -> str:
    """Shareable markdown report following the same hierarchy as the
    terminal render: banner → methodology → results → limitations →
    reproduce → raw data.

    Designed to paste directly into a GitHub issue, Slack message, or
    README. No adjectives. No sales pitch. The closing line (the
    hyperfine-equivalent) is first so it's visible in message previews.
    """
    import hashlib
    from sieve._pricing import dollars_saved, price_for, tier_label
    from sieve._agent_fixture import fixture_approx_tokens, fixture_description

    if not agg.runs:
        return "_No benchmark runs completed._\n"

    runs_n = len(agg.runs)
    turns_n = turns_per_run or len(agg.runs[0].sieve.turns)
    headline = build_headline(
        aggregated=agg, pricing_tier=pricing_tier, model=model, fixture=fixture,
    )
    sig = hashlib.sha256(
        f"{fixture}|{model}|{grader_model}|{turns_n}|{runs_n}|{pricing_tier}".encode()
    ).hexdigest()[:12]
    try:
        from importlib.metadata import version as _pkgver
        sieve_version = _pkgver("llm-sieve")
    except Exception:
        sieve_version = "unknown"

    lines: list[str] = []
    lines.append(f"# Sieve benchmark — {fixture} fixture — {model}")
    lines.append("")
    lines.append(f"> {headline}")
    lines.append("")

    # ── Methodology ────────────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- **Script:** 15-turn conversation "
        "(3 intros → 5 recalls → 4 deep follow-ups → 2 updates → 1 trap)"
    )
    lines.append(
        f"- **Baseline pass:** each turn POSTed directly to "
        f"`{direct_base_url}` with the '{fixture}' agent payload "
        f"({fixture_description(fixture)})"
    )
    lines.append(
        f"- **Sieve pass:** same payload → Sieve proxy at "
        f"`{sieve_base_url}` → same endpoint"
    )
    grader_note = (
        " **⚠ self-grading — same model as the one being tested**"
        if grader_model == model else ""
    )
    lines.append(
        f"- **Grading:** recall and trap verdicts by `{grader_model}` "
        f"at temperature 0.{grader_note}"
    )
    lines.append(
        f"- **Runs:** {runs_n} full 15-turn scripts. "
        f"Mean ± sample stddev reported below."
    )
    lines.append(
        "- **Latency:** wall-clock is included in raw data but not "
        "reported as a headline — it depends on model, hardware, and "
        "network, none of which Sieve controls."
    )
    lines.append("")

    # ── Results table ──────────────────────────────────────────────
    lines.append("## Results")
    lines.append("")
    lines.append("| Metric | Baseline (direct) | With Sieve | Δ |")
    lines.append("|---|---:|---:|---:|")
    base_tok = agg.baseline_tokens.render()
    sie_tok = agg.sieve_outbound_tokens.render()
    pct_mean = agg.reduction_pct.mean
    pct_std = agg.reduction_pct.stddev
    saved_mean = agg.tokens_saved.mean
    if pct_mean >= 0:
        pct_str = (
            f"−{pct_mean:.0f}%" if runs_n <= 1 or pct_std == 0
            else f"−{pct_mean:.0f}% ± {pct_std:.0f}%"
        )
        delta_md = f"**{pct_str}** (−{saved_mean:,.0f})"
    else:
        overhead_pct = abs(pct_mean)
        pct_str = (
            f"+{overhead_pct:.0f}%" if runs_n <= 1 or pct_std == 0
            else f"+{overhead_pct:.0f}% ± {pct_std:.0f}%"
        )
        delta_md = f"**{pct_str}** (+{-saved_mean:,.0f})"
    lines.append(
        f"| **Tokens sent to LLM / run** | {base_tok} | {sie_tok} | "
        f"{delta_md} |"
    )
    lines.append(
        f"| Correct recalls (Sieve) | — | {_recall_range(agg)} | — |"
    )
    trap_cell = (
        "refused ✓" if agg.all_traps_refused
        else "failed ✗" if any(x is False for x in agg.trap_absence_per_run)
        else "not reached" if all(x is None for x in agg.trap_absence_per_run)
        else "mixed"
    )
    lines.append(
        f"| Trap question (Sieve) | — | {trap_cell} | — |"
    )
    facts_lo = min(agg.facts_learned_per_run)
    facts_hi = max(agg.facts_learned_per_run)
    facts_cell = f"{facts_hi}" if facts_lo == facts_hi else f"{facts_lo}–{facts_hi}"
    lines.append(
        f"| Facts learned (Sieve) | — | {facts_cell} | — |"
    )
    lines.append("")

    # ── Cost panel ─────────────────────────────────────────────────
    if price_for(pricing_tier) > 0 and saved_mean > 0:
        per_run = dollars_saved(saved_mean, pricing_tier)
        per_k = per_run * 1000
        lines.append("### What this means per run")
        lines.append("")
        lines.append(
            f"At {tier_label(pricing_tier)}: saved **${per_run:,.4f}** per "
            f"run ({pct_mean:.0f}% of input-token cost). Extrapolated, "
            f"that's **${per_k:,.2f}** per 1K runs at the same scale."
        )
        lines.append("")
        lines.append(
            "> Input-token pricing only; response tokens are identical "
            "in both passes. Multiply by your actual conversation count."
        )
        lines.append("")

    # ── Trade-offs ─────────────────────────────────────────────────
    sieve_wall = agg.sieve_wall_clock_s.mean
    base_wall = agg.baseline_wall_clock_s.mean
    tradeoffs: list[str] = []
    if sieve_wall > base_wall:
        delta = sieve_wall - base_wall
        tradeoffs.append(
            f"- Wall-clock overhead on this run: +{delta:.1f}s total "
            f"(baseline {base_wall:.1f}s, Sieve {sieve_wall:.1f}s). "
            "This is retrieval + recall-tool overhead. On cloud models "
            "where payload size dominates generation time, Sieve is "
            "typically faster end-to-end. Not measured here."
        )
    if agg.baseline_truncation_observed:
        tradeoffs.append(
            "- **Baseline truncation observed:** the baseline payload "
            "exceeded the model's context window on one or more turns. "
            "Sieve compressed it to fit."
        )
    if tradeoffs:
        lines.append("## Trade-offs")
        lines.append("")
        lines.extend(tradeoffs)
        lines.append("")

    # ── Known limitations ─────────────────────────────────────────
    lines.append("## Known limitations")
    lines.append("")
    lines.append(
        f"- Single fixture size: `{fixture}` "
        f"(~{fixture_approx_tokens(fixture):,} tokens base payload). "
        "Heavier agents see larger reductions. Re-run with "
        "`--fixture large` or `--fixture xlarge` to test."
    )
    if grader_model == model:
        lines.append(
            "- **Recall grading by the same model being tested.** "
            "Skeptics correctly flag this as a potential bias source. "
            "Pass `--grader-model` for independent scoring."
        )
    else:
        lines.append(
            f"- Recall grading by `{grader_model}` (independent of "
            "the test model)."
        )
    lines.append(
        f"- Script is fixed and {turns_n} turns long. Longer "
        "conversations test different behaviour; see `--turns`."
    )
    if runs_n <= 1:
        lines.append(
            "- Only one run. No stddev estimate is available. "
            "Re-run with `--runs 3` or higher for tighter confidence."
        )
    else:
        lines.append(
            f"- {runs_n} runs gives a stddev estimate. Increase `--runs` "
            "for tighter confidence intervals."
        )
    if skipped_fixtures:
        lines.append(
            f"- User excluded these fixtures from this run: "
            f"{', '.join(skipped_fixtures)} (context-window concern)."
        )
    if context_window_warning:
        lines.append(f"- {context_window_warning}")
    lines.append("")

    # ── Reproducibility ───────────────────────────────────────────
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```")
    lines.append(
        f"sieve benchmark --fixture {fixture} --model {model} "
        f"--grader-model {grader_model} --turns {turns_n} --runs {runs_n} "
        f"--pricing {pricing_tier}"
    )
    lines.append("```")
    lines.append("")
    lines.append(f"- Config signature: `{sig}`")
    lines.append(f"- Sieve version: `{sieve_version}`")
    lines.append("")

    # ── Raw data (collapsed) ──────────────────────────────────────
    lines.append("<details>")
    lines.append("<summary>Raw data — per-turn results (last run)</summary>")
    lines.append("")
    last = agg.most_recent
    lines.append(
        "| # | Phase | Prompt | Baseline tokens | Sieve tokens | "
        "Baseline verdict | Sieve verdict |"
    )
    lines.append("|---:|---|---|---:|---:|:---:|:---:|")
    def _mk(t: TurnResult) -> str:
        if t.phase == "trap":
            if t.absence_signal is True: return "refused ✓"
            if t.absence_signal is False: return "fabricated ✗"
            return "—"
        if t.correct_recall is True: return "✓"
        if t.correct_recall is False: return "✗"
        return "—"
    for b, s in zip(last.baseline.turns, last.sieve.turns):
        prompt_md = b.prompt.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {b.index} | {b.phase} | {prompt_md} | "
            f"{b.inbound_tokens:,} | {s.outbound_tokens:,} | "
            f"{_mk(b)} | {_mk(s)} |"
        )
    lines.append("")
    # Trap transcript — showing the refusal is more persuasive than
    # claiming it (research: ripgrep-style "show your work").
    trap_turn = next(
        (t for t in last.sieve.turns if t.phase == "trap"), None
    )
    if trap_turn and trap_turn.response:
        lines.append("**Trap transcript (Sieve):**")
        lines.append("")
        lines.append(f"> **Q:** {trap_turn.prompt}")
        lines.append(">")
        lines.append(f"> **A:** {trap_turn.response}")
        lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def render_markdown(
    summary: CompareSummary | BenchmarkSummary,
    *,
    model: str,
    sieve_base_url: str,
    direct_base_url: str = "",
    pricing_tier: str = "local",
) -> str:
    """Return a markdown summary suitable for README / blog paste.

    Leads with the headline, then a tokens table, recalls, trap
    verdict, and a per-turn table. Markdown is the canonical "paste
    me somewhere" format.
    """
    from sieve._pricing import dollars_saved, price_for, tier_label

    lines: list[str] = []

    if isinstance(summary, CompareSummary):
        saved = summary.tokens_saved
        pct = summary.reduction_pct
        lines.append(f"## Sieve benchmark — {model}")
        lines.append("")
        lines.append(f"> {build_headline(summary=summary, pricing_tier=pricing_tier)}")
        lines.append("")
        lines.append("| Metric | Baseline (direct) | With Sieve | Δ |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| Tokens sent to LLM | {summary.baseline_tokens:,} "
            f"| {summary.sieve_outbound_tokens:,} "
            f"| **−{saved:,} ({pct:.1f}%)** |"
        )
        b_ttft = _avg([t.ttft_s for t in summary.baseline.turns if t.ttft_s > 0])
        s_ttft = _avg([t.ttft_s for t in summary.sieve.turns if t.ttft_s > 0])
        if b_ttft > 0 or s_ttft > 0:
            lines.append(
                f"| Time to first token | {b_ttft:.1f}s | {s_ttft:.1f}s | "
                f"{(s_ttft - b_ttft):+.1f}s |"
            )
        b_total = _avg([t.elapsed_s for t in summary.baseline.turns])
        s_total = _avg([t.elapsed_s for t in summary.sieve.turns])
        lines.append(
            f"| Avg wall-clock / turn | {b_total:.1f}s | {s_total:.1f}s "
            f"| {(s_total - b_total):+.1f}s |"
        )
        lines.append("")
        lines.append(
            f"**Facts learned (Sieve):** {summary.sieve.facts_learned}  "
        )
        if summary.sieve.gradable_recalls:
            lines.append(
                f"**Correct recalls (Sieve):** "
                f"{summary.sieve.correct_recalls}/{summary.sieve.gradable_recalls}  "
            )
        trap = summary.sieve.trap_absence_signal
        if trap is True:
            lines.append("**Trap query (Sieve):** absence signal fired ✓")
        elif trap is False:
            lines.append("**Trap query (Sieve):** hallucinated ✗")
        lines.append("")
        if price_for(pricing_tier) > 0 and saved > 0:
            per_run = dollars_saved(saved, pricing_tier)
            per_k = per_run * 1_000
            lines.append(
                f"**Savings ({tier_label(pricing_tier)}):** "
                f"${per_run:.4f} per run, ${per_k:,.2f} per 1K runs."
            )
            lines.append("")
        # Per-turn table.
        lines.append("### Per-turn")
        lines.append("")
        lines.append(
            "| # | Phase | Baseline tokens | Sieve tokens | "
            "Recall (baseline) | Recall (sieve) |"
        )
        lines.append("|---:|---|---:|---:|:---:|:---:|")
        for b, s in zip(summary.baseline.turns, summary.sieve.turns):
            def _mk(t: TurnResult) -> str:
                if t.phase == "trap":
                    if t.absence_signal is True:
                        return "✓"
                    if t.absence_signal is False:
                        return "✗"
                    return ""
                if t.correct_recall is True:
                    return "✓"
                if t.correct_recall is False:
                    return "✗"
                return ""
            lines.append(
                f"| {b.index} | {b.phase} | {b.inbound_tokens:,} "
                f"| {s.outbound_tokens:,} | {_mk(b)} | {_mk(s)} |"
            )
        return "\n".join(lines) + "\n"

    # Single-mode
    lines.append(f"## Sieve benchmark — {model}")
    lines.append("")
    lines.append(f"> {build_headline(single=summary)}")
    lines.append("")
    lines.append(f"**Facts learned:** {summary.facts_learned}  ")
    if summary.gradable_recalls:
        lines.append(
            f"**Correct recalls:** "
            f"{summary.correct_recalls}/{summary.gradable_recalls}  "
        )
    trap = summary.trap_absence_signal
    if trap is True:
        lines.append("**Trap query:** absence signal fired ✓")
    elif trap is False:
        lines.append("**Trap query:** hallucinated ✗")
    return "\n".join(lines) + "\n"

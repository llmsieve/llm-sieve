---
date: 2026-06-10
authors:
  - sieve-engineering
categories:
  - Engineering
description: >
  Why agent token bills grow faster than developers expect — and what
  the relationship between tokens and compute actually looks like at
  scale.
---

# The hidden cost of context

*Applies to: Sieve v1.0.x*

The conventional wisdom about LLM API costs is straightforward:
tokens × price per token = bill. If you double the prompt, you
double the cost. Predictable, linear, easy to budget.

This is wrong in a way that matters more the longer your agent runs.

<!-- more -->

## The naive view, and why it's almost right

For a single API call to a hosted LLM, the bill genuinely is close
to linear. OpenAI charges per million input tokens; Anthropic charges
per million input tokens; both have a posted rate, and at the level
of a single request, that rate is what you pay.

But the cost the LLM provider absorbs to serve that request is not
linear in token count. The transformer's attention mechanism — the
part that lets it relate every token to every other token —
[runs in O(N²)](https://arxiv.org/abs/1706.03762) where N is the
context length. A 4× longer prompt requires 16× the attention
compute, plus an additional linear factor for the KV cache.

For a while, the providers ate this cost themselves. Increasingly
they don't. The trend over the last 18 months has been **tiered
pricing at higher context** — at the time of writing, Anthropic
charges premium rates above 200k tokens, OpenAI's long-context tiers
on GPT-4o and o-series similarly step up, Google offers explicit
"long context" pricing on Gemini. The "tokens × rate" formula now
has multiple rates per provider depending on where you sit.

If you're serving your own model (Ollama, vLLM, LM Studio, anything
GPU-local), the O(N²) cost lands directly on your wallet via GPU
time. You don't see a stepped price — you see a stepped slowness
and a stepped electricity bill.

So the first hidden cost of long context is **compute per token
isn't constant**. As you move along the context axis, each token
costs more than the last.

## The agent-loop accumulation problem

This compound interests with a second pattern: agent loops grow
their context monotonically.

A typical agent setup ships, on every turn:

1. **The system prompt** — instructions about the agent's role,
   safety policies, tone. Usually 200-2000 tokens. Static, repeated
   verbatim every turn.
2. **The tool schemas** — JSON-Schema definitions of every tool
   the agent can call, including descriptions and parameter types.
   For a Cursor-like agent, easily 5000-15000 tokens. Static,
   repeated verbatim every turn.
3. **The conversation history** — every previous user message and
   assistant response, all the prior tool calls and their results.
   Grows monotonically with session length. For a multi-hour coding
   session, can reach 50,000+ tokens.
4. **The current user turn** — typically a single message, maybe a
   few hundred tokens.

Items 1 and 2 are constant. Item 4 is small. **Item 3 is what
ruins your bill.**

Concretely: an agent at turn 1 might send 6,000 tokens. The same
agent at turn 30, after a real session, might send 60,000 tokens —
not because the user is asking harder questions, but because the
prior 29 turns are being re-shipped on each request.

If costs were linear, you'd pay 10× more at turn 30. But because
of the O(N²) attention cost (which providers either pass through
as long-context pricing, or absorb as latency for self-hosted
setups), the *actual* compute per turn at turn 30 is closer to
100× the compute at turn 1. The bill stops being linear and starts
being a curve.

## The quality cost almost no one talks about

There's a third hidden cost, and it's the one most likely to take
a team by surprise.

[Liu et al. (2023)](https://arxiv.org/abs/2307.03172) documented
the "lost in the middle" effect: when relevant information sits in
the middle of a long context, the model attends to it materially
less well than if it were at the start or end. The model isn't
ignoring it — accuracy doesn't drop to zero — but the recall and
reasoning quality on facts buried in long context is measurably
worse.

For an agent loop, this is exactly the worst possible failure
mode. The user mentioned a key fact 20 turns ago. By turn 25 it's
in the middle of the context, surrounded by tool calls and
intermediate exchanges. The model can see it but doesn't *attend*
to it well. The agent confabulates, or asks the user to repeat
themselves, or makes a decision based on a stale assumption.

So at long context you're paying more compute, **and** getting
worse output for it. The two failure modes compound: more cost,
less correctness.

## Why "just use a bigger context window" doesn't fix this

The reflex response in 2024-2026 has been "fine, let's get a model
with a bigger context window." Claude 3.5 Sonnet has 200k. Gemini
1.5 Pro has 2M. Surely that solves it?

It moves the problem rather than solving it. A larger context
window means:

- You can fit a longer history before the framework starts
  truncating. Good.
- You pay more per turn because every token in that window is
  attended to. Bad — sometimes very bad.
- The "lost in the middle" effect doesn't go away. It happens
  later, but it still happens. Bad.
- Your latency per turn goes up because attention is O(N²) in the
  window you're actually using. Bad.

The bigger context window is genuinely useful for a class of
single-shot tasks (analyse this 500-page PDF, summarise this whole
codebase). For *iterative agent loops*, where the relevant context
is a small subset of a growing pile, the bigger window just gives
you more room to be wasteful.

The right answer is **don't ship what the model doesn't need on
this turn**.

## Three ways to ship less

There are three architectural approaches to this problem:

### 1. Compaction

When the conversation gets too long, the agent framework discards
or summarises the older turns. Every major agent framework does
some version of this. It's the path of least resistance.

The cost: **it's lossy**. The summariser is making a judgement
about what's important without knowing what the future turns will
ask for. If turn 47 needs a detail from turn 3 that the summariser
threw away, you're stuck.

### 2. Memory library + retrieval

Store facts in a database, retrieve them on demand, inject them
into the prompt only when relevant. The library lives inside your
agent code. Letta, mem0, llama-index, agent-framework built-ins
all sit here.

The cost: **you change your agent code**. Every model call has to
ask the library "what's relevant for this turn?" If you don't
control the agent — if you're using Cursor or Claude Code or
similar — you can't do this.

### 3. Transparent proxy

Run a service between your agent and the LLM that does the
retrieval transparently. The agent doesn't know it's there. One
URL change in the agent's config and the proxy handles the rest.

This is what [Sieve](https://github.com/llmsieve/llm-sieve) is.
It strips the static parts (tool schemas, repeated instructions),
retrieves the relevant historical facts from an encrypted local
store, and rewrites the prompt into a lean payload before it
reaches the LLM.

The cost: a network hop (negligible on loopback), and the proxy
has to infer what's relevant from prompt text alone — it doesn't
have privileged access to the agent's internal state.

## What this looks like in numbers

For a representative agent payload of ~13k base tokens (a
"Cursor-class" agent fixture in Sieve's benchmark), a 6-turn
conversation looks like this on a real Ollama model:

| Setup | Tokens sent to LLM / run | Verdict |
|---|---|---|
| Baseline (direct to LLM, full history) | 85,612 | exceeds many local model context windows |
| With Sieve | 3,463 | well inside any modern context window |

That's a 96% reduction over six turns. The exact percentage varies
with agent shape and conversation pattern, but the direction is
consistent — Sieve's [README](https://github.com/llmsieve/llm-sieve#performance)
documents 95% as the floor across the architectures and contexts
we tested.

For the same Ollama qwen3:30b-a3b model, the latency benefit shows
the compound effect: not just less cost, but faster, because the
model isn't grinding through 85k tokens of attention every turn.
On frontier models the per-turn latency reduction is 3-7× on
follow-up turns.

## Where to start thinking about this

If you're running an agent against a hosted LLM and your bill is
growing faster than your user count, the diagnosis is almost
certainly **context bloat over time**. Three quick checks:

1. **Log the input token count per turn.** OpenAI and Anthropic
   return this on every response. If turn-30 token counts are
   3-10× higher than turn-1, you're seeing the accumulation
   pattern.
2. **Sample a long session and look at what's in the prompt at
   turn 30.** Most of it will be re-shipped earlier turns and
   tool schemas the model already memorised within the first few
   turns.
3. **Measure the quality drop.** If your agent gets noticeably
   worse at long sessions — confabulating, forgetting context,
   asking for clarification on things the user told it earlier
   — you're seeing "lost in the middle" in action.

If any of these match what you're seeing, the cost of doing
nothing is materially worse than the cost of investigating one of
the three architectures above.

Sieve is one specific take on the third one — transparent proxy.
You can install it and benchmark it against your own agent in
under five minutes:

```bash
pipx install llm-sieve
sieve-install
sieve benchmark    # 15-turn baseline-vs-Sieve, sandboxed
```

Whether Sieve is the right answer for your setup depends on
factors we discuss in [Why Sieve](2026-06-09-why-sieve.md). But
the underlying observation — that agent context grows faster and
costs more than the linear-pricing model suggests — applies
regardless of which approach you take.

---

*This post was drafted with AI assistance and reviewed by the Sieve
maintainer before publication. Quantitative claims link to their
source. Sieve benchmark numbers were measured against v1.0.0; see
the `benchmark` command for reproducible runs.*

---
date: 2026-06-10
authors:
  - sieve-engineering
categories:
  - Design
description: >
  OpenClaw, Hermes, and the always-on agent workload — why
  gateway-style assistants are the heaviest context spenders yet.
image: https://llmsieve.dev/assets/social/social-card-always-on-agents.png
---

# What always-on agents stand to gain from a context proxy

*Applies to: Sieve v1.0.x*

The most interesting agents of 2026 don't run in your terminal. They
live in your chat apps. [OpenClaw](https://docs.openclaw.ai/)
describes itself as "a self-hosted gateway that connects your
favorite chat apps and channel surfaces… to AI coding agents."
[Hermes](https://github.com/NousResearch/hermes-agent), from Nous
Research, runs "on a $5 VPS, a GPU cluster, or serverless
infrastructure" and lives on "Telegram, Discord, Slack, WhatsApp,
Signal, and CLI — all from a single gateway process."

The defining property of this generation isn't a feature. It's that
they're *always on*. And always-on is, by some distance, the heaviest
context workload we've seen — which makes it worth working through
what a context-reduction proxy would change for them.

<!-- more -->

To be clear about what this post is: an architectural analysis, not a
tested integration guide. We'll show why the workload shape fits, what
the wiring would look like, and exactly what we haven't verified yet.

## The session that never ends

A coding agent's context problem is bounded by your attention span.
You open a session, it accumulates, you close it; tomorrow starts
fresh. We measured what happens *within* that arc in
[The hidden cost of context](2026-06-10-hidden-cost-of-context.md),
and it's already unflattering.

A gateway agent has no such mercy. The conversation with your
personal assistant is one conversation, indefinitely. Three
properties make it the worst case:

**The cadence is chat, not work.** WhatsApp and Telegram messages are
short — a sentence, a photo caption, a "yes do that." But the payload
that ships upstream on each one carries the agent's full standing
apparatus: system prompt, persona, tool catalogue, injected memory,
recent history. When the variable part of a request is twelve tokens
and the fixed part is twelve thousand, the fixed part *is* the bill.
Every "ok thanks" costs what a feature request costs.

**Tool catalogues ride along on every message.** These agents are
maximalists by design — messaging, scheduling, browsing, shell
access, file management. Hermes spawns subagents and runs cron
automations; OpenClaw bridges entire channel surfaces. Each
capability is schema text that gets re-serialised into every single
turn, whether or not the turn is "remind me at 6" or idle chitchat.

**Their memory systems add context — by design.** Hermes has a
genuinely novel learning loop: "agent-curated memory with periodic
nudges," skills that "self-improve during use." OpenClaw maintains
its own memory surfaces. This is the right call at the application
layer — the agent *should* know you. But mechanically, every learned
fact, skill description, and memory file is more material competing
for the outbound payload. The smarter the agent gets about you, the
more it costs to say hello to it.

None of this is a criticism of either project. It's the natural
consequence of an agent that's both persistent and capable, talking
over a protocol that re-sends everything, every time.

## What a traffic-path layer changes

The three shapes of memory infrastructure — SDK, platform, proxy —
got a full comparison in [Sieve, mem0, Zep: three shapes of agent
memory](2026-06-10-three-shapes-of-agent-memory.md). For gateway
agents specifically, the proxy shape has one decisive property:
**these agents are gateways themselves, so they already believe in
interposition.** They sit between your chat apps and a model;
Sieve sits between them and the model. Nobody has to adopt anybody's
SDK.

In the traffic path, the per-turn picture changes like this:

- **The fixed payload stops being fixed.** Tool schemas the upstream
  model has already seen, repeated instructions, and stale history
  get stripped before the request leaves the box. The agent keeps
  sending its full apparatus; the model stops re-reading it.
- **Memory injection becomes need-based.** Durable facts learned from
  the conversation are stored — encrypted, local — and injected only
  on turns that actually need them, rather than standing in the
  prompt permanently.
- **Absence gets gated.** A question about something never said
  should produce a refusal, not an invention. For an agent that
  speaks to you (and possibly your contacts) on WhatsApp, the
  difference between "I don't have that" and a confident fabrication
  is reputational, not cosmetic.
- **All of it is observable.** Every response carries
  [diagnostic headers](../../diagnostic-headers.md) —
  `X-Sieve-Inbound-Tokens` vs `X-Sieve-Outbound-Tokens` is the
  per-message answer to "what did that just save me?" For agents
  whose operators watch a metered API bill, that's a number worth
  logging.

And because the memory store lives below the agent rather than inside
it, it's complementary to what Hermes and OpenClaw already do: their
memory decides what the agent knows; a traffic-path layer decides
what actually needs to ship upstream *this turn*.

## What the wiring would look like

Both projects already expose the right surface, which is what makes
this analysis more than hypothetical.

OpenClaw supports custom OpenAI-compatible providers — its docs show
[exactly this pattern](https://docs.openclaw.ai/gateway/config-tools)
for LiteLLM-style proxies, with a `baseUrl` and the
`openai-completions` adapter. Pointed at Sieve instead:

```json5
{
  models: {
    mode: "merge",
    providers: {
      "sieve": {
        baseUrl: "http://127.0.0.1:11435/v1",
        api: "openai-completions",
        models: [
          { id: "qwen3:14b", name: "Qwen3 14B via Sieve", contextWindow: 128000 }
        ],
      },
    },
  },
}
```

Hermes is even more direct: "use any model you want — … OpenAI, or
your own endpoint." Sieve *is* an own-endpoint:
`http://127.0.0.1:11435/v1`, same wire protocol as whatever it
fronts.

In both cases the agent's code, skills, and memory systems are
untouched. That's the entire point of the shape.

## What we haven't verified — read before trying

Honesty section, and it matters more than the optimistic sections
above.

**We have not yet run Sieve underneath OpenClaw or Hermes.** The
analysis here is from their documentation and ours, not from a test
matrix. Agentic workloads are the hard case for any proxy: heavy
tool-calling, streaming everywhere, long sessions. Until we publish
measured numbers from real runs — with the methodology to reproduce
them — treat per-turn savings under these agents as *expected from
the architecture*, not demonstrated.

**Anthropic's native API is not in Sieve v1.0.** Sieve speaks
Ollama's `/api/chat` and OpenAI-compatible `/v1/chat/completions`.
OpenClaw's defaults lean toward Anthropic models on Anthropic's own
protocol; routing Claude through Sieve today means an
OpenAI-compatible path to it. If your agent runs local or
OpenAI-compatible models, the fit is immediate; if it runs
Anthropic-native, it isn't yet.

**Cold start applies.** Sieve activates progressively — a memory
layer with nothing in it can't save you much. An always-on agent is
actually the *best* case for this (the store only ever warms up), but
the first day won't look like the thirtieth.

## The standing offer

If you run OpenClaw, Hermes, or anything shaped like them, the
experiment is cheap and self-grading: point the model endpoint at
Sieve, watch `X-Sieve-Inbound-Tokens` against
`X-Sieve-Outbound-Tokens` for a day, and you'll have better numbers
for your setup than we could publish for ours. The
[Ollama guide](2026-06-10-persistent-memory-for-ollama.md) covers the
five-minute install; `sieve demo` shows you the absence-trap
behaviour before you trust it with anything real.

If something breaks under your agent — streaming, tool calls,
anything — [we want the issue
report](https://github.com/llmsieve/llm-sieve/issues). Always-on
agents are the workload this architecture exists for, and the gap
between "should work" and "works" is exactly the part worth doing in
public.

---

*This post was drafted with AI assistance and reviewed by the Sieve
maintainer before publication. Descriptions of OpenClaw and Hermes
quote their documentation as fetched on 2026-06-10; if we've
misrepresented either project, [open an
issue](https://github.com/llmsieve/llm-sieve/issues) and we'll
correct it. Sieve is open source under Apache 2.0.*

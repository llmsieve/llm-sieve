---
date: 2026-06-10
authors:
  - sieve-engineering
categories:
  - Guides
description: >
  Give any Ollama model persistent, encrypted memory in about five
  minutes — without changing your client code.
image: https://llmsieve.dev/assets/social/social-card-persistent-memory-for-ollama.png
---

# Persistent memory for Ollama, in about five minutes

*Applies to: Sieve v1.0.x*

Ollama gives you a local LLM endpoint that is fast, private, and
completely stateless. Close the chat, and everything you told the
model is gone. Keep the chat open, and every turn re-sends a growing
history until the context window fills up. Ask a local model about
something it was never told, and — depending on the model — it may
simply make something up.

This guide adds a persistent, encrypted memory to any Ollama setup
using Sieve, without changing your client code beyond one URL.

<!-- more -->

## The shape of the problem

Three separate annoyances show up when you run agents or long-lived
chats against a local model, and they have a common root.

**Nothing survives the session.** Tell your assistant on Monday that
you prefer Python and your deploy target is a Raspberry Pi, and on
Tuesday it knows neither. The model has no state; the application has
to carry all of it, every time.

**The payload only grows.** The standard workaround is to re-send
history: system prompt, tool schemas, every prior turn, on every
request. We measured the consequences of that pattern in
[The hidden cost of context](2026-06-10-hidden-cost-of-context.md) —
the short version is that per-turn cost grows with conversation
length, and on local hardware that growth comes out of your
tokens-per-second.

**Absence becomes fabrication.** When a question falls outside the
context you did send, smaller models in particular tend to answer
anyway. A model that was never told your colleague's name will, often
enough, invent one.

The common root: the endpoint is stateless and the burden of memory
falls on whatever sits in front of it. Most memory frameworks ask you
to adopt an SDK and call `add()`/`search()` yourself. The approach
here is different — put the memory *in the traffic path*, so the
client stays unchanged. We wrote up why we prefer the proxy shape in
[Why Sieve](2026-06-09-why-sieve.md).

## What you'll end up with

```text
your client ──► Sieve (127.0.0.1:11435) ──► Ollama (127.0.0.1:11434)
                 │
                 └── encrypted store at ~/.sieve/memory.db
```

Sieve speaks Ollama's native `/api/chat` as well as the
OpenAI-compatible `/v1/chat/completions`, so anything that can talk
to Ollama can talk to Sieve. On each turn it strips repeated
instructions, tool schemas, and stale history from the outbound
payload; learns durable facts from the conversation; and injects the
relevant ones back in when a later turn actually needs them. The
reply comes back to your client unchanged.

## Step 1 — install Sieve

You need Python 3.11+ and a running Ollama. The recommended installer
is [pipx](https://pipx.pypa.io/):

```bash
pipx install llm-sieve
sieve --version   # sieve, version 1.0.0 or later
```

Then run the guided setup:

```bash
sieve-install
```

If Ollama is running on `127.0.0.1:11434`, the installer auto-detects
it, shows you the models you already have pulled, downloads a ~50 MB
embedding model (one-time), creates the encrypted store, and offers
to start the proxy — with optional autostart on reboot. For a
scripted, no-prompts install:

```bash
sieve-install --no-input \
  --provider http://127.0.0.1:11434 \
  --model qwen3.5:9b
```

## Step 2 — move one URL

Sieve listens on `11435` — deliberately one port up from Ollama's
`11434`. Wherever your client points at Ollama, point it at Sieve
instead.

**Ollama-native clients:**

```bash
export OLLAMA_HOST=http://127.0.0.1:11435
```

**OpenAI-compatible clients:**

```python
client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",  # was: http://127.0.0.1:11434/v1
    api_key="not-used-by-sieve",           # still forwarded upstream
)
```

**Or just curl it:**

```bash
curl http://127.0.0.1:11435/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "Hi, my name is Alex and I work on embedded firmware."}],
    "stream": false
  }'
```

Model names, request shapes, response formats, streaming — all
unchanged. The client does not know Sieve exists.

## Step 3 — prove it's working

Two built-in commands, both sandboxed (they never touch your real
store):

```bash
sieve demo
```

runs a short scripted conversation: an identity introduces itself,
shares facts, asks for them back, and closes with a question about a
person who was never mentioned. What you want to see: recall hits on
the seeded facts, and a *refusal* — not a fabrication — on the trap
question.

```bash
sieve benchmark
```

sends the same 15 messages directly to your model and through Sieve,
then prints a delta table: tokens in vs out, facts learned, response
times, and the trap verdict. Five to ten minutes depending on your
hardware, and the numbers are yours rather than ours.

## Watching it work

Every response Sieve touches carries diagnostic headers, so you don't
have to take the proxy's behaviour on faith:

| Header | What it tells you |
|---|---|
| `X-Sieve-Inbound-Tokens` | Payload size before the trim |
| `X-Sieve-Outbound-Tokens` | Payload size actually sent to Ollama |
| `X-Sieve-Phase` | `OBSERVE` / `ACCUMULATE` / `ACTIVATE` |
| `X-Sieve-Fact-Count` | Facts in the store right now |
| `X-Sieve-Proxy-Us` | Sieve's own overhead, in microseconds |

The inbound/outbound pair is the one to watch first: it's the
per-request answer to "is this actually doing anything?" The full
list is in the [diagnostic headers](../../diagnostic-headers.md)
reference.

One thing to expect: **the first few turns feel like pass-through.**
Sieve activates progressively — it observes before it accumulates,
and accumulates before it actively trims and injects. `X-Sieve-Phase`
tells you exactly where it is in that ramp, and `sieve status` shows
the fact count growing.

## Where your data lives

Everything stays on your machine. Facts, entities, and episodes land
in a SQLCipher-encrypted SQLite database at `~/.sieve/memory.db`,
with the keyfile alongside it. There is no cloud component, no
account, and no telemetry — the proxy talks to exactly one remote
party, and it's the LLM endpoint you configured. If that endpoint is
Ollama on localhost, nothing leaves the box at all.

The store belongs to you, not to the package: upgrades via
`pipx upgrade llm-sieve` never touch `~/.sieve/`, and the only
command that deletes user data is `sieve uninstall --hard`, which
makes you type `DELETE` first.

## Honest caveats

**Small models still have small-model problems.** Sieve can put the
right facts in front of the model and refuse to let absence turn into
invention on the turns it gates, but a 1–3B model under ambiguity is
still a 1–3B model. The demo's trap turn is the honest check — run it
against the model you actually plan to use. Models in the 8B+ class
are where the absence-handling shines.

**Cold start is real.** A memory layer with nothing in it can't save
you tokens yet. Budget a handful of turns before the deltas get
interesting.

**Port collisions happen.** If something already owns `11435`, run
`sieve start --port 11436` and point your client there instead.

## Five minutes, summarised

```bash
pipx install llm-sieve
sieve-install            # auto-detects Ollama, guided from there
export OLLAMA_HOST=http://127.0.0.1:11435
sieve demo               # watch the recall hits and the trap refusal
```

One URL changed, no SDK adopted, no client code rewritten — and your
Ollama models stop forgetting who you are between sessions.

---

*This post was drafted with AI assistance and reviewed by the Sieve
maintainer before publication. Quantitative claims link to their
source. Sieve is open source under Apache 2.0 —
[github.com/llmsieve/llm-sieve](https://github.com/llmsieve/llm-sieve).*

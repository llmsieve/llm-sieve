<p align="center">
  <img src="branding/sieve-lockup-light.svg#gh-light-mode-only" alt="Sieve" width="280">
  <img src="branding/sieve-lockup-dark.svg#gh-dark-mode-only" alt="Sieve" width="280">
</p>

<p align="center"><strong>Transparent context reduction for LLMs.</strong></p>

<p align="center">
  <a href="https://github.com/llmsieve/llm-sieve/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <a href="https://pypi.org/project/llm-sieve/"><img alt="PyPI" src="https://img.shields.io/pypi/v/llm-sieve.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg"></a>
  <a href="PATENT_NOTICE"><img alt="Patent" src="https://img.shields.io/badge/patent-pending-lightgrey.svg"></a>
</p>

---

## 95% fewer tokens per turn. 3–7× faster followups. Encrypted, local-first, BYO LLM.

Sieve sits between your agent framework and your LLM endpoint. It watches the traffic, extracts durable facts into an encrypted local store, and rewrites bloated prompts into lean, on-demand context. Your agent keeps talking to what looks like its usual endpoint — the model just stops drowning in repeated tool descriptions, stale history, and instructions it already knows.

- **Homepage:** [llmsieve.com](https://llmsieve.com)
- **Documentation:** [llmsieve.dev](https://llmsieve.dev)
- **Source:** [github.com/llmsieve/llm-sieve](https://github.com/llmsieve/llm-sieve)

## Why Sieve

Agent frameworks pay for context three ways: tokens billed per call, latency that grows with payload size, and accuracy that degrades as the prompt gets noisier. The usual response — "just use a bigger context window" — moves the cost rather than removing it.

Sieve removes it. A bloated system prompt full of tool schemas and stale turns becomes a lean payload backed by a memory store the model can consult on demand. The proxy is transparent: your agent does not change, your endpoint does not change, and the store stays on your machine.

## What Sieve does

- **Extracts facts** from your conversations into an encrypted local store (SQLCipher).
- **Classifies incoming queries** so retrieval is targeted, not scattershot.
- **Strips redundant context** — tool descriptions, stale history, repeated instructions — before the request reaches the LLM.
- **Retrieves on demand** through a tiered pipeline (fingerprint → vector → cross-encoder rerank).
- **Refuses to fabricate** when a fact is not in the store (absence-signal layer, on by default).
- **Ships self-contained.** Embeddings run in-process via FastEmbed (BAAI/bge-small-en-v1.5, ~50MB, auto-downloaded). Fact extraction uses your own LLM. No separate embedding service, no second model to host.

## Performance

- **95% fewer tokens per turn** — invariant across 5 LLM architectures (Granite, Llama, Qwen, Mistral, GPT-OSS), 8B-72B model sizes, 8K-64K context windows, and 1-64 concurrent sessions. Range: 92-96% across 33 measurements; no cell below 92%.
- **3-7× faster followups** — Sieve ships ~150 tokens per turn; baseline ships the full conversation history. On Llama-3.1-70B: 4.44s → 1.24s p50 on follow-ups. On Qwen-2.5-72B: 10.74s → 1.63s p50.
- **Up to 9× less hallucination** on absence-trap queries — driven by the absence-signal layer (on by default), which refuses to fabricate when a queried fact is not in the store.
- **Sub-15ms recall** at 100k facts with full production crypto (SQLCipher + partition keys).

Full methodology, per-category analysis, and the underlying data will be published in a forthcoming paper; a link will be added here when it's out.


## Quick start

```bash
pipx install llm-sieve
sieve-install
```

Two commands. The first puts `sieve` and `sieve-install` on your `PATH`. The second walks you through setup — finds your LLM, downloads the ~50 MB embedding model, writes an encrypted store under `~/.sieve/`, offers to start the proxy, and leaves you on a green "ready" panel.

When it's done, point your agent at `http://127.0.0.1:11435` instead of your usual LLM endpoint. That is the whole integration.

> Don't have `pipx`? `python3 -m pip install --user pipx && pipx ensurepath` installs it and puts it on your `PATH`. Or use plain `pip install llm-sieve` inside a virtualenv. See [Installation](https://llmsieve.dev/installation/) for every supported path.

<p align="center"><img src="branding/sieve-quick-install.gif" alt="One-command install demo" width="720"></p>

### Managing Sieve — the interactive menu

Running `sieve wizard` (or just `sieve` on its own) drops you into a menu for day-two operations: start/stop the proxy, inspect the store, reconfigure, run the benchmark, uninstall. Everything is state-aware — options that don't apply right now (e.g. "stop" when the proxy isn't running) are marked as such.

```bash
sieve wizard
```

<p align="center"><img src="branding/sieve-wizard.gif" alt="Interactive management menu" width="720"></p>

### Prove the numbers on your own machine

```bash
sieve benchmark
```

Runs a 15-turn scripted conversation against your configured model twice — once direct, once through Sieve — and prints the delta. Works against any Ollama or OpenAI-compatible endpoint. The prompts don't depend on the model knowing specific facts.

<p align="center"><img src="branding/sieve-benchmark.gif" alt="Benchmark — 90% fewer tokens on a real run" width="720"></p>

Full walkthrough: [Getting started](https://llmsieve.dev/getting-started/).

## How it works

Every request through the proxy passes through a five-step pipeline:

1. **Classify** — the incoming message is tagged (recall, statement, multi-hop, absence-probe) so retrieval is proportionate to what the turn needs.
2. **Strip** — tool schemas, repeated instructions, and stale conversation are removed from the outbound payload.
3. **Retrieve** — a tiered search over the local store produces the minimum facts the model needs, reranked by a cross-encoder.
4. **Compose** — a lean payload is assembled: preserved system intent, targeted facts, the current turn.
5. **Forward** — the lean payload goes to your LLM endpoint; the response is parsed, and durable facts are extracted into the store for next time.

Architecture deep-dive: [llmsieve.dev](https://llmsieve.dev).

## How Sieve fits in the landscape

Several excellent projects address different facets of the context-and-memory problem. Sieve is positioned alongside them, not against them.

> **Most memory systems solve storage. Sieve solves delivery.**

| Approach | What it does well | Integration | Sieve adds |
|---|---|---|---|
| Agent + raw context | Simple, no setup | N/A | Reduces bloat without changing the agent |
| Agent + compaction | Keeps context manageable | Built-in | More precise retrieval vs crude truncation |
| RAG systems | Document retrieval | Requires SDK integration | Transparent proxy, no code changes |
| Letta (MemGPT) | Virtual context management | Requires SDK | Drop-in proxy, works alongside |
| **Sieve** | Token reduction + memory | Transparent proxy | — |

Sieve is complementary. It works alongside any of these approaches — reducing what gets sent to the model regardless of how the context was assembled. Better memory + leaner delivery = better results.

## Try it without wiring anything up

Two scripted flows ship with the CLI. Both run against a **sandboxed store** — your real `~/.sieve/memory.db` is never touched.

**`sieve demo`** — a six-message scripted conversation that introduces an identity, seeds a couple of facts, asks Sieve to recall them, and ends with a question about a person who was never mentioned. You see recall hits on the seeded facts and a refusal on the trap question.

**`sieve benchmark`** — the same idea at 15 turns with per-message accounting. Prints a table of inbound vs outbound tokens, facts learned per message, time per turn, and the verdict on the absence-trap. Works against any model `sieve.yaml` points at; the prompts don't depend on the model knowing specific facts, so the comparison is always apples-to-apples.

The benchmark is what backs the hero numbers at the top — run it and see what you get on your own hardware.

## Scripting the CLI

Everything the wizard does is reachable from individual CLI commands, so you can script operations instead of driving the menu:

```bash
sieve status                           # running state + store counts
sieve config show                      # current config (non-defaults highlighted)
sieve config set listen.port 11500     # validated + type-coerced
sieve store facts --limit 10          # inspect what Sieve has learned
sieve store stats                      # per-table row counts
sieve backup create                    # encrypted snapshot with checksum
sieve key rotate                       # re-encrypt the store with a new key
sieve uninstall --hard                 # remove everything (requires typing DELETE)
```

Full list with every flag: [CLI reference](https://llmsieve.dev/cli-reference/).

## Configuration

After `sieve-install`, your config lives at `~/.sieve/sieve.yaml`. The shipping example — with commentary on every option — is [`sieve.example.yaml`](sieve.example.yaml).

Key settings:

| Setting | What it does |
|--------|--------------|
| `listen.port` | Port the proxy listens on (default `11435`). |
| `provider.base_url` | Your LLM endpoint. Any OpenAI-compatible or Ollama server. |
| `provider.default_model` | Model to call when the agent does not pin one. |
| `embeddings.provider` | `fastembed` (default, in-process) or `ollama`. |
| `store.path` | Where the encrypted memory store lives (default `~/.sieve/memory.db`). |
| `writer.model` | `auto` routes fact extraction to `provider.default_model`; override for a dedicated writer. |
| `profile_owner.name` | Canonical identity pinned into fact extraction and validation. |

Full reference: [Configuration](https://llmsieve.dev/configuration/).

## Compatibility

**LLM providers.** Any OpenAI-compatible endpoint works. Sieve has been exercised against Ollama, vLLM, LM Studio, and hosted APIs (OpenAI, Anthropic via gateway). Point `provider.base_url` at the endpoint; point `provider.default_model` at the model name.

**Hardware.**

- Consumer GPU (12 GB+): runs a local LLM and Sieve side by side comfortably.
- Apple Silicon: Ollama + Sieve tested on M-series.
- CPU-only: use a hosted LLM for inference; Sieve itself is lightweight and runs in-process.
- Cloud: any Linux host; the store is a single encrypted SQLite file.

**Tested models.**

Qwen, DeepSeek, Llama, Mistral, GPT-OSS, Gemma, and Granite variants have been exercised through Ollama and OpenAI-compatible endpoints across the 8B–72B size range. Reasoning-token support (`options.think`) is gated on the model family.

## Security and privacy

- **Encrypted store.** All facts live in a SQLCipher-encrypted SQLite database. The keyfile is written alongside the store on first init.
- **Local-first.** Your conversation history and extracted facts never leave your machine. The only outbound traffic is whatever your LLM endpoint already makes.
- **Zero telemetry.** Sieve does not phone home. There is no analytics, no update check, no metrics endpoint that talks to us.
- **No account.** No login, no API key for Sieve itself. Bring your own LLM.

Reporting a vulnerability: do not open a public issue — see [SECURITY.md](SECURITY.md).

## Open source

Apache License 2.0. You are free to use, modify, and distribute Sieve, including commercially, under the terms in [LICENSE](LICENSE).

Sieve implements inventions covered by a pending UK patent application ([GB2608859.1](PATENT_NOTICE)). Apache 2.0 includes an automatic, royalty-free patent licence for users of the software — see the [patent notice](PATENT_NOTICE) for the terms.

## Contributing

Issues, bug reports, and pull requests are welcome. Contributions require a DCO sign-off (`git commit -s`). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you reference Sieve in academic work, please cite:

```bibtex
@software{sieve2026,
  author  = {Tennant-Hosein, Azard},
  title   = {Sieve: Transparent Context Reduction for LLMs},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/llmsieve/llm-sieve},
  note    = {Apache-2.0; UK patent pending GB2608859.1}
}
```

## License

[Apache License 2.0](LICENSE). See [PATENT_NOTICE](PATENT_NOTICE) for patent terms.

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

## Up to 97% token reduction. Up to 9× less hallucination. Gets smarter over time.

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

- **Up to 97% token reduction** on large agent payloads (96.9% outbound measured on 30-day progressive-activation run).
- **Up to 9× less hallucination** on absence-trap queries — questions about facts that were never stored (9.3× measured).
- **Gets smarter over time.** By Days 21–30 of the validation run, Sieve's answer accuracy overtakes the baseline — the store is now dense enough that retrieved facts beat the model's own context.
- **Validated across two independent runs:** 30-day progressive-activation run on qwen3:30b-a3b and 60-day longitudinal run on qwen3:14b, with cross-family grading.

Full methodology and detailed analysis will be published in a forthcoming paper. See [`evaluation/RESULTS_SUMMARY.md`](evaluation/RESULTS_SUMMARY.md) for headline figures.

<p align="center">
  <img src="docs/figures/token-divergence.svg#gh-dark-mode-only" alt="Context growth over 60 days — Baseline vs Sieve" width="720">
  <img src="docs/figures/token-divergence-light.svg#gh-light-mode-only" alt="Context growth over 60 days — Baseline vs Sieve" width="720">
</p>

<p align="center">
  <img src="docs/figures/hallucination-bars.svg#gh-dark-mode-only" alt="Hallucination: Baseline vs Sieve — 9.3× less" width="720">
  <img src="docs/figures/hallucination-bars-light.svg#gh-light-mode-only" alt="Hallucination: Baseline vs Sieve — 9.3× less" width="720">
</p>

<p align="center">
  <img src="docs/figures/accuracy-crossover.svg#gh-dark-mode-only" alt="Sieve gets smarter over time — accuracy by day-bucket" width="720">
  <img src="docs/figures/accuracy-crossover-light.svg#gh-light-mode-only" alt="Sieve gets smarter over time — accuracy by day-bucket" width="720">
</p>

<p align="center">
  <img src="docs/figures/hallucination-divergence.svg#gh-dark-mode-only" alt="Hallucination rate over 60 days" width="720">
  <img src="docs/figures/hallucination-divergence-light.svg#gh-light-mode-only" alt="Hallucination rate over 60 days" width="720">
</p>

## Quick start

```bash
pip install llm-sieve
sieve init        # zero-prompt setup; add --wizard to be asked about each option
sieve start       # proxy listens on http://127.0.0.1:11435
```

Point your agent at `http://127.0.0.1:11435` instead of your usual LLM endpoint. That is the whole integration.

### One command

<p align="center"><img src="branding/sieve-quick-install.gif" alt="Quick install demo" width="720"></p>

### Guided setup

If you prefer to be asked about each option — provider URL, model, port, encryption, store location — use the wizard. At the end it offers to run the benchmark below so you can verify token reduction on your own machine before you wire anything up.

```bash
sieve init --wizard
```

<p align="center"><img src="branding/sieve-wizard-install.gif" alt="Wizard install demo" width="720"></p>

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

## Demo mode

With the proxy running, open another terminal:

```bash
sieve demo
```

It runs a six-message scripted conversation — a new identity introduces themselves, shares a couple of facts, asks Sieve to recall them, and then asks about a person who was never mentioned. You will see recall hits on the seeded facts, and a refusal on the absence-trap question.

## Benchmark

Don't trust the numbers at the top — run the benchmark yourself:

```bash
sieve benchmark
```

It runs a 15-message scripted conversation (introduce facts → ask retrieval questions → deeper follow-ups → temporal updates → a trap query about something that was never mentioned) and prints a per-message table plus overall totals. You get:

- **Per-turn inbound vs outbound tokens** (the proxy reports both directly)
- **Facts learned per message** (polled from the store before and after each turn)
- **Time per turn**
- **Whether the absence-signal layer fired on the trap** (fuzzy but explicit — both the fact-count delta and the response text are shown)

The benchmark works against any OpenAI-compatible or Ollama-compatible model pointed at by `sieve.yaml` — the prompts do not depend on the model knowing specific facts.

## Managing Sieve

The CLI covers everyday operations without editing YAML:

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

After `sieve init`, your config lives at `~/.sieve/sieve.yaml`. The shipping example — with commentary on every option — is [`sieve.example.yaml`](sieve.example.yaml).

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

- `qwen3:30b-a3b` — 30-day longitudinal run.
- `qwen3:14b` — 60-day longitudinal run.

Other Qwen, DeepSeek, Llama, Mistral, and Gemma variants have been smoke-tested through Ollama. Reasoning-token support (`options.think`) is gated on the model family.

## Security and privacy

- **Encrypted store.** All facts live in a SQLCipher-encrypted SQLite database. The keyfile is written alongside the store on first init.
- **Local-first.** Your conversation history and extracted facts never leave your machine. The only outbound traffic is whatever your LLM endpoint already makes.
- **Zero telemetry.** Sieve does not phone home. There is no analytics, no update check, no metrics endpoint that talks to us.
- **No account.** No login, no API key for Sieve itself. Bring your own LLM.

Reporting a vulnerability: do not open a public issue — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Evaluation

Sieve was validated across two independent longitudinal runs — 30 days on `qwen3:30b-a3b` and 60 days on `qwen3:14b` — with cross-family grading (a different model family checks the answers than the one being tested). Headline numbers are in [`evaluation/RESULTS_SUMMARY.md`](evaluation/RESULTS_SUMMARY.md). Full methodology and detailed analysis will be published in a forthcoming paper.

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

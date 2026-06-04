# Changelog

All notable changes to this project will be documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-04

First public release.

Sieve is a transparent proxy that sits between your agent framework and your
LLM endpoint. It rewrites bloated prompts into lean, on-demand context backed
by an encrypted local memory store — without changing your agent or your
endpoint.

### Headline numbers

- **95% fewer tokens per turn**, measured invariant across 5 LLM
  architectures, 8B–72B model sizes, 8K–64K context windows, and 1–64
  concurrent sessions.
- **3–7× faster follow-ups** on frontier models.
- **Up to 9× less hallucination** on absence-trap queries.
- **Sub-15 ms recall** at 100,000 facts with full production crypto.

### Install

```bash
pipx install llm-sieve
sieve-install
```

Then point your agent at `http://127.0.0.1:11435` instead of your usual LLM
endpoint. That is the whole integration.

### What ships

- **`sieve-install`** — one-command first-run setup. Detects Anthropic /
  OpenAI / OpenAI-compatible / Ollama / custom providers; picks a model;
  downloads the embedding model; initialises the encrypted store;
  optionally enables autostart.
- **`sieve`** — day-to-day CLI. Start / stop / restart / status / demo /
  benchmark / update, plus subcommand groups for `store`, `config`,
  `key`, and `backup`.
- **`sieve wizard`** — state-aware interactive menu for day-two operations.
- **`sieve demo`** — sandboxed 6-turn scripted conversation that
  demonstrates fact extraction and the absence-signal trap.
- **`sieve benchmark`** — reproducible baseline-vs-Sieve comparison with
  multi-run aggregation and a shareable markdown report.
- **`sieve update`** — on-request PyPI check. Zero auto-telemetry.

### Architecture

- FastAPI proxy on a configurable port (default `11435`).
- In-process FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime, ~50 MB) for
  embeddings — no separate embedding service to run.
- Three-tier retrieval pipeline: fingerprint → vector → cross-encoder
  rerank. Query decomposition for multi-hop.
- Three-phase progressive-activation lifecycle (OBSERVE → ACCUMULATE →
  ACTIVATE) so cold-start behaviour doesn't degrade answer quality before
  the store has enough material.
- Absence-signal layer (on by default) that refuses to fabricate on facts
  not in the store.
- SQLCipher-encrypted local memory store with key rotation, encrypted
  backups, and schema versioning.

### Privacy and licensing

- **Zero telemetry.** Sieve does not phone home. The only outbound traffic
  is whatever your LLM endpoint already makes; `sieve update` talks to
  PyPI only when you invoke it.
- **Local-first.** Your conversation history and extracted facts never
  leave your machine.
- **Apache License 2.0.** Patent pending — UK patent application
  GB2608859.1 (filed 16 April 2026). See [PATENT_NOTICE](PATENT_NOTICE).

### Compatibility

- Python 3.11, 3.12, 3.13
- Linux, macOS (Intel + Apple Silicon), Windows via WSL2
- Any OpenAI-compatible or Ollama endpoint

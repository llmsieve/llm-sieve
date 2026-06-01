# Changelog

All notable changes to this project will be documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — Unreleased

First public release of Sieve — a transparent proxy that sits between an
agent framework and an LLM endpoint, rewriting bloated system prompts
into lean on-demand context backed by an encrypted local memory store.

### Highlights

- **95% fewer tokens per turn** — measured invariant across 5 LLM
  architectures (Granite, Llama, Qwen, Mistral, GPT-OSS), 8B–72B model
  sizes, 8K–64K context windows, and 1–64 concurrent sessions.
- **3–7× faster follow-ups** on frontier models — Sieve ships ~150
  tokens per turn; baseline ships the full conversation history.
- **Up to 9× less hallucination** on absence-trap queries — driven by
  the absence-signal layer (on by default), which refuses to fabricate
  when a queried fact is not in the store.
- **Sub-15 ms recall** at 100,000 facts with full production crypto
  (SQLCipher).

### What's included

- `sieve-install` — one-command first-run setup with provider
  detection (Anthropic / OpenAI / OpenAI-compatible / Ollama / Custom),
  model picker, embedding-model download, encrypted store
  initialisation, and optional autostart.
- `sieve` CLI — `start` / `stop` / `restart` / `status` / `demo` /
  `benchmark` / `wizard` / `init` / `uninstall`, plus subcommand
  groups for `store`, `config`, `key`, and `backup` operations.
- `sieve wizard` — state-aware interactive management menu for
  day-two operations.
- `sieve demo` — sandboxed 6-turn scripted conversation that
  demonstrates fact extraction and the absence-signal trap.
- `sieve benchmark` — reproducible baseline-vs-Sieve comparison with
  an interactive preflight wizard, multi-run aggregation, and
  shareable markdown reports.

### Architecture

- FastAPI proxy server on configurable port (default `11435`).
- In-process FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime,
  ~50 MB) for embeddings — no separate embedding service to host.
- Three-tier retrieval pipeline (fingerprint → vector → cross-encoder
  rerank).
- Query decomposition for multi-hop retrieval.
- Three-phase progressive-activation lifecycle (OBSERVE →
  ACCUMULATE → ACTIVATE) so cold-start behaviour doesn't degrade
  answer quality before the store has enough material.
- Absence-signal layer that refuses to fabricate on facts not in the
  store.
- SQLCipher-encrypted local memory store with key rotation, export,
  and backup tooling.

### Install

```bash
pipx install llm-sieve
sieve-install
```

Point your agent at `http://127.0.0.1:11435` instead of your usual
LLM endpoint. That is the whole integration.

### Compatibility

- Python 3.11+
- Linux, macOS (Intel + Apple Silicon), Windows via WSL2
- Any OpenAI-compatible or Ollama endpoint

### Licensing

Apache License 2.0. Patent pending — UK patent application
GB2608859.1 (filed 16 April 2026). See [PATENT_NOTICE](PATENT_NOTICE).

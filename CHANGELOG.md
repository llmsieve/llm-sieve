# Changelog

All notable changes to this project will be documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Progressive activation

The retrieval pipeline now follows a three-phase lifecycle — OBSERVE →
ACCUMULATE → ACTIVATE — so cold-start behaviour no longer degrades
answer quality before the store has enough material to beat passthrough.
Validated on a 30-day run against `qwen3:30b-a3b`:

- Up to 97% token reduction on large agent payloads (96.9% measured)
- Up to 9× less hallucination on absence-trap queries (9.3× measured)
- Sieve accuracy overtakes baseline by Days 21–30 (+0.012) — Sieve gets
  smarter over time as the store fills

## [1.0.0] — 2026-04-19

### First public release

Sieve is a transparent proxy that sits between an agent framework and an
LLM endpoint. It rewrites bloated system prompts into lean on-demand
context, backed by an encrypted local memory store.

### Highlights

- Up to 97% token reduction on large agent payloads
- Up to 9× less hallucination on absence-trap queries
- Self-contained: FastEmbed (in-process ONNX) for embeddings, user's own
  LLM for fact extraction. No external model dependencies beyond the LLM
  endpoint you already use.
- Works with any OpenAI-compatible or Ollama endpoint
- Encrypted local-first memory store (SQLCipher)
- 880+ automated tests

### What's included

- `sieve` CLI with `init`, `start`, `stop`, `status`, `demo`,
  `store init/status/migrate`, `backup create/list/restore`
- FastAPI proxy server on configurable port (default 11435)
- Three-tier retrieval pipeline with cross-encoder reranking
- Absence-signal layer (ships on) — refuses to fabricate when facts
  aren't in the store
- Query decomposition for multi-hop retrieval
- Opt-in validation metrics collector

### Patent

Patent pending — UK application GB2608859.1 (filed 16 April 2026).
See [PATENT_NOTICE](PATENT_NOTICE).

### License

Apache License 2.0.

# Changelog

All notable changes to this project will be documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — release candidate (v1.0.0-rc1)

### Installer — 5-branch provider picker (closes Ollama bias)

The first-run installer no longer assumes Ollama. The wizard's first
question is now an explicit pick of the provider you have:

  1. Anthropic (Claude)
  2. OpenAI
  3. OpenAI-compatible endpoint (OpenRouter, Groq, vLLM, LM Studio, …)
  4. Ollama (local)
  5. Custom URL

Each branch handles URL + key + probe per its conventions, with
provider-specific model defaults so cloud branches succeed without
hitting `/v1/models` listing endpoints that Anthropic and OpenAI
don't expose anonymously.

`sieve-install --no-input` now auto-detects in priority order:
ANTHROPIC_API_KEY → OPENAI_API_KEY → local Ollama. On total miss it
prints a richer error with concrete next-steps instead of a one-line
refusal.

### Writer — `<think>`-tag defence + skip-empty classifier

* `writer.py` now strips `<think>...</think>` blocks before its JSON
  parse, defending against reasoning models that emit traces even
  when asked not to. Closes a real silent-fact-drop hole observed
  on `gpt-oss:20b` via Ollama cloud (~3-5/30 turns affected).
* New `sieve/_writer_classifier.py` skips the writer LLM call
  entirely on turns that can't contain extractable facts (filler,
  pure questions with no proper-noun anchor, social greetings).
  Measured to skip ~70-80% of representative traffic with no
  fact-share regressions. Controlled by `writer.skip_empty_turns:
  true` in config (default on).
* New wizard question asks whether to use the same model for the
  writer step (default) or a dedicated smaller one. Warns when
  the chosen target looks like a reasoning model.

### Docs — Phase 3 numbers + brand-respectful landing page

`docs/index.md` rewritten to lead with the empirical headline pitch
backed by the Phase 3 release-candidate evidence (recall repo tag
`v1.1.0-phase3-rc`):

* **95% fewer tokens** — invariant across 5 LLM architectures, 8B-72B
  model sizes, 8K-64K windows, 1-64 concurrency
* **3-7× faster followups** — measured p50s on Llama-70B and Qwen-72B
* **Sub-15ms recall** at 100k facts with full production crypto

The layout uses grid cards, a confidence-via-transparency "honest
scope" section, and a discreet desaturated-navy divider style — all
inside the brand book (teal reserved for the mark, sentence case,
no drop shadows). Adds the previously-orphan `diagnostic-headers.md`
to the nav.

### Sundry

* `sieve.__version__` now exposed via `importlib.metadata`. Closes
  `AttributeError: module 'sieve' has no attribute '__version__'`.
* `sieve init` URL handling: auto-prepends `http://` to bare
  hostnames before probing. Closes the confusing "Request URL is
  missing an 'http://' or 'https://' protocol" error on valid input.
* `sieve init` success message now shows configured provider + listen
  URL + a hint about `sieve demo`.

### Methodology

The skip-empty classifier was motivated by a 60-cell measurement
battery (writer × hardware × optimisation conditions). That research
+ its design doc, results writeup, test harness, and the pre-release
audit live in the private evaluation repository, not in this package.

## [Earlier unreleased work]

### `sieve-install` — one-command first-run

New dedicated entry point that wraps the full setup flow: provider
detection (local Ollama → LAN → cloud), model picker, embedding-model
download, store initialisation, optional autostart, and a "Sieve is
ready" panel. Rolls back on Ctrl-C or any intermediate failure.

```bash
pipx install llm-sieve
sieve-install
```

`sieve init` still works for backward compatibility.

### Interactive management menu

`sieve` (or `sieve wizard` explicitly) now opens a state-aware menu
for day-two operations: service control, store inspection, config
editing, benchmark, demo, reinstall, uninstall. Options that don't
apply to the current state are shown as unavailable rather than
hidden.

### Benchmark UX

- Sandboxed by default — `sieve benchmark` runs against a scratch
  store that's torn down afterwards, leaving the user's real store
  untouched. Pass `--use-main-store` for the old behaviour.
- Preflight wizard for interactive runs (fixture size, model,
  grader, turns, runs, pricing).
- Context preflight panel that explains what will happen when
  baseline payload exceeds the model's effective `num_ctx`.
- `--grader-model` flag for independent scoring (self-grading is
  flagged in the shareable report).

### Progressive activation

The retrieval pipeline now follows a three-phase lifecycle — OBSERVE →
ACCUMULATE → ACTIVATE — so cold-start behaviour no longer degrades
answer quality before the store has enough material to beat passthrough.
Validated on a 30-day run against `qwen3:30b-a3b`:

- Up to 97% token reduction on large agent payloads (96.9% measured)
- Up to 9× less hallucination on absence-trap queries (9.3× measured)
- Sieve accuracy overtakes baseline by Days 21–30 (+0.012) — Sieve gets
  smarter over time as the store fills

### Branding

Three demo GIFs rendered via VHS + a bespoke Docker image — quick
install, wizard navigation, and a real benchmark-run replay. See
[`branding/README.md`](https://github.com/llmsieve/llm-sieve/blob/main/branding/README.md) for the render pipeline.

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
See [PATENT_NOTICE](https://github.com/llmsieve/llm-sieve/blob/main/PATENT_NOTICE).

### License

Apache License 2.0.

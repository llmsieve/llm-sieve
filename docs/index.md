---
hide:
  - navigation
  - toc
---

# Sieve

**Transparent context reduction for LLMs.**

Sieve is a proxy between your agent framework and your LLM endpoint. It strips tool schemas, repeated instructions, and stale history from every outbound turn, rewrites the prompt into a lean payload, and backs retrieval with an encrypted local memory store. Your agent doesn't change. Your endpoint doesn't change. The model just stops drowning in the same 20,000 tokens every turn.

<div class="sieve-divider"></div>

## What you get

<div class="grid cards" markdown>

-   ### 95% fewer tokens

    Measured invariant across **5 LLM architectures** (Granite, Llama, Qwen, Mistral, GPT-OSS), **2 model-size classes** (8B → 72B), **4 context windows** (8K → 64K), and **5 concurrency levels** (1 → 64).

    Range: **92–96%** on 33 measurements; no cell below 92%.

-   ### 3–7× faster followups

    Sieve ships ~150 tokens per turn; baseline ships the full conversation history. Latency drops automatically.

    On Llama-3.1-70B at concurrency=1: **4.44s → 1.24s** p50 on follow-ups. On Qwen-2.5-72B: **10.74s → 1.63s** p50.

-   ### Zero data leaves your machine

    Encrypted local store (SQLCipher) for facts and retrieval. No telemetry, no phone-home.

    You bring your own LLM endpoint — Sieve is indifferent to local vs hosted. **Sub-15 ms recall** at 100k facts with full production crypto.

</div>

<div class="sieve-divider"></div>

## Two commands to ship

```bash
pipx install llm-sieve
sieve-install
```

Point your agent at `http://127.0.0.1:11435` instead of your usual LLM endpoint. That's the whole integration.

The installer asks five short questions: provider type, endpoint, API key (if applicable), model, and writer choice. It auto-detects `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / a running local Ollama and offers them as one-keystroke happy paths. See [Getting started](getting-started.md) for the walk-through.

## Honest scope

- **Sieve doesn't ship a bundled writer model.** You bring an LLM endpoint; Sieve uses it both for serving your agent and for its internal extraction step. This is by design — a tested in-process small writer doesn't yet meet our quality bar, so we route through the model you already trust. See [the architecture](https://github.com/llmsieve/llm-sieve/blob/main/ARCHITECTURE.md) for why.
- **Sieve runs on CPU.** Embeddings are FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime, ~50 MB), in-process, no GPU needed.
- **Sieve doesn't change your model's outputs.** Streaming, tool calls, and structured outputs all pass through unchanged. Sieve only modifies the *prompt* going in.

## What's not in this docs site

For the marketing site — animated diagrams of Sieve at work, demos, and the full pitch — visit [**llmsieve.com**](https://llmsieve.com). This site (`llmsieve.dev`) is the developer reference.

<div class="sieve-divider"></div>

## Reference

<div class="grid cards" markdown>

-   ### Get going

    [**Getting started**](getting-started.md) — install, run `sieve-install`, send your first request.

    [**Installation**](installation.md) — every supported install path, with platform notes.

-   ### Operate

    [**Configuration**](configuration.md) — every option in `sieve.yaml`.

    [**CLI reference**](cli-reference.md) — every command and flag.

    [**Diagnostic headers**](diagnostic-headers.md) — request-level visibility for debugging.

-   ### Build

    [**Source on GitHub**](https://github.com/llmsieve/llm-sieve) — code, issues, releases.

    [**Architecture**](https://github.com/llmsieve/llm-sieve/blob/main/ARCHITECTURE.md) — the request pipeline, store schema, conflict resolution.

    [**Changelog**](changelog.md) — what changed and why.

</div>

---

**Apache 2.0**, patent pending. See [PATENT_NOTICE](https://github.com/llmsieve/llm-sieve/blob/main/PATENT_NOTICE).

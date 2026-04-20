# Sieve

**Transparent context reduction for LLMs.**

Sieve is a proxy between your agent framework and your LLM endpoint. It strips tool schemas, repeated instructions, and stale history from every outbound turn, rewrites the prompt into a lean payload, and backs retrieval with an encrypted local memory store. Your agent doesn't change. Your endpoint doesn't change. The model just stops drowning in the same 20,000 tokens every turn.

- **Up to 97% token reduction** on large agent payloads (96.9% outbound on a 30-day progressive-activation run)
- **Up to 9× less hallucination** on absence-trap queries (9.3× measured)
- **Self-contained** — FastEmbed in-process + your own LLM. No second embedding service, no separate writer model.
- **Works with any OpenAI-compatible or Ollama endpoint**
- **Encrypted local-first memory store** (SQLCipher)
- **Apache 2.0**, patent pending

---

## Start here

- [**Getting started**](getting-started.md) — install, run `sieve-install`, send your first request
- [**Installation**](installation.md) — every supported path + platform notes
- [**Configuration**](configuration.md) — every option in `sieve.yaml`
- [**CLI reference**](cli-reference.md) — every command and flag

## Two commands to ship

```bash
pipx install llm-sieve
sieve-install
```

Point your agent at `http://127.0.0.1:11435` instead of your usual LLM endpoint. That's the whole integration.

## Further reading

- Repository: [github.com/llmsieve/llm-sieve](https://github.com/llmsieve/llm-sieve)
- Homepage: [llmsieve.com](https://llmsieve.com)
- Changelog: [CHANGELOG.md](https://github.com/llmsieve/llm-sieve/blob/main/CHANGELOG.md)
- Patent notice: [PATENT_NOTICE](https://github.com/llmsieve/llm-sieve/blob/main/PATENT_NOTICE)

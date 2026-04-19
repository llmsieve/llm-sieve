# Sieve

**Transparent context reduction for LLMs.**

Sieve is a proxy between your agent framework and your LLM endpoint. It rewrites bloated system prompts into lean, on-demand context, backed by an encrypted local memory store.

- **Up to 88% token reduction** on large agent payloads
- **Up to 6× less hallucination** on absence-trap queries
- **Self-contained** — FastEmbed (in-process) + your own LLM
- Works with any OpenAI-compatible or Ollama endpoint
- Encrypted local-first memory store (SQLCipher)
- Apache 2.0, patent pending

---

## Start here

- [**Getting started**](getting-started.md) — install, initialise, send your first request
- [**Installation**](installation.md) — platform notes and provider setups
- [**Configuration**](configuration.md) — every option in `sieve.yaml` documented

## Integration in three lines

```bash
pip install llm-sieve
sieve init
sieve start
```

Point your agent at `http://127.0.0.1:11435` instead of your usual LLM endpoint. That is the whole integration.

## Further reading

- Repository: [github.com/llmsieve/llm-sieve](https://github.com/llmsieve/llm-sieve)
- Homepage: [llmsieve.com](https://llmsieve.com)
- Changelog: [CHANGELOG.md](https://github.com/llmsieve/llm-sieve/blob/main/CHANGELOG.md)
- Patent notice: [PATENT_NOTICE](https://github.com/llmsieve/llm-sieve/blob/main/PATENT_NOTICE)

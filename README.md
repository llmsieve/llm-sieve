# Sieve

Transparent context reduction for LLMs.

Sieve is a drop-in proxy between an agent framework and an LLM endpoint.
It rewrites bloated system prompts into lean, on-demand context backed by
an encrypted local memory store.

- **Up to 88% token reduction** on large agent payloads
- **Up to 6× less hallucination** on absence-trap queries
- **Self-contained** — FastEmbed (in-process ONNX) + your own LLM. No
  external embedding service required.
- **Works with any OpenAI-compatible or Ollama endpoint**
- **Encrypted local-first** memory store (SQLCipher)
- **Apache 2.0** license, patent pending ([GB2608859.1](PATENT_NOTICE))

## Install

```bash
pip install llm-sieve
sieve init
sieve start
```

## Configuration

Copy `sieve.example.yaml` to `~/.sieve/sieve.yaml` and edit.

## License

[Apache 2.0](LICENSE). See [PATENT_NOTICE](PATENT_NOTICE) for patent terms.

More documentation coming in a follow-up release.

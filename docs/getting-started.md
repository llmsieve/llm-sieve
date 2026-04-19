# Getting started

This walkthrough takes you from a fresh machine to a running Sieve proxy with an agent talking through it.

## Prerequisites

- **Python 3.11 or newer.** Verify with `python --version`.
- **An LLM endpoint.** Either a local server (Ollama, vLLM, LM Studio) or a hosted OpenAI-compatible API. The rest of this guide assumes Ollama on `http://127.0.0.1:11434`; substitute your endpoint freely.
- **~200 MB free disk.** ~50 MB for the FastEmbed embedding model, the rest for Sieve, dependencies, and the initial store.

## Step 1 — Install

```bash
pip install llm-sieve
```

Check the install landed:

```bash
sieve --version
```

You should see `sieve, version 1.0.0` (or later).

## Step 2 — Initialise

```bash
sieve init
```

`sieve init` is a one-shot, zero-prompt setup. It:

1. **Detects your LLM provider.** If Ollama is reachable on `127.0.0.1:11434`, it is auto-selected. Otherwise you are prompted for a base URL.
2. **Health-checks the provider.** A non-fatal warning is printed if the endpoint does not respond; you can fix the URL in the config later.
3. **Downloads the embedding model.** `BAAI/bge-small-en-v1.5` (~50 MB ONNX) is fetched and cached by FastEmbed. This happens once per machine.
4. **Writes `~/.sieve/sieve.yaml`.** Based on the packaged `sieve.example.yaml`, with your provider URL substituted.
5. **Creates the encrypted store.** A SQLCipher database at `~/.sieve/memory.db` with its keyfile at `~/.sieve/.sieve_key`.

When it prints **Ready!**, you are done.

### Prefer to be asked about every option?

```bash
sieve init --wizard
```

Drops you into an interactive six-step setup covering provider choice (Ollama / OpenAI / Anthropic / custom), model selection, listen port, encryption (generated key or custom passphrase), and store location. Every choice defaults to the same value `sieve init` would have picked, so pressing Return through the wizard is equivalent to the lazy path. See the [CLI reference](cli-reference.md#sieve-init) for the full flow.

## Step 3 — Start the proxy

```bash
sieve start
```

Sieve binds to `127.0.0.1:11435` by default and forwards to your provider. Logs stream to the terminal; a PID file lives at `~/.sieve/sieve.pid` so `sieve stop` and `sieve status` can find the process.

Leave this terminal running and open a new one for the next steps.

## Step 4 — Point your agent at Sieve

Wherever your agent is configured to talk to an LLM, swap the base URL to `http://127.0.0.1:11435`. The Sieve proxy speaks the same wire protocol as the upstream — your agent does not need to know Sieve is there.

Concretely, for an OpenAI-compatible client:

```python
client = OpenAI(
    base_url="http://127.0.0.1:11435/v1",
    api_key="not-used-by-sieve",
)
```

For an Ollama client, set `OLLAMA_HOST=http://127.0.0.1:11435` (or the equivalent config in your client).

## Step 5 — Send a test query

The simplest check is `sieve demo`:

```bash
sieve demo
```

It runs a scripted six-message conversation that introduces an identity, states a few facts, asks Sieve to recall them, and ends with an absence-trap question about a person who was never mentioned. You should see recall hits on the seeded facts and a refusal on the absence-trap turn.

Alternatively, send a request of your own with `curl`:

```bash
curl http://127.0.0.1:11435/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "Hi, my name is Alex."}],
    "stream": false
  }'
```

## What to expect

**Cold start.** The first few turns of a new conversation will feel like a plain pass-through — Sieve has nothing to retrieve yet. Token savings and targeted recall kick in as the store fills up. You can watch this happen with `sieve status`, which prints the number of facts and entities stored.

**Warm operation.** Once the store has content, tool schemas and repeated instructions start being stripped, relevant facts start being injected on demand, and absence-probe questions begin being refused rather than fabricated.

**Per-turn diagnostics.** Sieve adds response headers to every turn — `X-Sieve-Rounds`, `X-Sieve-Proxy-Us`, and a few others — which are safe to log if you want to watch what the proxy is doing.

## Next steps

- Read the [configuration reference](configuration.md) to tune the pipeline.
- Browse the [installation guide](installation.md) for platform-specific notes (Apple Silicon, WSL, cloud providers).
- Bookmark `sieve status` and `sieve stop`. That is the whole day-two operational surface.

## Troubleshooting

**`sieve start` says the port is already in use.** Another process is bound to 11435. Either stop it or start Sieve elsewhere: `sieve start --port 11436`.

**The proxy starts but the provider is unreachable.** Edit `~/.sieve/sieve.yaml` — `provider.base_url` — to point at your actual endpoint, then `sieve stop && sieve start`.

**The embedding model failed to download.** Re-run `sieve init`; it is idempotent. If a firewall is blocking the model host, use a machine with outbound HTTPS to seed the FastEmbed cache (`~/.cache/fastembed`) and copy it over.

# Installation

This page covers every supported way to install Sieve, platform-specific notes, and how to wire it to your LLM provider.

For the fastest path, jump to [Getting started](getting-started.md). This page is for people who need detail.

## System requirements

- **Python 3.11 or 3.12.** Older Pythons are not supported.
- **Linux, macOS (Intel or Apple Silicon), or Windows via WSL.** Windows native is not a supported target.
- **~200 MB free disk.** ~50 MB for the embedding model; the rest for Sieve, dependencies, and the initial store.
- **An LLM endpoint.** OpenAI-compatible or Ollama. You can use a local server or a hosted API; Sieve is indifferent.

No GPU is required for Sieve itself — embeddings run on CPU via FastEmbed. If you are running a local LLM as your provider, that component has its own hardware requirements.

## Install from PyPI (recommended)

```bash
pip install llm-sieve
```

The distribution is `llm-sieve`; the command it installs is `sieve`.

We strongly recommend installing into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate.bat    # Windows PowerShell / cmd (WSL users: use the bash line)

pip install llm-sieve
```

## Install from source

For development, or to run an unreleased revision:

```bash
git clone https://github.com/llmsieve/llm-sieve.git
cd llm-sieve
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The `[dev]` extra pulls in `pytest`, `pytest-asyncio`, `pytest-httpx`, and `ruff`. The main runtime has no test-only dependencies.

## Platform notes

### Linux

The reference platform. `pip install llm-sieve` should be the entire install on any reasonably recent distribution.

SQLCipher is pulled in via the `sqlcipher3` wheel and does not require system packages on supported Python versions. If your distribution ships a very old `libsqlite3`, install into a venv using a current Python rather than relying on the system interpreter.

### macOS

Both Intel and Apple Silicon are supported.

- **Apple Silicon.** FastEmbed uses ONNX Runtime which has native arm64 wheels — no Rosetta translation needed. If you are running Ollama locally as your provider, Ollama's Metal backend will use the GPU automatically.
- **Intel.** Nothing special required. Use a current Python from python.org or Homebrew.

### Windows

Use **WSL2 with a Linux distribution** (Ubuntu 22.04 or 24.04 are well-tested). Sieve is not currently packaged for native Windows; the SQLCipher and FastEmbed toolchains behave differently on Win32, and we do not ship wheels for that target.

From inside WSL, the Linux instructions apply unchanged.

## Provider setups

### Local LLM with Ollama

If you already run Ollama, there is nothing extra to do — `sieve init` auto-detects it on `127.0.0.1:11434`. Pull a model Sieve will use for fact extraction and general inference:

```bash
ollama pull qwen3.5:9b     # tested default
# or a bigger model if your hardware allows:
ollama pull qwen3:14b
```

Then run `sieve init` and accept the defaults.

### Local LLM with vLLM or LM Studio

Both expose an OpenAI-compatible HTTP server. Point `sieve init` at the base URL when prompted:

```bash
sieve init --provider http://127.0.0.1:8000
```

vLLM's default port is 8000; LM Studio's server is configurable. The model name you use in your agent code must match a model the server is actually serving.

### Hosted LLM with an OpenAI-compatible endpoint

Any hosted endpoint that speaks the OpenAI API works. Examples of base URLs you might pass to `--provider`:

- **OpenAI:** `https://api.openai.com/v1`
- **Anthropic via a gateway** (LiteLLM, OpenRouter, etc.): the gateway's URL
- **Self-hosted gateway:** whatever URL your gateway exposes

Auth for the upstream is configured on the upstream itself (e.g. `OPENAI_API_KEY` in your agent, which Sieve forwards). Sieve does not intercept or require its own API key.

## The FastEmbed embedding model

On first init, Sieve downloads **BAAI/bge-small-en-v1.5** — a 384-dimensional English embedding model, ~50 MB as ONNX — and caches it under `~/.cache/fastembed`. The download happens once per machine.

If you cannot reach HuggingFace from the machine running Sieve, seed the cache manually on a machine that can, then copy `~/.cache/fastembed` across. Once the model files are in place, `sieve init` will reuse them.

To switch to an Ollama-hosted embedding model instead, edit `embeddings` in your `sieve.yaml` — see the [configuration reference](configuration.md).

## Verify the installation

After `sieve init`:

```bash
sieve --version     # prints the installed version
sieve status        # shows config state and store stats (proxy will be "not running" yet)
```

Then run the proxy and smoke-test it:

```bash
sieve start         # in terminal A
sieve demo          # in terminal B — scripted six-message conversation
```

## Upgrading

```bash
pip install --upgrade llm-sieve
sieve status        # sanity-check after upgrade
```

Your store and config in `~/.sieve/` survive upgrades. The store schema is versioned and will be migrated automatically if an upgrade requires it.

## Uninstalling

```bash
pip uninstall llm-sieve
```

Your data lives under `~/.sieve/`. Nothing is removed from there by `pip uninstall` — remove it yourself if you want a clean slate:

```bash
rm -rf ~/.sieve
rm -rf ~/.cache/fastembed
```

The second command also removes the cached embedding model; you only need that if you want to reclaim the disk.

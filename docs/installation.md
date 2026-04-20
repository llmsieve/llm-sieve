# Installation

Every supported install path, platform notes, and how to wire Sieve to your LLM.

If this is your first time, skim [Getting started](getting-started.md) instead — it covers the common path in about a minute. This page is for people who need detail.

## System requirements

- **Python 3.11 or newer.** Older Pythons aren't supported.
- **Linux, macOS (Intel or Apple Silicon), or Windows via WSL2.** Native Windows is not a supported target.
- **~200 MB free disk.** ~50 MB is the embedding model; the rest is Sieve, dependencies, and the initial store.
- **An LLM endpoint.** OpenAI-compatible or Ollama. Sieve is indifferent to local vs hosted.

No GPU is needed for Sieve itself — embeddings run on CPU via FastEmbed. If you're running a local LLM as your provider, that component has its own hardware requirements.

## Install paths — which should you use?

| You want to… | Use |
|---|---|
| Install once and use the CLI day-to-day | **pipx** (recommended) |
| Install into an existing Python project's venv | **pip into a venv** |
| Run an unreleased revision or hack on Sieve | **from source** |
| Run Sieve in a container | **pipx inside the image** (see [Docker](#docker)) |

## Install from PyPI with pipx (recommended)

[pipx](https://pipx.pypa.io/) installs each Python CLI tool into its own isolated venv and exposes the entry points on your `PATH`. That avoids dependency conflicts with anything else you have installed.

```bash
# Install pipx if you don't have it
python3 -m pip install --user pipx
python3 -m pipx ensurepath
# Open a new terminal so PATH updates

# Install Sieve
pipx install llm-sieve
```

The distribution is `llm-sieve` on PyPI; it installs **two** commands: `sieve` (the CLI) and `sieve-install` (the one-shot setup flow).

### Upgrading

```bash
pipx upgrade llm-sieve
sieve --version       # confirm the new version
```

Your store and config in `~/.sieve/` survive upgrades. The store schema is versioned; the writer migrates automatically if an upgrade requires it.

### Uninstalling

```bash
pipx uninstall llm-sieve
```

This removes the CLI but leaves your `~/.sieve/` data alone. To wipe that too, see [Uninstalling](#uninstalling).

## Install from PyPI with pip + venv

Use this if you want Sieve inside a specific project's virtualenv rather than globally on your `PATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate.bat      # Windows — prefer WSL instead

pip install llm-sieve
```

You'll need to activate that venv before running `sieve` in new terminals.

## Install from source

For development or to run an unreleased revision:

```bash
git clone https://github.com/llmsieve/llm-sieve.git
cd llm-sieve
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The `[dev]` extra pulls in `pytest`, `pytest-asyncio`, `pytest-httpx`, and `ruff`. The runtime itself has no test-only dependencies.

## Platform notes

### Linux

The reference platform. `pipx install llm-sieve` is the entire install on any reasonably recent distribution.

SQLCipher ships via the `sqlcipher3` wheel and needs no system packages on supported Python versions. If your distribution ships a very old `libsqlite3`, install into a venv using a current Python rather than the system interpreter.

Autostart on boot is via systemd user units — `sieve-install` sets this up for you if you confirm. Management lives under `systemctl --user`:

```bash
systemctl --user status sieve.service
systemctl --user restart sieve.service
journalctl --user -u sieve.service -f
```

### macOS

Both Intel and Apple Silicon work.

- **Apple Silicon.** FastEmbed uses ONNX Runtime, which has native arm64 wheels — no Rosetta translation needed. If Ollama is your provider, its Metal backend will use the GPU automatically.
- **Intel.** Nothing special. Use a current Python from python.org or Homebrew.

Autostart is via a launchd `LaunchAgent` — `sieve-install` sets it up if you confirm. The plist lives at `~/Library/LaunchAgents/com.llmsieve.sieve.plist`; manage with `launchctl load/unload <plist>`.

### Windows

Use **WSL2 with a Linux distribution** (Ubuntu 22.04 or 24.04 are well-tested). Sieve is not currently packaged for native Windows — the SQLCipher and FastEmbed toolchains behave differently on Win32 and we don't ship wheels for that target.

Install WSL if you don't already have it (PowerShell as admin, then reboot):

```powershell
wsl --install
```

From inside the WSL shell, the Linux instructions above apply unchanged. Run Ollama inside WSL too so there's no crossing of the WSL ↔ host network boundary.

## Provider setups

### Local LLM with Ollama

If Ollama is already running, there's nothing extra to do — `sieve-install` auto-detects it on `127.0.0.1:11434`. Pull a model Sieve will use for both fact extraction and general inference:

```bash
ollama pull qwen3.5:9b        # tested default
# or something larger if you have the VRAM:
ollama pull qwen3:14b
```

Then run `sieve-install` and accept the defaults.

### Local LLM with vLLM or LM Studio

Both expose an OpenAI-compatible HTTP server. Point the installer at the base URL:

```bash
sieve-install --provider http://127.0.0.1:8000
```

vLLM's default port is `8000`; LM Studio's is configurable. The model name you use in your agent must match one the server actually serves.

### Hosted OpenAI-compatible endpoint

Any hosted endpoint that speaks the OpenAI API works. Examples of base URLs you might pass to `--provider`:

- **OpenAI:** `https://api.openai.com/v1`
- **Anthropic via a gateway** (LiteLLM, OpenRouter, …): the gateway's URL
- **Self-hosted gateway:** whatever URL your gateway exposes

Auth for cloud endpoints is a bearer token. Pass it with `--api-key` or set the conventional env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) — `sieve-install` picks it up automatically when it recognises the host. The key is forwarded verbatim to the upstream; Sieve doesn't intercept or require its own.

### Mixing providers

Sieve is single-provider at a time — the proxy has one configured upstream. If you need to switch between, say, a local Ollama for dev and a hosted API for prod, keep two configs and pass `-c` at start:

```bash
sieve start -c ~/.sieve/sieve-local.yaml
sieve start -c ~/.sieve/sieve-openai.yaml
```

## The FastEmbed embedding model

On first setup, Sieve downloads **BAAI/bge-small-en-v1.5** — a 384-dimensional English embedding model, ~50 MB as ONNX — and caches it under `~/.cache/fastembed`. Happens once per machine.

If you can't reach HuggingFace from the machine running Sieve, seed the cache on a machine that can and copy `~/.cache/fastembed` across. Once the files are in place, `sieve-install` reuses them and skips the download.

To switch to an Ollama-hosted embedding model instead, edit the `embeddings` block in `~/.sieve/sieve.yaml` — see [Configuration](configuration.md).

## Verify the installation

```bash
sieve --version
sieve status              # shows config + store state ("proxy not running" until start)
sieve start &             # background the proxy; ctrl-Z then `bg` also works
sieve demo                # scripted 6-message conversation against a sandboxed store
```

## Docker

There's no official Sieve image yet. If you need one, `pipx install llm-sieve` inside a base Python image is the simplest recipe:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir pipx && pipx install llm-sieve
ENV PATH="/root/.local/bin:${PATH}"
# Pre-cache FastEmbed so the first start is instant:
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"
CMD ["sieve", "start", "--host", "0.0.0.0"]
```

Mount a host directory at `/root/.sieve` to persist the store across restarts.

## Upgrading

```bash
pipx upgrade llm-sieve        # or: pip install --upgrade llm-sieve inside your venv
sieve status                  # sanity-check
```

## Uninstalling

Sieve ships two uninstall modes via the CLI:

```bash
sieve uninstall               # soft (default): preserves ~/.sieve/
sieve uninstall --hard        # requires typing DELETE, removes ~/.sieve/
```

Both print the `pipx uninstall llm-sieve` command to run afterwards — Sieve can't remove the Python package it's currently executing from.

To remove manually instead:

```bash
pipx uninstall llm-sieve      # or: pip uninstall llm-sieve
rm -rf ~/.sieve               # learned facts + config + key
rm -rf ~/.cache/fastembed     # optional — embedding model cache
```

The `fastembed` cache is only worth deleting if you're reclaiming disk. If you reinstall later, Sieve will re-download it transparently.

---
description: >
  From a blank terminal to a working Sieve proxy in about a minute —
  install, guided setup, and a live demo of transparent context
  reduction for your LLM.
---

# Getting started

A blank terminal to a working Sieve proxy in about a minute. The main path assumes you have Python, an LLM endpoint, and a terminal. If any of that isn't true yet, expand the green boxes as they come up — they cover the prerequisites.

## What Sieve does, in one paragraph

Your agent (Claude Code, Cursor, Cline, your own script — anything that talks to an LLM) sends the same system prompt, the same tool schemas, and a growing conversation history on **every single turn**. That payload gets bigger and more expensive over time. Sieve is a small proxy you run locally: your agent sends its request to Sieve, Sieve strips the bloat and hands a lean payload to the LLM, and the LLM's reply comes back to the agent unchanged. Over time Sieve learns durable facts from your conversations and can inject just the relevant ones when a turn needs them.

## Prerequisites

- **Python 3.11 or newer.** Check with `python3 --version`.
- **A terminal** (macOS Terminal, Linux shell, WSL2 on Windows — native Windows isn't supported yet).
- **An LLM endpoint** — Ollama locally, a hosted OpenAI-compatible API, or a self-hosted server like vLLM / LM Studio.
- **~200 MB free disk** (~50 MB is Sieve's embedding model, downloaded once).

??? info "New to all this? Start here."
    **What's Python?** A programming language. You need it installed because Sieve is written in it. On macOS: `brew install python` or download from [python.org](https://www.python.org/downloads/). On Ubuntu/Debian: `sudo apt install python3 python3-pip`. Verify with `python3 --version`; if it prints 3.11 or higher, you're set.

    **What's an LLM endpoint?** A server you can send chat messages to and get completions back. Sieve sits between your code and that server. The easiest path to having one locally is Ollama — expand the next box.

    **What's a terminal?** The black window with a prompt where you type commands. On macOS it's in Applications → Utilities → Terminal. On Linux it's whatever your distro ships (GNOME Terminal, Konsole, etc.). On Windows, install WSL2 first (`wsl --install` in PowerShell as administrator), then everything in this guide runs inside a Linux shell.

??? info "Don't have an LLM endpoint yet? Install Ollama."
    Ollama is the shortest path to a local LLM. It runs quietly in the background at `http://127.0.0.1:11434` — which is what Sieve will auto-detect in Step 2.

    ```bash
    # macOS / Linux — one-liner installer
    curl -fsSL https://ollama.com/install.sh | sh

    # Pull a small, capable model Sieve can use
    ollama pull qwen3.5:9b
    ```

    On Windows, download the installer from [ollama.com](https://ollama.com). If you're on WSL, install Ollama inside WSL — it'll be reachable at `127.0.0.1` from both WSL and Sieve.

    Any reasonably recent Ollama model works. Tested favourites: `qwen3.5:9b`, `qwen3:14b`, `qwen3:30b-a3b`. If you're GPU-poor, try a 4-bit quantisation of a smaller model (e.g. `qwen3.5:9b-q4_k_m`).

## Step 1 — Install Sieve

The recommended installer is [pipx](https://pipx.pypa.io/). It puts the `sieve` and `sieve-install` commands on your `PATH` without you having to manage a virtualenv.

```bash
pipx install llm-sieve
```

Verify it landed:

```bash
sieve --version
```

You should see `sieve, version 1.0.0` (or later).

??? info "Don't have pipx? Install it first."
    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```
    Then close and reopen your terminal so the new `PATH` takes effect. `pipx ensurepath` adds `~/.local/bin` to your shell's startup file so commands installed with pipx are found automatically.

??? tip "Prefer plain pip and a virtualenv?"
    ```bash
    python3 -m venv ~/.venvs/sieve
    source ~/.venvs/sieve/bin/activate
    pip install llm-sieve
    ```
    Identical result — you just need to activate that venv before running `sieve` in new terminals. pipx is really just "pip install into a hidden venv and expose the entry points."

??? warning "`sieve: command not found` after install"
    Your shell can't find the command. Two fixes in order:
    1. `pipx ensurepath` and open a new terminal.
    2. If that's not enough, add `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` or `~/.zshrc`.

## Step 2 — Run the installer

```bash
sieve-install
```

This is the only setup command you need. It:

1. **Finds your LLM.** If Ollama is running on `127.0.0.1:11434`, it's auto-selected. Otherwise you're asked where your LLM lives.
2. **Lets you pick a model.** For Ollama it shows the models you already have pulled. For cloud providers it asks for a model name and API key.
3. **Downloads the embedding model.** ~50 MB one-time, cached under `~/.cache/fastembed`.
4. **Creates the encrypted store.** SQLCipher database at `~/.sieve/memory.db` with a keyfile alongside. Both generated for you.
5. **Offers to start the proxy** (and optionally enable autostart on reboot via systemd / launchd).
6. **Prints a green "Sieve is ready" panel** with the provider URL, model, and three follow-up commands.

??? tip "Scripted / CI install — skip all the prompts"
    ```bash
    sieve-install --no-input \
      --provider http://127.0.0.1:11434 \
      --model qwen3.5:9b
    ```
    `--no-input` declines autostart by default (non-interactive systems rarely want it). Use `sieve start` afterward if you want the proxy running.

??? tip "Cloud endpoint (OpenAI, Anthropic via gateway, OpenRouter, etc.)"
    ```bash
    sieve-install \
      --provider https://api.openai.com/v1 \
      --model gpt-4o-mini \
      --api-key sk-…
    ```
    Or set the standard env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and the installer picks it up automatically when it recognises the host.

## Step 3 — Point your agent at Sieve

Sieve listens on `http://127.0.0.1:11435` — the port next to Ollama's `11434`, deliberately, so the intent is obvious. Anywhere your agent is currently pointed at your LLM, change the base URL to `http://127.0.0.1:11435` and leave everything else the same. Sieve speaks the same wire protocol as the upstream (Ollama-native or OpenAI-compatible), so the agent does not need to know Sieve is there.

=== "OpenAI-compatible (Python)"

    ```python
    client = OpenAI(
        base_url="http://127.0.0.1:11435/v1",   # was: your LLM's URL
        api_key="not-used-by-sieve",            # still forwarded to upstream
    )
    ```

=== "Ollama-native client"

    Set `OLLAMA_HOST=http://127.0.0.1:11435` (or the equivalent config in the client). Model names, request shapes, and response formats are unchanged.

=== "curl smoke-test"

    ```bash
    curl http://127.0.0.1:11435/api/chat \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "Hi, my name is Alex."}],
        "stream": false
      }'
    ```

## Step 4 — Prove it works

```bash
sieve demo
```

Runs a short scripted conversation **against a sandboxed store** (your real `~/.sieve/memory.db` is not touched). A new identity introduces themselves, shares a few facts, asks Sieve to recall them, and ends with a question about someone who was never mentioned. You should see recall hits on the seeded facts and a refusal — not a fabrication — on the trap.

For the token-reduction numbers on your own hardware:

```bash
sieve benchmark
```

15 messages through both baseline (direct to LLM) and Sieve, with a delta table of tokens, facts learned, response time, and the absence-trap verdict. Takes 5–10 min depending on model speed. Also sandboxed.

## Day-two operations — the interactive menu

Running `sieve` with no arguments (or `sieve wizard` explicitly) drops you into a menu for ongoing management:

- Start / stop / restart / autostart the proxy
- Browse what Sieve has learned (facts, entities, episodes)
- Adjust configuration without opening YAML
- Run the demo or benchmark
- Reinstall or uninstall cleanly

Options that don't apply right now (e.g. "stop" when the proxy isn't running) are marked unavailable rather than hidden — the full map is always visible.

## What to expect once it's wired up

**The first few turns feel like pass-through.** Sieve hasn't learned anything yet. Token savings and targeted recall kick in as the store fills. Watch fact count grow with `sieve status`.

**Warm operation.** Once the store has content, tool schemas and repeated instructions start being stripped, relevant facts are injected only when a turn needs them, and absence-probe questions get refused rather than fabricated.

**Per-turn diagnostics.** Every response carries Sieve headers — `X-Sieve-Rounds`, `X-Sieve-Proxy-Us`, `X-Sieve-Inbound-Tokens`, `X-Sieve-Outbound-Tokens`. Log them if you want to watch the proxy work.

## Next steps

- [Installation](installation.md) — every supported install path + platform notes (Apple Silicon, WSL, Linux distros).
- [Configuration](configuration.md) — every option in `sieve.yaml`.
- [CLI reference](cli-reference.md) — every command and flag.

## Troubleshooting

**`sieve-install` says the provider isn't reachable.** The URL can't be contacted from this machine. For Ollama: check it's running (`ollama list`). For a hosted endpoint: re-check the URL and API key. `sieve-install` is idempotent — rerun freely.

**`sieve start` says the port is already in use.** Another process owns `11435`. Either stop it or `sieve start --port 11436` (point your agent at the new port too).

**The embedding model failed to download.** Re-run `sieve-install`. If a firewall blocks HuggingFace, seed `~/.cache/fastembed` on a machine that can reach HF and copy it across.

**I want to wipe everything and start over.** `sieve uninstall --hard` (requires typing `DELETE`) removes `~/.sieve/` completely. Then re-run `sieve-install`.

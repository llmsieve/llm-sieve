# Sieve — Wizard install GIF (vendor-agnostic).
#
# Render:
#   ./branding/render-gifs.sh wizard
#
# This tape exercises the new 5-branch provider picker introduced in
# v1-rc. To keep the recording generic (no vendor endorsement) it picks
# option 3 (OpenAI-compatible endpoint) and points at a deliberately
# generic URL — the principle the GIF demonstrates is "Sieve works with
# any OpenAI-compatible endpoint you bring."
#
# VHS drives the interactive prompts directly. The render-gifs.sh
# wrapper sets up an isolated $HOME and shims for pip/pipx so the
# tape runs from scratch.
#
# Required during recording: a small stub server at
# https://demo.llmsieve.dev/v1 returning {"data": [{"id": "demo-model"}]}
# for /models. The render-gifs.sh wrapper can start this stub
# automatically with --stub.

Output branding/sieve-wizard-install.gif
Set FontSize 15
Set Width 960
Set Height 620
Set Theme "Catppuccin Mocha"
Set TypingSpeed 70ms
Set Padding 22
Set Shell bash

Sleep 1500ms

# Kick off the install. sieve-install is the "Rolls-Royce" first-run
# experience — splash + provider picker + model + ready panel.
Type "sieve-install"
Enter
Sleep 2800ms

# (If a local Ollama or env-key is auto-detected the wizard offers
# it as a one-keystroke happy path. The recording host is configured
# to not auto-detect anything, so we land on the 5-branch picker.)
Sleep 1000ms

# Pick option 3 — OpenAI-compatible endpoint. This branch demonstrates
# that Sieve handles any /v1/chat/completions provider — vLLM, OpenRouter,
# Groq, llama.cpp, LM Studio, etc.
Type "3"
Enter
Sleep 1500ms

# Endpoint URL — generic demo URL that doesn't endorse any vendor.
Type "https://demo.llmsieve.dev/v1"
Enter
Sleep 1500ms

# API key — leave blank (the demo endpoint accepts any key).
Enter
Sleep 1500ms

# Model picker — accept the default (demo-model).
Enter
Sleep 1500ms

# Writer choice — accept default (use the same model for writer).
# For non-thinking models this is the simplest and best setup.
Enter
Sleep 1500ms

# Autostart on login — accept default (yes on desktop).
Enter
Sleep 1500ms

# Start now? — yes.
Enter
Sleep 1500ms

# Final confirm — apply the plan.
Type "y"
Enter
Sleep 7000ms

# Hold on the ready panel — this is the wow moment.
Sleep 4000ms

# Show what we built. `sieve status` confirms the proxy is running.
Type "sieve status"
Enter
Sleep 3500ms

# Clean up so the GIF ends in a stable state.
Type "sieve stop"
Enter
Sleep 2000ms

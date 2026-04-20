# Sieve — Quick install GIF (one-command path).
#
# Story: user runs `pipx install llm-sieve && sieve-install`, the
# installer walks them through ~15 seconds of setup, ending on the
# green "Sieve is ready" panel. That panel is the hero frame — it
# holds for 4 seconds at the end.
#
# Render:
#   FORCE_DOCKER=1 ./branding/render-gifs.sh quick
#
# The Docker image (sieve-vhs:latest) has llm-sieve pre-installed
# AND the FastEmbed ONNX model pre-cached at /root/.cache/fastembed,
# so the embedding-model step finishes instantly in the recording
# instead of dumping HF download progress + token warnings into the
# frame.

Output branding/sieve-quick-install.gif

# ── Styling ─────────────────────────────────────────────────────
Set FontSize 15
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms
Set Padding 22
Set Shell bash

# Inherit the pre-cached FastEmbed model — without this the isolated
# $HOME the render script gives us is empty and FastEmbed has to
# re-download, defeating the Docker warmup.
Env HOME "/root"

# Title pause so the GIF doesn't start mid-sentence.
Sleep 1500ms

# ── Step 1: pipx install (shim renders the success banner) ─────
Type "pipx install llm-sieve"
Sleep 500ms
Enter
Sleep 3000ms

# ── Step 2: one-command setup ──────────────────────────────────
# --no-input + explicit --provider and --model keep the render
# deterministic; the story in the GIF is "it's one command", not
# "here's every prompt" (that's the wizard GIF).
Type "sieve-install --no-input --provider http://ollama.local:11434 --model qwen3.5:9b"
Sleep 500ms
Enter

# The installer runs: splash + welcome (~2.5s reading time), config
# write + FastEmbed preload (cached → instant) + store init +
# start (~3-5s real time since we pre-warmed FastEmbed).
Sleep 7000ms

# ── Step 3: hero frame — hold on the "Sieve is ready" panel ────
# Long hold so viewers absorb provider/model/proxy-URL/autostart/
# try-it. This is what people will screenshot.
Sleep 5000ms

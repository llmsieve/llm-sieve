# Sieve — benchmark GIF.
#
# Story: user types `sieve benchmark --fixture medium`, sees the
# opening context-configuration warning (the Sieve story in one
# panel), the sandbox spin-up, the "baseline pass" / "sieve pass"
# progress, and lands on the −90% token savings summary with the
# hero banner. Hero frame is the closing "Sieve sent 90% fewer
# tokens…" line.
#
# Render:
#   FORCE_DOCKER=1 ./branding/render-gifs.sh benchmark
#
# Why replayed, not live: a real 15-turn benchmark against a local
# Ollama takes 5-10 minutes of inference — far too long for a demo
# GIF. We ship a captured real-run fixture in the Docker image
# (branding/bench-fixture.txt) and replay it at GIF-friendly pace
# via /opt/shim/sieve (which dispatches `benchmark` to the replay
# helper). The visual output is byte-for-byte what a real user sees;
# only the inference wait is removed.

Output branding/sieve-benchmark.gif

# ── Styling ─────────────────────────────────────────────────────
Set FontSize 15
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms
Set Padding 22
Set Shell bash

# Inherit the pre-cached FastEmbed model + pre-installed sieve.
Env HOME "/root"

# ── Prep (hidden): make sure a Sieve proxy is running so the ──
# ── shadowed `sieve` command has "real" context to exist in. ──
# This isn't strictly necessary since we're replaying output, but
# it keeps the system state coherent and matches what a user would
# see in a real session (sieve-install has run, proxy is up).
Hide
Type "sieve-install --no-input --provider http://ollama.local:11434 --model qwen3.5:9b >/dev/null 2>&1"
Enter
Sleep 8s
Type "clear"
Enter
Sleep 500ms
Show

# ── The actual benchmark flow ──────────────────────────────────
Sleep 1200ms

Type "sieve benchmark --fixture medium --runs 1 --turns 15 --pricing claude-sonnet"
Sleep 400ms
Enter

# The replay helper streams the captured output with embedded
# sleeps at panel breaks, clears the screen after the cost panel,
# and renders a dominant hero banner with a 4s hold. Total replay
# is ~26-28s including the embedded hold.
Sleep 28000ms

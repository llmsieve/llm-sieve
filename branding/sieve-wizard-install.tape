# Sieve — Wizard install GIF (guided path).
#
# Render:
#   ./branding/render-gifs.sh wizard
#
# VHS drives the interactive prompts directly. The render-gifs.sh wrapper
# sets up an isolated $HOME and a pip shim so the tape runs from scratch.

Output branding/sieve-wizard-install.gif
Set FontSize 15
Set Width 960
Set Height 620
Set Theme "Catppuccin Mocha"
Set TypingSpeed 70ms
Set Padding 22
Set Shell bash

Sleep 1500ms

# Kick off the guided wizard.
Type "sieve init --wizard"
Enter
Sleep 2800ms

# Step 1 — provider type (1=Ollama). Accept default.
Enter
Sleep 1200ms

# Step 2 — provider base URL. Accept default (auto-detected Ollama).
Enter
Sleep 3500ms

# Step 3 — model picker. Pick qwen3:14b (position 24 in this machine's
# Ollama list). If you re-render on a different machine, update the
# number to whatever position your preferred model occupies — or just
# press Enter to accept the default.
Type "24"
Enter
Sleep 1200ms

# Step 4 — listen port. Accept default (11435); on this recording
# machine another dev server is already on 11435, so type a free port.
Type "11437"
Enter
Sleep 1200ms

# Step 5 — auto-generate encryption key. Accept default (Yes).
Enter
Sleep 1200ms

# Step 6 — store path. Accept default.
Enter
Sleep 1800ms

# Summary + Proceed? — confirm.
Type "y"
Enter
Sleep 7000ms

# Decline the benchmark offer for this GIF (we'll show it separately).
Type "n"
Enter
Sleep 1800ms

# The wizard wrote the port we typed (11437), and picked qwen3:14b as
# the model, so no post-wizard config tweaks are needed. Just clear the
# screen so `sieve start` and `sieve demo` below read cleanly.
Hide
Type "clear"
Enter
Sleep 400ms
Show

# Start the proxy. Redirect so request-level logs don't crowd the GIF.
Type "sieve start >/dev/null 2>&1 &"
Enter
Sleep 3500ms

# Run the demo.
Type "sieve demo"
Enter
Sleep 130000ms

# Inspect what the store learned.
Type "sieve store stats"
Enter
Sleep 3500ms

# Clean up.
Type "sieve stop"
Enter
Sleep 2000ms

# Sieve — Wizard install GIF (guided path).
#
# Render:
#   docker run --rm --network host -v "$PWD":/work -w /work \
#     ghcr.io/charmbracelet/vhs branding/sieve-wizard-install.tape
#
# Output: branding/sieve-wizard-install.gif
#
# VHS drives the interactive prompts directly — no expect, no TTY hacks.
# Needs Ollama on :11434.

Output branding/sieve-wizard-install.gif
Set FontSize 15
Set Width 960
Set Height 620
Set Theme "Catppuccin Mocha"
Set TypingSpeed 70ms
Set Padding 22

Sleep 1500ms

# Kick off the guided wizard.
Type "sieve init --wizard"
Enter
Sleep 2500ms

# Step 1 — accept detected Ollama URL.
Enter
Sleep 1200ms

# Step 2 — accept default model.
Enter
Sleep 1200ms

# Step 3 — accept default port (11435).
Enter
Sleep 1200ms

# Step 4 — accept default store location.
Enter
Sleep 1200ms

# Step 5 — accept encryption enabled.
Enter
Sleep 1200ms

# Step 6 — confirm.
Type "y"
Enter
Sleep 6000ms

# Decline the benchmark offer for this GIF (we'll show it separately).
Type "n"
Enter
Sleep 1800ms

# Start the proxy.
Type "sieve start &"
Enter
Sleep 3000ms

# Run the demo.
Type "sieve demo"
Enter
Sleep 20000ms

# Inspect what the store learned.
Type "sieve store stats"
Enter
Sleep 3500ms

# Clean up.
Type "sieve stop"
Enter
Sleep 2000ms

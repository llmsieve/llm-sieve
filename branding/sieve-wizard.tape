# Sieve — interactive wizard GIF.
#
# Story: the user types `sieve wizard`, sees the splash + menu,
# navigates to Service → Status, backs out, opens Store → Last 10
# facts, backs out, quits cleanly. Shows: the menu framework,
# state-aware options, the back/quit navigation, the "(unavailable)"
# markers. No install / no uninstall — this is the management
# surface for a user who's already set up.
#
# Render:
#   FORCE_DOCKER=1 ./branding/render-gifs.sh wizard
#
# The tape ensures an existing install via `sieve-install --no-input`
# at the top, pre-seeds a few facts so the Store → Last 10 submenu
# has something to display, then launches the wizard.

Output branding/sieve-wizard.gif

# ── Styling ─────────────────────────────────────────────────────
Set FontSize 15
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms
Set Padding 22
Set Shell bash

Env HOME "/root"

# ── Prep (hidden): install + seed a few facts directly in the ─
# ── store, then clear ──────────────────────────────────────────
# The Hide/Show pair keeps the preparation invisible in the final
# GIF. We insert facts via a direct Python call rather than through
# the live proxy so the recording doesn't wait on LLM inference.
Hide
Type "sieve-install --no-input --provider http://ollama.local:11434 --model qwen3.5:9b >/dev/null 2>&1"
Enter
Sleep 8s
Type "sieve-seed-facts"
Enter
Sleep 3s
Type "clear"
Enter
Sleep 500ms
Show

# ── The actual wizard flow ──────────────────────────────────────
Sleep 1200ms

Type "sieve wizard"
Sleep 400ms
Enter
Sleep 3000ms

# Option 2 — Service
Type "2"
Sleep 200ms
Enter
Sleep 2500ms

# Option 4 — Status detail
Type "4"
Sleep 200ms
Enter
Sleep 3500ms

# Back to service menu
Type "b"
Sleep 200ms
Enter
Sleep 1500ms

# Back to top
Type "b"
Sleep 200ms
Enter
Sleep 1500ms

# Option 3 — Store
Type "3"
Sleep 200ms
Enter
Sleep 2500ms

# Option 2 — Last 10 facts
Type "2"
Sleep 200ms
Enter
Sleep 3500ms

# Back to store menu (needs an enter for the "Press enter to return")
Enter
Sleep 1000ms

# Back to top
Type "b"
Sleep 200ms
Enter
Sleep 2000ms

# ── Hero frame — hold on the top menu showing state ────────────
Sleep 2500ms

# Quit
Type "q"
Sleep 200ms
Enter
Sleep 800ms

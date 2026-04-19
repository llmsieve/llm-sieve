# Sieve — Quick install GIF (one-command path).
#
# Render:
#   ./branding/render-gifs.sh quick
#
# The render script takes care of starting a clean proxy, warming the
# model, and pointing VHS at a shell that has `sieve` on PATH.

Output branding/sieve-quick-install.gif
Set FontSize 15
Set Width 960
Set Height 580
Set Theme "Catppuccin Mocha"
Set TypingSpeed 65ms
Set Padding 22
Set Shell bash

Sleep 1200ms

# Show the install invocation — the render wrapper has pre-printed a
# realistic "Successfully installed" banner so the GIF tells a coherent
# story without the VHS container actually re-downloading llm-sieve.
Type "pip install llm-sieve"
Enter
Sleep 2800ms

# Zero-prompt init — real.
Type "sieve init"
Enter
Sleep 4500ms

# Start the proxy in the background so the demo below has something to hit.
Type "sieve start &"
Enter
Sleep 3200ms

# The real demo — six scripted turns that show facts being learned and
# the absence-signal path working on a trap query.
Type "sieve demo"
Enter
Sleep 22000ms

# Peek at what Sieve learned.
Type "sieve status"
Enter
Sleep 3000ms

# Clean up.
Type "sieve stop"
Enter
Sleep 2000ms

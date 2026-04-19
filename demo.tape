# Sieve CLI Demo — VHS recording script
# Install VHS: go install github.com/charmbracelet/vhs@latest
# Run: vhs demo.tape
# Output: branding/sieve-demo.gif
#
# Pacing is ~1.5x slower than the initial release so the new CLI
# commands (store stats) stay readable.

Output branding/sieve-demo.gif
Set FontSize 15
Set Width 900
Set Height 560
Set Theme "Catppuccin Mocha"
Set TypingSpeed 75ms
Set Padding 20

# Title pause
Sleep 1500ms

# Install
Type "pip install llm-sieve"
Enter
Sleep 3500ms

# Init
Type "sieve init"
Enter
Sleep 4500ms

# Start (real recording: background + switch terminals)
Type "sieve start &"
Enter
Sleep 2500ms

# Demo — this is the main event
Type "sieve demo"
Enter
Sleep 20s

# New in 1.1: store inspection
Type "sieve store stats"
Enter
Sleep 3500ms

# Status check
Type "sieve status"
Enter
Sleep 3500ms

# Stop
Type "sieve stop"
Enter
Sleep 2500ms

# Sieve CLI Demo — VHS recording script
# Install VHS: go install github.com/charmbracelet/vhs@latest
# Run: vhs demo.tape
# Output: sieve-demo.gif

Output sieve-demo.gif
Set FontSize 15
Set Width 900
Set Height 520
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Padding 20

# Title pause
Sleep 1s

# Install
Type "pip install llm-sieve"
Enter
Sleep 3s

# Init
Type "sieve init"
Enter
Sleep 4s

# Start
Type "sieve start"
Enter
Sleep 2s

# Demo — this is the main event
Type "sieve demo"
Enter
Sleep 18s

# Status check
Type "sieve status"
Enter
Sleep 3s

# Stop
Type "sieve stop"
Enter
Sleep 2s

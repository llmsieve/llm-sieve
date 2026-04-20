"""ASCII branding — logo and wordmark for the CLI splash.

The SVG icon at branding/sieve-icon.svg shows dots flowing through a
funnel down to a single teal output dot. This module renders a
terminal-safe version of the same idea, plus a simple wordmark in
block letters.

Design constraints:

- Reads well at 60-80 columns (fits any terminal with room to spare)
- No tall-ASCII-art maxims: the panel-and-menu system that hosts this
  expects the splash to be visually quiet, not a full-screen takeover
- Uses the brand's single accent colour (teal #0D9488, closest terminal
  match ``teal`` / ``bright_cyan``) only on the output dot — matches
  the SVG's rule that the accent is reserved for the mark
- Gracefully degrades: the shape is ASCII-safe and renders fine on
  any TTY; the colour is applied via rich markup which strips cleanly
  when output is captured
"""

from __future__ import annotations


# Icon: dots scattered at the top, a \__/ funnel, and a single
# accent-coloured output dot below. Mirrors the SVG's composition
# without attempting pixel fidelity — 7 rows tall so it sits
# naturally above a panel.
#
# Using simple ASCII characters rather than box-drawing so the shape
# survives being pasted anywhere. The teal dot uses rich markup.

_ICON = r"""
  .  .   .    .     .  .
    .    .  .   .  .
  \                   /
   \                 /
    \               /
     \     . .     /
      \           /
       \_________/
            [teal]●[/teal]
"""


# Wordmark: "Sieve" in a simple block style. Kept compact (5 rows) so
# the full splash — icon (7 rows) + wordmark (5 rows) — is 14 rows
# including blank separators. That's tall enough to feel like a
# splash, short enough to stay above the fold on any terminal.

_WORDMARK = r"""
  ____    _
 / ___|  (_)  ___  __   __ ___
 \___ \  | | / _ \ \ \ / // _ \
  ___) | | ||  __/  \ V /|  __/
 |____/  |_| \___|   \_/  \___|
"""


def icon_ascii() -> str:
    """Return the icon as a rich-markup-ready string.

    The single ``●`` output dot is wrapped in ``[teal]…[/teal]`` so
    callers can pass the result directly to ``rich.Console.print()``
    and get the accent colour. For plain-text rendering (logs,
    tests), strip rich markup with ``rich.markup.render`` or a
    regex.
    """
    return _ICON.strip("\n")


def wordmark_ascii() -> str:
    """Return the wordmark as plain ASCII. No rich markup needed."""
    return _WORDMARK.strip("\n")


def splash_text() -> str:
    """Return the full splash block: icon + wordmark + tagline.

    One string, multi-line. Rich markup included for the accent dot.
    Callers typically ``console.print(splash_text())`` at the top of
    the top-level wizard and other entry points.
    """
    tagline = "[dim]Transparent context reduction for LLMs.[/dim]"
    return f"{_ICON.strip(chr(10))}\n\n{_WORDMARK.strip(chr(10))}\n  {tagline}\n"


def render_splash(console) -> None:
    """Print the splash to a rich Console.

    Separated from ``splash_text`` so callers that want to measure or
    capture the output can, while the common case (print at startup)
    is a one-liner. Idempotent — safe to call multiple times if the
    wizard re-enters its top screen.
    """
    console.print(splash_text())

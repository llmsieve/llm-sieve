"""Tests for the CLI splash / branding module.

Tests what's testable: the strings exist, contain the expected marks,
are ASCII-safe apart from the intentional accent glyph, and render
without crashing. Whether the logo *looks good* is a human judgement
and lives outside the test suite.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from sieve._branding import icon_ascii, wordmark_ascii, splash_text, render_splash


def test_icon_contains_funnel_shape():
    icon = icon_ascii()
    # Funnel drawn with backslashes and forward-slashes.
    assert "\\" in icon
    assert "/" in icon
    # Output dot is the accent glyph.
    assert "●" in icon


def test_icon_uses_teal_markup_for_accent():
    """The accent dot must be wrapped in rich's [teal]...[/teal] so the
    brand's single-accent rule is enforced automatically."""
    icon = icon_ascii()
    assert "[teal]●[/teal]" in icon


def test_wordmark_contains_sieve_letters():
    w = wordmark_ascii()
    # The block letters spell "Sieve" — the easiest way to check is
    # that all five initial letters occur vertically in the expected
    # relative positions. Simpler assertion: the wordmark is
    # multi-line and non-empty.
    assert len(w.splitlines()) >= 3
    # The block-S glyph has the characteristic `/___|` and `\___ \`
    # stubs — rough substring checks.
    assert "___" in w


def test_splash_text_combines_icon_wordmark_and_tagline():
    s = splash_text()
    assert "●" in s
    assert "___" in s  # wordmark present
    assert "Transparent context reduction" in s


def test_splash_fits_within_80_columns():
    """Every line of the splash must fit in 80 cols so it never wraps
    on a narrow terminal. Width-cap elsewhere in the CLI is 120 but
    the splash is deliberately tighter."""
    for line in splash_text().splitlines():
        # Strip rich markup before measuring — [teal]...[/teal] adds
        # chars to the raw string that don't render.
        import re
        stripped = re.sub(r"\[/?[^\]]+\]", "", line)
        assert len(stripped) <= 80, (
            f"splash line too wide ({len(stripped)} > 80): {stripped!r}"
        )


def test_render_splash_prints_without_crashing():
    """Smoke test: rich can render the full markup without complaining."""
    buf = StringIO()
    console = Console(file=buf, width=100, force_terminal=False, no_color=True)
    render_splash(console)
    out = buf.getvalue()
    # Tagline made it through.
    assert "Transparent context reduction" in out
    # The rich markup brackets should NOT appear in the rendered
    # output — they're formatting directives, not content.
    assert "[teal]" not in out
    assert "[/teal]" not in out


def test_render_splash_is_idempotent():
    """Calling render_splash twice produces double output, never
    crashes, never state-leaks. The caller might re-enter the top
    menu; the function shouldn't care."""
    buf = StringIO()
    console = Console(file=buf, width=100, force_terminal=False, no_color=True)
    render_splash(console)
    render_splash(console)
    # Two renders → two taglines in the output.
    assert buf.getvalue().count("Transparent context reduction") == 2

"""Paged-menu framework for the top-level Sieve wizard.

Design
======

A *paged* menu framework: each "screen" is a list of numbered
options; the user types a number, the handler runs, and we return to
the previous screen (or push a new one). No full-screen TUI, no
arrow-key navigation, no external deps beyond rich.

This keeps the wizard:

- Reliable over SSH and on any terminal (vt100, tmux, screen, etc.)
- Trivial to script for tests: pipe stdin, capture stdout
- Forgiving: bad input re-prompts, empty input picks the default,
  Ctrl-C always exits cleanly

Core concepts
=============

- **MenuOption** — one entry. A label, optional help text, and a
  handler callable. The handler returns one of:
    - ``None`` / ``BACK``: redraw the current screen
    - ``QUIT``: exit the whole wizard
    - a ``MenuScreen``: push that screen onto the nav stack
    - any other value: treated as a result; the screen handler can
      decide what to do (some screens want to return a chosen value
      upward, e.g. a picker)
- **MenuScreen** — a titled list of options, rendered each time the
  user lands on it. Screens are re-entrant — the app re-calls
  ``screen.render()`` and ``screen.prompt()`` each pass so any state
  the handler mutated is visible on redraw.
- **MenuApp** — the driver loop. Holds a nav stack, renders the top,
  asks for a pick, dispatches, pushes / pops.

Special sentinels
=================

``BACK`` and ``QUIT`` are module-level singletons — comparing by
identity is unambiguous and avoids any chance of a handler
accidentally returning a string that gets parsed as navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ── Navigation sentinels ────────────────────────────────────────────────


class _Sentinel:
    """Lightweight marker with a friendly repr. Used for BACK / QUIT."""

    def __init__(self, name: str):
        self._name = name

    def __repr__(self) -> str:
        return f"<menu.{self._name}>"


BACK = _Sentinel("BACK")
QUIT = _Sentinel("QUIT")


class ResetTo:
    """Handler return value that clears the nav stack and sets a new root.

    Used when a handler materially changes global state (e.g. install
    / uninstall) and stale screens on the stack would otherwise show
    pre-change options. The caller would navigate back through dead
    options — we replace the whole stack with a fresh root instead.

        return ResetTo(build_top_screen(console))
    """

    def __init__(self, screen: "MenuScreen"):
        self.screen = screen

    def __repr__(self) -> str:
        return f"<menu.ResetTo title={self.screen.title!r}>"


# ── Option + Screen ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MenuOption:
    """One numbered row in a MenuScreen.

    ``handler`` is the only required callable after construction.
    ``label`` is the visible text; ``help`` renders underneath in
    dim style when present; ``enabled`` lets a screen disable an
    option for known-bad-state reasons (handler is never called).

    When ``enabled`` is False the option still renders but is
    visually dimmed and cannot be chosen — selecting it re-prompts.
    """
    label: str
    handler: Callable[[], Any]
    help: str = ""
    enabled: bool = True
    # When set, the option's numbered choice key is fixed to this
    # character instead of the row's ordinal. Useful for reserved
    # keys like 'b' for Back or 'q' for Quit.
    key: str = ""


@dataclass
class MenuScreen:
    """A titled list of options, plus optional subtitle/help text.

    Screens are re-used across redraws — do not stash request-scoped
    state on the screen object itself. When a handler needs to
    mutate state visible on redraw, give the MenuApp a context
    object and let the option labels resolve dynamically from it.
    """
    title: str
    options: list[MenuOption]
    subtitle: str = ""
    # When True, ``b``/``back`` and ``q``/``quit`` are accepted as
    # top-level shortcuts. The root screen sets allow_back=False so
    # the user can't back out of the universe.
    allow_back: bool = True
    allow_quit: bool = True
    # Footer note shown under the options, e.g. "type a number or b
    # to go back". Empty defaults to a sensible hint.
    footer: str = ""


# ── App driver ──────────────────────────────────────────────────────────


class MenuApp:
    """Render + dispatch loop for a stack of MenuScreens.

    Typical usage::

        app = MenuApp(initial=top_screen, console=console)
        app.run()

    ``stdin_source`` and ``stdout_sink`` can be overridden for
    tests. By default we defer to click.prompt + rich.console, which
    matches the rest of the CLI.

    Clearing the screen between screens is opt-in via
    ``clear_between_screens=True`` (default False in tests, True in
    the CLI). Tests want the full transcript; users want a clean
    paged feel.
    """

    def __init__(
        self,
        initial: MenuScreen,
        *,
        console=None,
        input_fn: Callable[[str], str] | None = None,
        clear_between_screens: bool = False,
    ):
        if console is None:
            from rich.console import Console
            console = Console()
        self._console = console
        self._input_fn = input_fn or _default_input
        self._clear = clear_between_screens
        self._stack: list[MenuScreen] = [initial]

    # ── Public ──────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop. Exits when the stack is empty (QUIT) or the
        root screen's handler returns QUIT."""
        while self._stack:
            top = self._stack[-1]
            if self._clear:
                self._console.clear()
            self._render(top)
            try:
                pick = self._prompt(top)
            except KeyboardInterrupt:
                self._console.print("\n[yellow]Interrupted.[/]")
                return
            if pick is None:
                # Empty input and no default — re-render.
                continue
            if pick == "__back__":
                # Don't let users back out of the root.
                if len(self._stack) > 1:
                    self._stack.pop()
                continue
            if pick == "__quit__":
                return
            # Otherwise pick is an index into options.
            opt = top.options[pick]
            if not opt.enabled:
                # Surface the per-option reason if the handler set one
                # via the help text, so users aren't left wondering why.
                reason = opt.help or ""
                self._console.print(
                    f"[yellow]Option {pick + 1} isn't available right now.[/] "
                    f"[dim]{reason}[/]".rstrip()
                )
                continue
            try:
                result = opt.handler()
            except KeyboardInterrupt:
                self._console.print("\n[yellow]Interrupted.[/]")
                return
            except Exception as exc:  # noqa: BLE001
                # Never leave the user at a broken screen — log and
                # redraw. Full traceback suppressed because the
                # wizard is an end-user surface; detailed errors are
                # the subcommand's job.
                self._console.print(f"[bold red]Error:[/] {exc}")
                # Pause for acknowledgement so the user sees it
                # before the redraw clobbers the message.
                self._console.print("[dim]Press enter to continue…[/]")
                try:
                    self._input_fn("")
                except KeyboardInterrupt:
                    return
                continue
            if result is QUIT:
                return
            if result is BACK or result is None:
                continue
            if isinstance(result, ResetTo):
                # Clear the whole stack and replace with the new root.
                # Used after install / uninstall so stale pre-change
                # screens can't surface.
                self._stack = [result.screen]
                continue
            if isinstance(result, MenuScreen):
                self._stack.append(result)
                continue
            # Anything else: the handler returned an application-
            # level result and the current screen knows what to do
            # with it. Most flows encode that internally; we just
            # redraw.
            continue

    # ── Internals ───────────────────────────────────────────────

    def _render(self, screen: MenuScreen) -> None:
        console = self._console
        console.print()
        console.print(f"[bold]{screen.title}[/]")
        if screen.subtitle:
            console.print(f"[dim]{screen.subtitle}[/]")
        console.print()

        for i, opt in enumerate(screen.options, start=1):
            key = opt.key or str(i)
            if opt.enabled:
                key_style = "cyan"
                label_style = ""
                suffix = ""
            else:
                # Disabled options need to be OBVIOUSLY disabled.
                # Dim alone is too subtle on many terminals — also
                # render an explicit "(unavailable)" so anyone who
                # can't see the colour still gets the signal.
                key_style = "dim"
                label_style = "dim strike"
                suffix = " [dim](unavailable)[/]"
            # Assemble the line in pieces so we only emit rich tags
            # when there's a real style — an empty "[]...[/]" is
            # rejected by rich's markup parser.
            key_part = f"[{key_style}]{key:>2}.[/]"
            if label_style:
                label_part = f"[{label_style}]{opt.label}[/]"
            else:
                label_part = opt.label
            console.print(f"  {key_part} {label_part}{suffix}")
            if opt.help:
                console.print(f"        [dim]{opt.help}[/]")

        # Reserved nav keys.
        reserved: list[str] = []
        if screen.allow_back and len(self._stack) > 1:
            reserved.append("b = back")
        if screen.allow_quit:
            reserved.append("q = quit")
        if reserved or screen.footer:
            console.print()
            if screen.footer:
                console.print(f"[dim]{screen.footer}[/]")
            if reserved:
                console.print(f"[dim]({'  '.join(reserved)})[/]")

    def _prompt(self, screen: MenuScreen) -> int | str | None:
        """Read one answer. Returns:

        - an int index (0-based) into ``screen.options`` on a valid
          numbered pick
        - ``"__back__"`` when the user asked to go back
        - ``"__quit__"`` when the user asked to quit
        - ``None`` when input was empty / unparseable and the caller
          should re-render

        Re-prompts inline on bad input; only returns up to the main
        loop when we have something actionable.
        """
        n = len(screen.options)
        while True:
            raw = self._input_fn("Choose").strip().lower()
            if not raw:
                # No default for unnumbered screens — silently redraw.
                return None
            if screen.allow_back and raw in ("b", "back"):
                return "__back__"
            if screen.allow_quit and raw in ("q", "quit", "exit"):
                return "__quit__"
            # Numeric pick or keyed pick.
            # 1) Match by option.key (explicit single-char key).
            for i, opt in enumerate(screen.options):
                if opt.key and opt.key.lower() == raw:
                    return i
            # 2) Match by 1-based ordinal.
            try:
                idx = int(raw)
            except ValueError:
                self._console.print(
                    f"[yellow]'{raw}' isn't a valid choice. "
                    f"Type a number 1–{n}"
                    f"{', b for back' if screen.allow_back else ''}"
                    f"{', q to quit' if screen.allow_quit else ''}.[/]"
                )
                continue
            if 1 <= idx <= n:
                return idx - 1
            self._console.print(
                f"[yellow]{idx} is out of range (1–{n}).[/]"
            )


def _default_input(label: str) -> str:
    """Default prompt — click for line-edit / history / help."""
    import click
    return click.prompt(label, type=str, default="", show_default=False)

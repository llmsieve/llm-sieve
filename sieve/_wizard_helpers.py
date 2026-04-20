"""Shared helpers for numbered-prompt wizards.

Both the benchmark wizard and the top-level `sieve` wizard present
numbered option lists and live-queried model pickers. Keeping them
in one module ensures consistent UX across the surface area.

Design principles (derived from the benchmark-output research):

- Numbered options over free-text prompts whenever the set is
  bounded. ``1/2/3`` + enter is fast, unambiguous, and works on any
  SSH session regardless of terminal capabilities.
- Never leave the user hanging. Every call either returns a value,
  exits cleanly, or re-prompts with an explicit re-statement.
- Live model listing uses the configured provider. We support Ollama
  (``/api/tags``) and OpenAI-style (``/v1/models``). Falls back to a
  free-text prompt on network failure — never silently hangs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import httpx


def model_context_window(base_url: str, model: str, timeout: float = 4.0) -> int | None:
    """Query Ollama's ``/api/show`` for the model's effective ``num_ctx``.

    Returns the integer num_ctx when we can determine it, otherwise
    ``None``. Silent fall-through on non-Ollama endpoints (OpenAI /
    LM Studio / etc. don't expose this; a 404 means "can't tell").

    Ollama payload structure (v0.1.x through v0.4.x):

        {
          "modelfile": "...",              # human-readable
          "parameters": "num_ctx 8192\\n...",
          "model_info": {
            "llama.context_length": 131072,  # the *architectural* max
            ...
          }
        }

    The user-effective context is ``parameters.num_ctx`` if set,
    otherwise the model's training default (``llama.context_length``
    or similar). Ollama defaults to 2048 or 4096 when neither is
    specified in a Modelfile — we return the explicit setting when
    available, which is what actually gets enforced at inference time.
    """
    base = base_url.rstrip("/")
    try:
        r = httpx.post(
            f"{base}/api/show",
            json={"name": model},
            timeout=timeout,
        )
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json() or {}
    except Exception:
        return None

    # 1. Look in the parsed parameters string first — this is the
    # Modelfile-configured value that Ollama actually enforces.
    params_text = data.get("parameters") or ""
    if isinstance(params_text, str):
        import re as _re
        m = _re.search(r"num_ctx\s+(\d+)", params_text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass

    # 2. Fall back to the model's architectural context length. This
    # is an upper bound — the *actual* runtime context is what the
    # Modelfile specifies, or Ollama's default (2048/4096) if absent.
    model_info = data.get("model_info") or {}
    for key, val in model_info.items():
        if isinstance(key, str) and key.endswith(".context_length"):
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def list_models(base_url: str, timeout: float = 4.0) -> list[str]:
    """Return the model names exposed by an LLM endpoint.

    Tries Ollama's ``/api/tags`` first (common case for local users),
    then OpenAI's ``/v1/models`` as a fallback. Returns an empty list
    on any failure — the caller is expected to fall back to a
    free-text prompt so the user can enter a model name manually.

    Model names are de-duplicated and sorted case-insensitively for a
    stable menu ordering.
    """
    base = base_url.rstrip("/")
    names: list[str] = []
    try:
        r = httpx.get(f"{base}/api/tags", timeout=timeout)
        if r.status_code == 200:
            data = r.json() or {}
            names = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        names = []
    if not names:
        try:
            r = httpx.get(f"{base}/v1/models", timeout=timeout)
            if r.status_code == 200:
                data = r.json() or {}
                items = data.get("data", data) or []
                names = [m.get("id", "") for m in items if isinstance(m, dict) and m.get("id")]
        except Exception:
            pass
    seen: set[str] = set()
    out: list[str] = []
    for n in sorted(names, key=lambda s: s.lower()):
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


@dataclass(frozen=True)
class NumberedChoice:
    """One option in a numbered picker."""
    label: str
    value: str
    help: str = ""


def pick_numbered(
    prompt: str,
    choices: Sequence[NumberedChoice],
    *,
    default: str | None = None,
    console=None,
    allow_free_text: bool = False,
    free_text_prompt: str = "Enter custom value",
) -> str:
    """Render a numbered picker and return the chosen value.

    ``default`` is the value that will be used when the user just hits
    enter. It does NOT need to match one of the choices' values — if
    it doesn't, it's silently appended as "(custom)".

    When ``allow_free_text`` is True, an extra numbered option is added
    at the end: "N. Other (type a value)". Selecting it re-prompts with
    ``free_text_prompt`` and accepts any string.

    Input loop never exits silently: on unparseable input it re-prompts
    with the available range. On empty input, returns ``default`` if
    provided; otherwise re-prompts.
    """
    if console is None:
        from rich.console import Console
        console = Console()

    items = list(choices)
    # If default doesn't match a choice value and we're not allowing
    # free text, append it silently so the user can still press enter.
    has_default_in_list = any(c.value == default for c in items)
    if default is not None and not has_default_in_list and not allow_free_text:
        items.append(NumberedChoice(label=f"{default} (default)", value=default))
        has_default_in_list = True

    free_text_index: int | None = None
    if allow_free_text:
        free_text_index = len(items) + 1

    # Render
    console.print()
    console.print(f"[bold]{prompt}[/]")
    for i, c in enumerate(items, start=1):
        marker = "  "
        if default is not None and c.value == default:
            marker = "[dim]*[/] "
        console.print(f"  {marker}[cyan]{i:>2}.[/] {c.label}")
        if c.help:
            console.print(f"        [dim]{c.help}[/]")
    if free_text_index is not None:
        console.print(f"  [cyan]{free_text_index:>2}.[/] Other — type a value")
    if default is not None:
        console.print(f"[dim]  (enter = default: {default})[/]")

    import click
    while True:
        raw = click.prompt(
            "Choose",
            type=str,
            default="",
            show_default=False,
        ).strip()
        if not raw:
            if default is not None:
                return default
            console.print("[yellow]Enter a number.[/]")
            continue
        try:
            idx = int(raw)
        except ValueError:
            console.print(f"[yellow]'{raw}' isn't a number; try 1–{len(items) + (1 if free_text_index else 0)}.[/]")
            continue
        if free_text_index is not None and idx == free_text_index:
            value = click.prompt(free_text_prompt, type=str, default="").strip()
            if not value:
                console.print("[yellow]Empty value — try again.[/]")
                continue
            return value
        if 1 <= idx <= len(items):
            return items[idx - 1].value
        console.print(
            f"[yellow]{idx} is out of range; try 1–"
            f"{len(items) + (1 if free_text_index else 0)}.[/]"
        )


def pick_model(
    prompt: str,
    *,
    base_url: str,
    default: str | None = None,
    console=None,
    exclude: list[str] | None = None,
) -> str:
    """Pick a model from the live endpoint, with a free-text escape.

    On successful listing: numbered picker with one entry per model.
    On empty listing: single free-text prompt with the default
    pre-filled. Either way, the user always gets a value back; no
    silent hang, no crash on a down endpoint.

    ``exclude`` removes the named models from the picker — used by the
    grader picker to keep the "pick a different model" suggestion
    meaningful (a user who's got qwen3.5:9b as test model shouldn't
    see qwen3.5:9b as the top grader option).
    """
    if console is None:
        from rich.console import Console
        console = Console()

    names = list_models(base_url)
    if exclude:
        ex = set(exclude)
        names = [n for n in names if n not in ex]

    if not names:
        import click
        console.print(
            f"[yellow]Could not list models from {base_url}. "
            "Enter a model name manually.[/]"
        )
        val = click.prompt(
            prompt,
            default=default if default else "",
            show_default=bool(default),
        ).strip()
        return val or (default or "")

    choices = [NumberedChoice(label=n, value=n) for n in names]
    return pick_numbered(
        prompt,
        choices,
        default=default,
        console=console,
        allow_free_text=True,
        free_text_prompt="Enter a model name (not in the list above)",
    )

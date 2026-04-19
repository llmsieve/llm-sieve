# Sieve branding assets

All the visual assets that ship with the open-source distribution — logos, social cards, and the CLI demo GIF.

## Logos

| File | Use |
|---|---|
| `sieve-icon.svg` | Square mark for app icons, favicons, OS tiles |
| `sieve-icon-favicon.svg` | Simplified favicon variant |
| `sieve-wordmark.svg` | Text-only mark |
| `sieve-lockup.svg` | Full horizontal lockup (icon + wordmark) |
| `sieve-lockup-dark.svg` / `sieve-lockup-light.svg` | Theme-specific lockups used by the README |

## CLI demo GIFs

The README shows animated demos of the two install paths. Both are rendered from VHS tapes, so anyone can regenerate them from source.

| GIF | Tape | Shows |
|---|---|---|
| `sieve-demo.gif` | `../demo.tape` | Initial release: install → init → demo → status |
| `sieve-quick-install.gif` | `sieve-quick-install.tape` | Zero-prompt path: `pip install`, `sieve init`, `sieve demo`, `sieve status` |
| `sieve-wizard-install.gif` | `sieve-wizard-install.tape` | Guided setup: `sieve init --wizard`, the six-step wizard, `sieve demo`, `sieve store stats` |

### Rendering the GIFs

Requirements:

- VHS installed locally (`go install github.com/charmbracelet/vhs@latest`), or
- Docker with `ghcr.io/charmbracelet/vhs:latest` + a Python 3.13 virtualenv with sieve installed
- Ollama running at `http://localhost:11434` with the model from `sieve.yaml` loaded

One-command regeneration once VHS is available:

```bash
./branding/render-gifs.sh quick      # → sieve-quick-install.gif
./branding/render-gifs.sh wizard     # → sieve-wizard-install.gif
./branding/render-gifs.sh all        # → both
```

The render script isolates each recording under a fresh `$HOME` so it never touches your real `~/.sieve` state, and uses a small `pip` shim so the `pip install llm-sieve` line in the tapes renders as a realistic success banner without hitting PyPI.

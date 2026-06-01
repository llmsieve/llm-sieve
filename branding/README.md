# Sieve branding assets

Everything that ships with the open-source distribution — logos, social cards, CLI demo GIFs, and the VHS tapes + render pipeline that produce them.

## Logos

| File | Use |
|---|---|
| `sieve-icon.svg` | Square mark for app icons, favicons, OS tiles |
| `sieve-icon-favicon.svg` | Simplified favicon variant |
| `sieve-wordmark.svg` | Text-only mark |
| `sieve-lockup.svg` | Full horizontal lockup (icon + wordmark) |
| `sieve-lockup-dark.svg` / `sieve-lockup-light.svg` | Theme-specific lockups used by the README |

## Social card

| File | Use |
|---|---|
| `social-card-clean.svg` | Editable source |
| `social-card-clean-1200x630.png` | OpenGraph / Twitter card (1200×630) |

## CLI demo GIFs

Three tapes, three stories. All 1200×700, Catppuccin Mocha, rendered from VHS tapes so anyone can reproduce them.

| GIF | Tape | Story |
|---|---|---|
| `sieve-quick-install.gif` | `sieve-quick-install.tape` | One-command install. `pipx install llm-sieve && sieve-install`, ending on the green "Sieve is ready" panel. |
| `sieve-wizard.gif` | `sieve-wizard.tape` | Interactive management menu. Navigates Service → Status, Store → Last 10 facts, back to the top menu showing state. |
| `sieve-benchmark.gif` | `sieve-benchmark.tape` | Real benchmark run replay: 15-turn medium fixture → 90% token reduction, ending on the hero "Sieve sent 90% fewer tokens" banner. |

## Rendering the GIFs

The supported render path is Docker — native VHS crashes on some Debian kernels due to chromium zygote sandboxing. The `branding/Dockerfile.vhs` image ships `llm-sieve` pre-installed plus a cached FastEmbed ONNX model so recordings don't contain HuggingFace download clutter.

### First-time build

```bash
# From the repo root — builds sieve-vhs:latest
docker build -f branding/Dockerfile.vhs -t sieve-vhs:latest .
```

Rebuild this image whenever the wheel changes (after a `pip` release or local edit).

### Render

```bash
FORCE_DOCKER=1 ./branding/render-gifs.sh quick        # → sieve-quick-install.gif
FORCE_DOCKER=1 ./branding/render-gifs.sh wizard       # → sieve-wizard.gif
FORCE_DOCKER=1 ./branding/render-gifs.sh benchmark    # → sieve-benchmark.gif
FORCE_DOCKER=1 ./branding/render-gifs.sh all          # → all three
```

Per-tape speed defaults:

- quick / wizard — ffmpeg speeds the output to 2.5× so the tape can record real work without the final GIF feeling laggy.
- benchmark — 1.2× only, because the replay helper (see below) already paces itself.

Override with `SPEED=1.5 ./branding/render-gifs.sh …`.

### How the benchmark GIF works

A full `sieve benchmark` run takes 5–10 minutes of real LLM inference — far too long for a demo GIF. Instead:

1. `branding/bench-fixture.txt` contains the captured stdout from a real 90%-reduction run.
2. `/opt/shim/sieve-bench-demo` (baked into the Docker image) replays that fixture with embedded read-pauses at panel boundaries.
3. `/opt/shim/sieve` dispatches the `benchmark` subcommand to the replay helper; everything else (`sieve wizard`, `sieve status`, …) falls through to the real CLI.

The tape types `sieve benchmark --fixture medium --runs 1 --turns 15 --pricing claude-sonnet` — byte-for-byte what a real user would type, byte-for-byte the output they'd see. Only the inference wait is removed.

### Hostname sanitisation

The tapes reference `http://ollama.local:11434` instead of a real LAN IP. The render script passes `--add-host "ollama.local:${OLLAMA_HOST_IP:-192.168.1.149}"` to docker so the container can still reach your LAN Ollama. Override per-host:

```bash
OLLAMA_HOST_IP=10.0.0.42 FORCE_DOCKER=1 ./branding/render-gifs.sh all
```

This keeps the rendered GIFs free of anyone's home LAN IP while letting the render actually hit a real server during the (hidden) preparation steps.

## Vendor-agnostic recording

`sieve-quick-install.tape` records the 5-branch provider picker (Anthropic / OpenAI / OpenAI-compatible / Ollama / Custom). To keep the GIF vendor-neutral, the tape picks option 3 (OpenAI-compatible) and points the wizard at a generic stub URL.

For the stub URL to actually respond during recording, start `branding/stub_provider.py` alongside the renderer:

```bash
# Terminal 1: launch the stub on 127.0.0.1:8765
python branding/stub_provider.py &

# Terminal 2: render the tape
./branding/render-gifs.sh quick
```

The stub serves both OpenAI-compatible (`/v1/models`, `/v1/chat/completions`) and Ollama-compatible (`/api/tags`, `/api/chat`) endpoints with canned responses. It's purpose-built for recording — no production use. See `branding/stub_provider.py` for the surface.

When recording, point sieve-install at `http://127.0.0.1:8765/v1` (option 3) — the rendered GIF shows a generic URL, not any specific cloud provider. Sieve works with any OpenAI-compatible endpoint you bring.

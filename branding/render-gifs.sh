#!/usr/bin/env bash
#
# Render the Sieve CLI demo GIFs from their VHS tapes.
#
# Usage:
#   branding/render-gifs.sh quick      # renders sieve-quick-install.gif
#   branding/render-gifs.sh wizard     # renders sieve-wizard-install.gif
#   branding/render-gifs.sh all        # renders both
#
# Requirements (one of):
#   - Native VHS (go install github.com/charmbracelet/vhs@latest) + ttyd + ffmpeg, OR
#   - Docker with ghcr.io/charmbracelet/vhs pulled
#   - A locally-installed `sieve` (from `uv pip install -e .` in the repo)
#   - A running Ollama on :11434 with the model in sieve.yaml loaded
#
# The script runs `sieve stop` before each recording so the tape starts
# from a clean slate, uses a fake `pip` shim so the `pip install` line
# in the tapes renders as a realistic success banner without hitting
# PyPI, and isolates each recording under a fresh $HOME so your real
# ~/.sieve state is never touched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$REPO_ROOT/.venv/bin/sieve" ]]; then
  echo "error: expected a venv at .venv/ with sieve installed (try: uv venv + uv pip install -e .)" >&2
  exit 1
fi

# Prefer native VHS if it's on PATH. Fall back to the Docker image.
# Allow FORCE_DOCKER=1 to override when the native vhs binary crashes
# on the host's chromium sandboxing (common on Debian-family boxes).
VHS_BIN=""
if [[ -z "${FORCE_DOCKER:-}" ]]; then
  if command -v vhs >/dev/null 2>&1; then
    VHS_BIN="vhs"
  elif [[ -x "$HOME/go/bin/vhs" ]]; then
    VHS_BIN="$HOME/go/bin/vhs"
  fi
fi

# Build a throwaway shim directory that lives only for this run.
SHIM_DIR="$(mktemp -d)"
trap 'rm -rf "$SHIM_DIR"' EXIT

# Believable-looking fake pip — VHS types `pip install llm-sieve` and
# we want a realistic success banner, not a 404 or a command-not-found.
cat > "$SHIM_DIR/pip" <<'SHIM'
#!/usr/bin/env bash
if [[ "$1" == "install" ]]; then
  shift
  echo "Collecting $*"
  echo "  Downloading llm_sieve-1.0.0-py3-none-any.whl (142 kB)"
  echo "Installing collected packages: llm-sieve"
  echo "Successfully installed $*"
else
  echo "pip (shim) — passthrough disabled in demo mode"
fi
SHIM
chmod +x "$SHIM_DIR/pip"

# Same for pipx — the recommended install path in our docs.
cat > "$SHIM_DIR/pipx" <<'SHIM'
#!/usr/bin/env bash
if [[ "$1" == "install" ]]; then
  shift
  echo "  installed package $* 1.0.0, installed using Python 3.12.3"
  echo "  These apps are now globally available"
  echo "    - sieve"
  echo "    - sieve-install"
  echo "done! ✨ 🎉 ✨"
else
  echo "pipx (shim) — passthrough disabled in demo mode"
fi
SHIM
chmod +x "$SHIM_DIR/pipx"

# Ensure no proxy is running from a prior session.
"$REPO_ROOT/.venv/bin/sieve" stop >/dev/null 2>&1 || true

# SPEED is the playback multiplier applied via ffmpeg after VHS is done.
# e.g. SPEED=2.5 means 40-second raw recording becomes a 16-second GIF.
# Defaults differ by tape (see per-case overrides below):
#   quick / wizard — 2.5× (recorded near real-time, want snap)
#   benchmark      — 1.2× (replay already paces itself)
# A user-supplied SPEED= in the environment wins over the defaults.
USER_SPEED="${SPEED:-}"

# Guess the output GIF path from the "Output" directive in the tape.
gif_from_tape() {
  local tape="$1"
  awk '/^Output / {print $2; exit}' "$tape"
}

render_one() {
  local tape="$1"
  local fake_home
  fake_home="$(mktemp -d)"
  echo "→ rendering $tape  (isolated HOME: $fake_home)"
  if [[ -n "$VHS_BIN" ]]; then
    HOME="$fake_home" \
      PATH="$SHIM_DIR:$REPO_ROOT/.venv/bin:$PATH" \
      "$VHS_BIN" "$tape"
  else
    # Prefer our custom image that has llm-sieve pre-installed into
    # /opt/sieve AND the FastEmbed ONNX model pre-cached at
    # /root/.cache/fastembed. That side-steps (a) the .venv shebang
    # issue (host venv's python path doesn't resolve in-container),
    # and (b) the 10-15s of HF-download clutter in the recording.
    local image="sieve-vhs:latest"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
      echo "! $image not built yet; falling back to upstream VHS." \
           " Build with: docker build -f branding/Dockerfile.vhs -t sieve-vhs:latest ." >&2
      image="ghcr.io/charmbracelet/vhs:latest"
    fi
    # We deliberately don't override HOME — the image's /root has
    # the FastEmbed cache, and the container is disposable so any
    # state written under /root disappears when it exits.
    #
    # OLLAMA_HOST_IP lets the tapes reference a neutral hostname
    # (ollama.local) in the rendered GIFs instead of leaking the
    # recorder's LAN IP. Override per-host with OLLAMA_HOST_IP=... .
    local ollama_ip="${OLLAMA_HOST_IP:-192.168.1.149}"
    docker run --rm \
      --network host \
      --add-host "ollama.local:${ollama_ip}" \
      -v "$REPO_ROOT":/work \
      -w /work \
      "$image" "$tape"
  fi
  rm -rf "$fake_home"
  # Kill any proxy that the tape may have left running in the background.
  "$REPO_ROOT/.venv/bin/sieve" stop >/dev/null 2>&1 || true

  # Speed-adjust the rendered GIF. SPEED=1 means "leave alone".
  local gif
  gif="$(gif_from_tape "$tape")"
  if [[ -n "$gif" && -f "$gif" ]] && awk "BEGIN{exit !($SPEED > 1)}"; then
    echo "→ speeding up $gif by ${SPEED}× via ffmpeg"
    local tmp="${gif%.gif}.raw.gif"
    mv "$gif" "$tmp"
    ffmpeg -y -i "$tmp" \
      -vf "setpts=PTS/${SPEED},split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
      -loop 0 "$gif" </dev/null >/dev/null 2>&1
    rm -f "$tmp"
  fi
}

DEFAULT_SPEED=2.5
BENCH_SPEED=1.2

case "${1:-all}" in
  quick)
    SPEED="${USER_SPEED:-$DEFAULT_SPEED}" render_one branding/sieve-quick-install.tape
    ;;
  wizard)
    SPEED="${USER_SPEED:-$DEFAULT_SPEED}" render_one branding/sieve-wizard.tape
    ;;
  benchmark)
    SPEED="${USER_SPEED:-$BENCH_SPEED}" render_one branding/sieve-benchmark.tape
    ;;
  all)
    SPEED="${USER_SPEED:-$DEFAULT_SPEED}" render_one branding/sieve-quick-install.tape
    SPEED="${USER_SPEED:-$DEFAULT_SPEED}" render_one branding/sieve-wizard.tape
    SPEED="${USER_SPEED:-$BENCH_SPEED}"   render_one branding/sieve-benchmark.tape
    ;;
  *)
    echo "usage: $0 {quick|wizard|benchmark|all}" >&2
    exit 2
    ;;
esac

echo "✓ done — GIFs in branding/"

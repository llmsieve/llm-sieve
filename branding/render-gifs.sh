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
VHS_BIN=""
if command -v vhs >/dev/null 2>&1; then
  VHS_BIN="vhs"
elif [[ -x "$HOME/go/bin/vhs" ]]; then
  VHS_BIN="$HOME/go/bin/vhs"
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

# Ensure no proxy is running from a prior session.
"$REPO_ROOT/.venv/bin/sieve" stop >/dev/null 2>&1 || true

# SPEED is the playback multiplier applied via ffmpeg after VHS is done.
# e.g. SPEED=2.5 means 40-second raw recording becomes a 16-second GIF.
# Defaults to 2.5× so the tape can record real inference turns without
# the final GIF feeling laggy.
SPEED="${SPEED:-2.5}"

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
    docker run --rm \
      --network host \
      -v "$REPO_ROOT":/work \
      -v "$SHIM_DIR":/shim \
      -v "$REPO_ROOT/.venv":/venv \
      -v "$fake_home":/tmphome \
      -e "HOME=/tmphome" \
      -e "PATH=/shim:/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      -w /work \
      ghcr.io/charmbracelet/vhs "$tape"
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

case "${1:-all}" in
  quick)
    render_one branding/sieve-quick-install.tape
    ;;
  wizard)
    render_one branding/sieve-wizard-install.tape
    ;;
  all)
    render_one branding/sieve-quick-install.tape
    render_one branding/sieve-wizard-install.tape
    ;;
  *)
    echo "usage: $0 {quick|wizard|all}" >&2
    exit 2
    ;;
esac

echo "✓ done — GIFs in branding/"

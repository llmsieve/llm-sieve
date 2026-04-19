#!/usr/bin/env bash
#
# Render the Sieve CLI demo GIFs from their VHS tapes.
#
# Usage:
#   branding/render-gifs.sh quick      # renders sieve-quick-install.gif
#   branding/render-gifs.sh wizard     # renders sieve-wizard-install.gif
#   branding/render-gifs.sh all        # renders both
#
# Requirements:
#   - Docker with the charmbracelet/vhs image pulled
#   - A locally-installed `sieve` (from `pip install -e .` in the repo)
#   - A running Ollama on :11434 with the model in sieve.yaml loaded
#
# The script runs `sieve stop` before and after each recording so the
# tape starts from a clean slate, and uses a fake `pip` shim so the
# `pip install llm-sieve` line in the tape looks realistic without
# actually hitting PyPI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x "$REPO_ROOT/.venv/bin/sieve" ]]; then
  echo "error: expected a venv at .venv/ with sieve installed (try: uv venv + uv pip install -e .)" >&2
  exit 1
fi

# Build a throwaway VHS shim directory that lives only for this run.
SHIM_DIR="$(mktemp -d)"
trap 'rm -rf "$SHIM_DIR"' EXIT

# A believable-looking fake pip — VHS types `pip install llm-sieve`
# and we want to show a realistic success banner, not the 404 you'd get
# from a re-install or a command-not-found from the VHS container.
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

# Ensure a clean proxy state before each recording.
"$REPO_ROOT/.venv/bin/sieve" stop >/dev/null 2>&1 || true

render_one() {
  local tape="$1"
  # Give each recording an isolated $HOME so we don't clobber ~/.sieve.
  local fake_home
  fake_home="$(mktemp -d)"
  echo "→ rendering $tape  (isolated HOME: $fake_home)"
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
  rm -rf "$fake_home"
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

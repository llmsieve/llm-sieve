# VHS + llm-sieve pre-installed, for rendering the branding GIFs.
#
# Why not just run VHS against the repo's .venv?
# The .venv's scripts have hardcoded shebangs (/home/ath/.../.venv/
# bin/python3) which don't resolve inside the container. Installing
# the wheel INTO a container-local venv produces shebangs that
# resolve naturally.
#
# Why not run native VHS on the host?
# VHS's bundled chromium crashes on some Debian kernels due to
# zygote/ namespace sandbox restrictions. Docker gives us a stable
# render environment.

FROM ghcr.io/charmbracelet/vhs:latest

# The VHS image ships Python 3.13 but ensurepip isn't available.
# Install python3-venv, then build our isolated venv.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/sieve \
    && /opt/sieve/bin/pip install --no-cache-dir --upgrade pip

# Wheel gets copied at build time. Rebuild the image whenever the
# wheel changes (the render script handles this).
COPY dist/llm_sieve-1.0.0-py3-none-any.whl /tmp/llm_sieve-1.0.0-py3-none-any.whl
RUN /opt/sieve/bin/pip install --no-cache-dir /tmp/llm_sieve-1.0.0-py3-none-any.whl \
    && rm /tmp/llm_sieve-1.0.0-py3-none-any.whl

# Pre-download the FastEmbed ONNX model so the installer's
# "Preparing embedding model…" step finishes in milliseconds during
# recording. Without this the GIF has a 10-15 second block of HF
# download progress + token warnings, which muddies the story.
# The model is cached at /root/.cache/fastembed.
RUN /opt/sieve/bin/python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')" \
    2>/dev/null || true

# Pip / pipx shims for the "pipx install llm-sieve" demo line.
# These match the behaviour of the host render script's shims.
RUN mkdir -p /opt/shim \
    && printf '#!/bin/bash\nif [[ "$1" == "install" ]]; then shift; echo "Collecting $*"; echo "  Downloading llm_sieve-1.0.0-py3-none-any.whl (142 kB)"; echo "Installing collected packages: llm-sieve"; echo "Successfully installed $*"; else echo "pip (shim) — passthrough disabled in demo mode"; fi\n' > /opt/shim/pip \
    && printf '#!/bin/bash\nif [[ "$1" == "install" ]]; then shift; echo "  installed package $* 1.0.0, installed using Python 3.13.5"; echo "  These apps are now globally available"; echo "    - sieve"; echo "    - sieve-install"; echo "done! \xe2\x9c\xa8 \xf0\x9f\x8e\x89 \xe2\x9c\xa8"; else echo "pipx (shim) — passthrough disabled in demo mode"; fi\n' > /opt/shim/pipx \
    && chmod +x /opt/shim/pip /opt/shim/pipx

# Helper script the wizard tape calls to pre-seed facts in the
# store without going through the full LLM pipeline. Keeps the
# preparation fast + deterministic during Hide blocks in tapes.
RUN cat > /usr/local/bin/sieve-seed-facts <<'PYSEED' \
 && chmod +x /usr/local/bin/sieve-seed-facts
#!/opt/sieve/bin/python3
"""Seed a handful of facts directly into the store for demo recording."""
import uuid
from datetime import datetime, timezone
from sieve.config import RecallConfig
from sieve.store import MemoryStore

FACTS = [
    "User is Sam",
    "User is a marine biologist",
    "User lives in Porto",
    "User has a dog called Mabel",
    "Mabel is a border terrier",
    "Users partner is Alex",
    "User has a cat called Luna",
]

cfg = RecallConfig.load()
ms = MemoryStore(cfg.store)
ms.open()
now = datetime.now(timezone.utc).isoformat()
for content in FACTS:
    ms._conn.execute(
        "INSERT INTO facts (id, content, created_at, confidence, "
        "fact_type, status, source) VALUES (?, ?, ?, 0.9, "
        "'objective', 'current', 'writer_s1')",
        (uuid.uuid4().hex, content, now),
    )
ms._conn.commit()
ms.close()
PYSEED

# Benchmark replay: the real `sieve benchmark` run takes 5-10 minutes
# of actual LLM inference. For the GIF we want the exact visual output
# without the wait. We ship a captured stdout fixture + a replay helper
# that prints it with realistic pacing, and shadow `sieve` in /opt/shim
# so a tape typing `sieve benchmark ...` dispatches to the replay.
# Everything else (sieve wizard, sieve status, etc.) falls through to
# the real /opt/sieve/bin/sieve.
COPY branding/bench-fixture.txt /opt/bench-fixture.txt

RUN cat > /opt/shim/sieve-bench-demo <<'PYREPLAY' \
 && chmod +x /opt/shim/sieve-bench-demo
#!/opt/sieve/bin/python3
"""Replay a captured `sieve benchmark` run with GIF-friendly pacing.

Two segments:
  1. Stream the captured run up to and including the cost panel.
     Insert short sleeps at panel boundaries for read time.
  2. Clear the screen and render a dominant hero banner — the
     "Sieve sent 90% fewer tokens" line. This keeps the last
     visible frame of the GIF clean and screenshot-worthy instead
     of leaving three dense trailing panels (trade-offs, known
     limits, reproduce) cluttering the hero.
"""
import sys
import time
from pathlib import Path

FIXTURE = Path("/opt/bench-fixture.txt")
lines = FIXTURE.read_text().splitlines(keepends=True)

# Substrings that mark panel boundaries (they appear exactly once in
# the fixture). After the line containing the key prints, we sleep
# the given seconds so viewers can read the panel that just closed.
PAUSES = [
    ("[3] Run a same-context",            3.0),   # context-config panel closed
    ("pass --grader-model",               1.5),   # self-grading note closed
    ("Estimated time:",                   1.0),
    ("run 1/1 — baseline pass",           2.0),   # "running" feel
    ("Latency: not reported",             2.5),   # methodology panel closed
    ("Facts learned",                     2.5),   # results table closed
    ("Per 1K runs:",                      0.5),   # cost panel closed — brief hold before cut
]

# Stop streaming after the cost panel's closing border. Everything
# below is replaced by the hero-only frame.
STOP_AFTER = "Per 1K runs:"
saw_stop = False
idx = 0
for line in lines:
    sys.stdout.write(line)
    sys.stdout.flush()
    time.sleep(0.01)
    while idx < len(PAUSES) and PAUSES[idx][0] in line:
        time.sleep(PAUSES[idx][1])
        idx += 1
    if STOP_AFTER in line:
        saw_stop = True
        continue
    if saw_stop and line.strip().startswith("╰"):
        break  # close of cost panel — cut here

# Small beat, then clear + hero banner.
time.sleep(1.2)
sys.stdout.write("\x1b[2J\x1b[H")  # clear screen, cursor home

# Render the hero as three centred lines. 100 cols wide matches the
# terminal's rich-panel width so the rule lines feel native.
BOLD   = "\x1b[1m"
GREEN  = "\x1b[32m"
DIM    = "\x1b[2m"
RESET  = "\x1b[0m"
rule = "─" * 100

print()
print()
print(f"  {DIM}{rule}{RESET}")
print()
print(f"  {BOLD}{GREEN}Sieve sent 90% fewer tokens{RESET} "
      f"(220,683 → 21,513) over 1 × 15-turn qwen3.5:9b")
print(f"  conversations on the 'medium' fixture; "
      f"correct recalls {BOLD}5/6{RESET}; trap {BOLD}refused{RESET}.")
print()
print(f"  {DIM}{rule}{RESET}")
print()
print(f"  {DIM}📎 Shareable report: "
      f"/root/.sieve/benchmarks/2026-04-20T21-20-46Z.md{RESET}")
sys.stdout.flush()

# Hero hold — let viewers read + screenshot.
time.sleep(4.0)
PYREPLAY

# Shadow `sieve`: dispatch `benchmark` to the replay, forward everything
# else to the real CLI. Placed in /opt/shim which is first on PATH.
RUN cat > /opt/shim/sieve <<'SIEVESHIM' \
 && chmod +x /opt/shim/sieve
#!/bin/bash
if [[ "$1" == "benchmark" ]]; then
  exec /opt/shim/sieve-bench-demo "${@:2}"
fi
exec /opt/sieve/bin/sieve "$@"
SIEVESHIM

ENV PATH="/opt/shim:/opt/sieve/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

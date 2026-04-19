#!/bin/bash
# Sterilise — clean slate for next test cycle.
#
# Resolves the repo root relative to this script so it works from any clone
# location. The active store lives at ~/.sieve/memory.db (see
# data/sieve.yaml). Prior versions wiped the wrong directory, so cycles
# seeded onto stale state without noticing.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${SIEVE_CONFIG:-data/sieve.yaml}"

echo "=== STERILISING (repo: $REPO_ROOT) ==="

# 1. Stop Recall if running
pkill -f "src.cli start" 2>/dev/null || true
pkill -f "src\\.main" 2>/dev/null || true
sleep 2

# 2. Delete canonical memory store + keyfile
rm -f ~/.sieve/memory.db ~/.sieve/memory.db-shm ~/.sieve/memory.db-wal ~/.sieve/.sieve_key

# 3. Reinitialise store
cd "$REPO_ROOT"
PYTHONPATH=. SIEVE_CONFIG="$CONFIG" .venv/bin/python -c "
from src.config import RecallConfig
from src.store import MemoryStore
c = RecallConfig.load()
s = MemoryStore(c.store)
s.open()
s.init_schema()
print('Store initialised:', s.stats())
s.close()
"

# 4. Clear cached runtime state
rm -f "$REPO_ROOT"/data/*.log "$REPO_ROOT"/data/fingerprint_cache.json

# 5. Start Recall proxy in background
cd "$REPO_ROOT"
nohup .venv/bin/python -m src.cli start --config "$CONFIG" \
  > "$REPO_ROOT/data/recall.log" 2>&1 &
echo "Recall PID: $!"
sleep 5

# 6. Verify Recall running
for i in 1 2 3 4 5; do
  if curl -sf http://127.0.0.1:11435/api/tags > /dev/null; then
    echo "OK Recall running"
    break
  fi
  sleep 1
done

# 7. Verify Ollama responding
curl -sf http://127.0.0.1:11434/api/tags > /dev/null \
  && echo "OK Ollama running" \
  || echo "FAIL Ollama not responding"

echo "=== STERILISED — ready for next cycle ==="

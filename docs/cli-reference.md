# CLI reference

Every Sieve command, flag, and option. For a guided walkthrough see [Getting started](getting-started.md); for config-key detail see [Configuration](configuration.md).

## Entry points

The `llm-sieve` distribution installs two executables:

| Command | Purpose |
|---|---|
| [`sieve-install`](#sieve-install) | One-shot first-run setup. Provider → model → autostart → start → ready panel. |
| [`sieve`](#top-level-commands) | Day-to-day management. Interactive menu when called without a subcommand. |

`sieve-install` is the command you run once. Everything else is under `sieve`.

## `sieve-install`

One flow, no surprises. Walks through provider detection, model selection, embedding-model download, store initialisation, and optionally enables autostart. Ends on a green "Sieve is ready" panel with your config summary.

```bash
sieve-install
sieve-install --no-input --provider http://127.0.0.1:11434 --model qwen3.5:9b
sieve-install --provider https://api.openai.com/v1 --model gpt-4o-mini --api-key sk-...
```

**Flags:**

| Flag | Purpose |
|---|---|
| `--no-input` | Skip all prompts; use defaults. For CI / scripted installs. |
| `--provider URL` | Skip the "where's your LLM" step. |
| `--model NAME` | Skip the model picker. |
| `--api-key TOKEN` | Bearer token for cloud endpoints. Read from env (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) if `--provider` is a recognised cloud URL. |

Safe to rerun any time. Probes the provider, rolls back cleanly on Ctrl-C, preserves the existing config as a backup if you reinstall.

---

## Top-level commands

| Command | Purpose |
|---|---|
| [`sieve`](#sieve-wizard) / [`sieve wizard`](#sieve-wizard) | Interactive management menu |
| [`sieve start`](#sieve-start) | Run the proxy in the foreground |
| [`sieve stop`](#sieve-stop) | Gracefully stop a running proxy |
| [`sieve restart`](#sieve-restart) | Stop and start in one step |
| [`sieve status`](#sieve-status) | Proxy + store state |
| [`sieve demo`](#sieve-demo) | Scripted 6-message demo (sandboxed) |
| [`sieve benchmark`](#sieve-benchmark) | Reproducible baseline-vs-Sieve benchmark (sandboxed) |
| [`sieve init`](#sieve-init) | Legacy: create `~/.sieve/`, write default config, init store |
| [`sieve uninstall`](#sieve-uninstall) | Remove Sieve (soft by default) |

**Subcommand groups:**

| Group | Purpose |
|---|---|
| [`sieve store`](#store) | Inspect / export / wipe the memory store |
| [`sieve config`](#config) | Show / edit / reset runtime configuration |
| [`sieve key`](#key) | Rotate / show / import / export the encryption key |
| [`sieve backup`](#backup) | Create / list / restore encrypted backups |

---

## `sieve wizard`

Interactive management menu. Same as running `sieve` with no arguments.

```bash
sieve                # shortest form
sieve wizard         # explicit
```

State-aware top menu:

1. **Reinstall** — change provider / re-init; preserves the existing config as a backup.
2. **Service** — start / stop / restart / enable autostart.
3. **Store** — stats, facts, entities, episodes, recent activity.
4. **Config** — adjust settings without editing YAML.
5. **Benchmark** — measure Sieve's value; same as `sieve benchmark`.
6. **Demo** — 6-turn scripted conversation.
7. **Uninstall** — stop, disable autostart, remove `~/.sieve/`.

Options that don't apply right now (e.g. "stop" when the proxy isn't running) are marked unavailable rather than hidden. Press `b` at any submenu to go back, `q` to quit.

## `sieve start`

Run the proxy in the foreground until interrupted (Ctrl-C).

**Flags:**

| Flag | Purpose |
|---|---|
| `-c, --config PATH` | Load a specific `sieve.yaml` instead of the default lookup. |
| `--host HOST` | Override the listen host. |
| `-p, --port N` | Override the listen port. |
| `-v, --verbose` | Enable debug logging. |

Writes a PID file to `~/.sieve/sieve.pid` so `sieve status` and `sieve stop` can find the process.

## `sieve stop`

Sends `SIGTERM` to the PID in `~/.sieve/sieve.pid` and waits up to 5 seconds for graceful shutdown. Cleans up stale PID files if the recorded process has already exited.

## `sieve restart`

Runs `sieve stop` then replaces the current process with `sieve start` via `os.execvp`. Flags are forwarded:

```bash
sieve restart --port 11500 --verbose
```

**Flags:** `-p, --port N`, `-v, --verbose`.

## `sieve status`

Prints the running state, listen address, and high-level store statistics (fact count, entity count). If the proxy isn't running, shows the "start Sieve with…" hint.

## `sieve demo`

Sends six scripted messages through a sandboxed proxy (a scratch store that's torn down afterward). Demonstrates fact extraction and the absence-signal trap without touching your real store.

**Flags:**

| Flag | Purpose |
|---|---|
| `--wait-for-write` / `--no-wait-for-write` | Poll the store after each turn until the async writer has committed before sending the next. Default: on. |
| `--max-wait SECONDS` | Max wait per turn if `--wait-for-write` is on. |
| `--use-main-store` | Run against the live proxy and store. **Advanced** — adds demo facts to your real data. Requires `sieve start`. |

## `sieve benchmark`

Reproducible baseline-vs-Sieve comparison. Drives a 15-message scripted conversation twice — once direct to the LLM, once through Sieve — and prints the delta. Sandboxed by default. Always writes a shareable markdown report to `~/.sieve/benchmarks/`.

Run without flags in an interactive terminal and you get a short preflight wizard (fixture size, model, grader, turns, runs, pricing). Any flag suppresses the wizard; `--no-input` always does.

**Flags:**

| Flag | Purpose |
|---|---|
| `-c, --config PATH` | Alternate `sieve.yaml`. |
| `--fixture {small,medium,large,xlarge}` | Payload size. `small` = light agent, `medium` = Cursor-like (default), `large` = Claude Code mid-session, `xlarge` = autonomous run. |
| `--model NAME` | Model to test. Defaults to `provider.default_model`. |
| `--grader-model NAME` | Model used to grade recall + trap. Defaults to `--model` (self-grading, not recommended for shareable reports). |
| `--turns N` | Turns per run. Default 15. |
| `--runs N` | Number of full script runs (for mean ± stddev). Default 3. |
| `--pricing TIER` | Pricing tier for the cost panel: `local`, `claude-opus`, `claude-sonnet`, `claude-haiku`, `gpt-4o`, `gpt-4o-mini`. Default `local` (no $ shown). |
| `--format {rich,json,markdown}` | Terminal output format. A markdown report is always written regardless. |
| `--no-input` | Skip the wizard; use flag values and defaults. |
| `--use-main-store` | Run against the live proxy and store. **Advanced.** |

The prompts don't depend on the model knowing specific facts, so comparisons are apples-to-apples across models.

## `sieve init`

Legacy setup command. `sieve-install` is the preferred entry point for new installs — it wraps the same work with a nicer flow and rollback on failure. `sieve init` is still supported for users who learned it before `sieve-install` existed, and for CI scripts that already call it.

```bash
sieve init                       # zero-prompt, uses defaults
sieve init --wizard              # six-step interactive setup (older UX)
sieve init --force               # overwrite an existing ~/.sieve/
sieve init --provider URL        # override the auto-detected provider
```

**Flags:**

| Flag | Purpose |
|---|---|
| `--provider URL` | LLM provider base URL. |
| `--wizard` | Six-step interactive guided setup. |
| `--force` | Reinitialise even if `~/.sieve/` already exists. |

## `sieve uninstall`

Remove Sieve. Default is **soft** — leaves `~/.sieve/` in place so your data survives.

```bash
sieve uninstall            # soft: preserves ~/.sieve/
sieve uninstall --hard     # requires typing DELETE; removes ~/.sieve/
```

**Flags:**

| Flag | Purpose |
|---|---|
| `--soft` *(default)* | Preserves data; prints the `pipx uninstall llm-sieve` instruction. |
| `--hard` | Requires typing `DELETE`; recursively removes `~/.sieve/`. |

`--soft` and `--hard` are mutually exclusive. Sieve can't remove the Python package it's currently executing from — that's what the follow-up `pipx uninstall llm-sieve` is for.

---

## store

```bash
sieve store <subcommand>
```

### `sieve store init`

Create the encrypted database and schema at the configured path. Idempotent — re-running on an existing DB no-ops.

### `sieve store status`

Compact summary: path, file size, row counts.

### `sieve store stats`

Detailed statistics — per-table row counts (facts, entities, relationships, episodes, preferences, sessions, known_unknowns, vec_facts, audit_log, fingerprints), database file size, average facts per entity.

### `sieve store facts [--limit N] [--search QUERY]`

List current facts, newest first. `--limit` defaults to 50. `--search` filters on substring match against the fact content column.

### `sieve store entities [--limit N] [--search QUERY]`

List entities with per-entity fact counts.

### `sieve store relationships [--limit N]`

List `source → rel → target` triples with confidence scores.

### `sieve store episodes [--limit N]`

List episodic summaries.

### `sieve store export --format json|csv --output PATH`

Decrypted dump for backup or migration.

- `--format json` — single JSON file with `facts` / `entities` / `relationships` / `episodes` sections.
- `--format csv` — a directory of `facts.csv` / `entities.csv` / `relationships.csv` / `episodes.csv`.

### `sieve store wipe`

Delete all data rows from the store while preserving the schema and keyfile. Requires typing `WIPE` to confirm. Use when you want to start over without re-initialising.

### `sieve store migrate --to PATH`

Copy the store to a new location, verify integrity. Update `store.path` in `sieve.yaml` afterwards.

---

## config

```bash
sieve config <subcommand>
```

### `sieve config show`

Prints the effective configuration as a table. Non-default values are highlighted so you can see what you've changed relative to shipped defaults.

### `sieve config set <key> <value>`

Set a specific YAML path, with type coercion and validation:

```bash
sieve config set provider.base_url http://192.168.1.100:11434
sieve config set listen.port 11436
sieve config set pipeline.context_format structured
sieve config set ablation.absence_signal false
```

Integers, floats, and booleans are coerced from strings. Enum values (`context_format`, `tools.compression`, `embeddings.provider`, `mode_override`) are validated against their allowed sets. Unknown paths are rejected — the full settable list is in [Configuration](configuration.md).

Restart Sieve for the change to take effect.

### `sieve config reset`

Reset configuration to shipped defaults. Requires confirmation. Preserves two fields the user almost never wants reset: `provider.base_url` and `store.path`.

### `sieve config edit`

Open `~/.sieve/sieve.yaml` in `$EDITOR`. After save, the file is re-parsed; invalid YAML triggers an automatic rollback to the prior content.

---

## key

```bash
sieve key <subcommand>
```

### `sieve key show`

Keyfile location and SHA-256 fingerprint (first 16 hex chars). The raw key is never printed.

### `sieve key rotate`

Re-encrypt the store with a new key. Requires typing `ROTATE` to confirm. Prompts for auto-generate (default) or custom passphrase. Uses SQLCipher's `PRAGMA rekey` to rewrite every page in-place; rolls back the keyfile on any failure.

Always run `sieve backup create` before rotating.

### `sieve key export`

Print the current key to stdout for backup. Requires explicit confirmation. Store the output somewhere safe — anyone with this key can read your store.

### `sieve key import <keyfile>`

Import a key from a file. Verifies the key opens the current store before overwriting `~/.sieve/.sieve_key`.

---

## backup

```bash
sieve backup <subcommand>
```

### `sieve backup create [--output PATH]`

Create an encrypted backup of the store. Timestamped filename (`recall_backup_YYYY-MM-DDTHHMMSS.db.enc`) with SHA-256 checksum sidecar. Default destination: `~/.sieve/backups/`.

### `sieve backup list`

Table of available backups: timestamp, size, checksum status, full path.

### `sieve backup restore <backup-id>`

Restore from a named backup. Asks for confirmation before overwriting the current store. The current store is copied to `pre_restore_*.db.bak` first in case you need to undo.

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Runtime error — store not found, wrong confirmation, provider unreachable |
| 2 | Invalid input — unknown config path, bad enum value, conflicting flags |

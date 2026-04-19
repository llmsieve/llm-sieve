# CLI reference

Every Sieve command, flag, and option. Grouped as it appears in `sieve --help`.

For a guided walkthrough use [Getting started](getting-started.md); for conceptual detail on individual config keys see [Configuration](configuration.md).

## Top-level commands

| Command                       | Purpose                                                 |
| ----------------------------- | ------------------------------------------------------- |
| [`sieve init`](#sieve-init)   | Create `~/.sieve/`, write default config, create store  |
| [`sieve start`](#sieve-start) | Run the proxy in the foreground                         |
| [`sieve stop`](#sieve-stop)   | Gracefully stop a running proxy                         |
| [`sieve restart`](#sieve-restart) | Stop and start in one step                          |
| [`sieve status`](#sieve-status) | Show proxy + store state                              |
| [`sieve demo`](#sieve-demo)   | Scripted 6-message demo against a running proxy         |
| [`sieve benchmark`](#sieve-benchmark) | Reproducible 15-message benchmark + summary    |
| [`sieve uninstall`](#sieve-uninstall) | Remove Sieve (soft by default)                  |

Subcommand groups:

| Group                           | Purpose                                   |
| ------------------------------- | ----------------------------------------- |
| [`sieve store`](#store)         | Inspect / export / wipe the memory store  |
| [`sieve config`](#config)       | Show / edit / reset runtime configuration |
| [`sieve key`](#key)             | Rotate / show / import / export the key   |
| [`sieve backup`](#backup)       | Create / list / restore encrypted backups |

---

## `sieve init`

Initialise Sieve. Two modes:

### Default (lazy)

```bash
sieve init
```

Zero prompts. Auto-detects Ollama on `localhost:11434`, generates a 256-bit encryption key, downloads the FastEmbed embedding model, writes `~/.sieve/sieve.yaml` with shipping defaults, and initialises the encrypted store.

**Flags:**
- `--provider URL` — override the auto-detected provider base URL.
- `--force` — reinitialise even if `~/.sieve/` already exists.

### Wizard

```bash
sieve init --wizard
```

Six interactive steps:

1. **Provider** — `1` Ollama (auto-detect) · `2` OpenAI · `3` Anthropic · `4` Custom URL.
2. **Model** — for Ollama, pick from the list of local models. For cloud providers, free-text.
3. **Port** — the port Sieve itself listens on. Defaults to `11435`. Rejects bound ports.
4. **Encryption** — generate a random key (default) or supply a custom passphrase.
5. **Store location** — defaults to `~/.sieve/memory.db`.
6. **Confirmation** — summary of your choices; only applied if you confirm.

Every step retries on failure (unreachable provider, port in use, etc.). The default lazy path is unchanged when `--wizard` is not passed.

## `sieve start`

Run the proxy in the foreground until interrupted.

**Flags:**
- `-c, --config PATH` — load a specific `sieve.yaml` instead of the default lookup.
- `--host` — override the listen host (default from config).
- `-p, --port N` — override the listen port.
- `-v, --verbose` — enable debug logging.

Writes a PID file to `~/.sieve/sieve.pid` so `sieve status` and `sieve stop` can find the running process.

## `sieve stop`

Sends `SIGTERM` to the PID in `~/.sieve/sieve.pid` and waits up to 5 seconds for graceful shutdown. Cleans up stale PID files if the recorded process has already exited.

## `sieve restart`

Runs `sieve stop` then replaces the current process with `sieve start` via `os.execvp`, so the new foreground proxy takes over the terminal. Flags you pass are forwarded:

```bash
sieve restart --port 11500 --verbose
```

**Flags:** `-p, --port N`, `-v, --verbose`.

## `sieve status`

Prints the running state, listen address, and high-level store statistics (fact count, entity count). If the proxy is not running, shows the "start Sieve with" hint.

## `sieve benchmark`

Reproducible proof that anyone can run on their own hardware. Drives a 15-message scripted conversation through the proxy and reports per-message inbound/outbound tokens, facts learned, response time, and whether the absence-signal layer fires on a trap query about something that was never introduced.

Flags:

- `-c, --config PATH` — alternate sieve.yaml
- `--model NAME` — override the model (defaults to `provider.default_model`)

Requires the proxy to be running: start it with [`sieve start`](#sieve-start) in another terminal first. The benchmark works against any OpenAI-compatible or Ollama-compatible model — the prompts do not depend on the model knowing specific facts.

At the end of [`sieve init --wizard`](#sieve-init), Sieve offers to run the benchmark for you so you can verify the reduction on your machine before wiring anything up.

## `sieve demo`

Sends six scripted messages through a running proxy to demonstrate fact extraction and the absence-signal trap. Requires `sieve start` in another terminal.

## `sieve uninstall`

Remove Sieve. Default behaviour is **soft** — leaves `~/.sieve/` (store, config, key) in place so your data survives.

**Flags:**
- `--soft` *(default when no flag is passed)* — preserves data; prints the `pip uninstall llm-sieve` instruction.
- `--hard` — requires typing `DELETE` to confirm; recursively removes `~/.sieve/`.

`--soft` and `--hard` are mutually exclusive.

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

Detailed statistics — per-table row counts (facts, entities, relationships, episodes, preferences, sessions, known_unknowns, vec_facts, audit_log, fingerprints), database file size, and average facts per entity.

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

Delete all data rows from the store while preserving the schema and keyfile. Requires typing the literal string `WIPE` to confirm. Use this when you want to start over without re-initialising.

### `sieve store migrate --to PATH`

Copy the store to a new location, verify integrity. Update `store.path` in `sieve.yaml` afterwards.

---

## config

```bash
sieve config <subcommand>
```

### `sieve config show`

Prints the effective configuration as a table. Non-default values are highlighted so you can see what you've changed relative to the ship defaults.

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

Reset the configuration to ship defaults. Requires confirmation. Preserves two fields the user almost never wants reset: `provider.base_url` and `store.path`.

### `sieve config edit`

Open `~/.sieve/sieve.yaml` in `$EDITOR`. After save, the file is re-parsed as `RecallConfig`; invalid YAML triggers an automatic rollback to the prior content.

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

Create an encrypted backup of the store. Timestamped filename (`recall_backup_YYYY-MM-DDTHHMMSS.db.enc`) with SHA-256 checksum sidecar. By default writes to `~/.sieve/backups/`.

### `sieve backup list`

Table of available backups: timestamp, size, checksum status, full path.

### `sieve backup restore <backup-id>`

Restore from a named backup. Asks for confirmation before overwriting the current store. The current store is copied to a `pre_restore_*.db.bak` file first in case you need to undo.

---

## Exit codes

| Code | Meaning                                                                 |
| ---- | ----------------------------------------------------------------------- |
| 0    | Success                                                                 |
| 1    | Runtime error — store not found, wrong confirmation, provider unreachable |
| 2    | Invalid input — unknown config path, bad enum value, conflicting flags  |

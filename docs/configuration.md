# Configuration

Sieve reads its configuration from a single YAML file. After `sieve init`, that file lives at `~/.sieve/sieve.yaml`. You can override the path with `sieve start --config <path>` or by setting `SIEVE_CONFIG`.

The shipping example — with commentary on every option — is [`sieve.example.yaml`](https://github.com/llmsieve/llm-sieve/blob/main/sieve.example.yaml). This page documents the same options with more context.

## File lookup order

`sieve start` (and `sieve status`, etc.) resolves the config in this order:

1. `--config` flag on the command line
2. `SIEVE_CONFIG` environment variable
3. `./sieve.yaml` in the current working directory
4. `~/.sieve/sieve.yaml`

The first match wins; nothing is merged.

## Editing via the CLI

You rarely need to open `sieve.yaml` in a text editor. The `sieve config` commands cover the common cases:

```bash
sieve config show                                    # current values (non-default highlighted)
sieve config set listen.port 11436                   # coerced + validated before write
sieve config set provider.base_url http://host:PORT
sieve config edit                                    # open in $EDITOR, rolls back invalid YAML
sieve config reset                                   # ship defaults; preserves provider URL + store path
```

`sieve config set` whitelists the keys that can be set this way — see the [CLI reference](cli-reference.md#config) for the full list and validation rules.

## Reference

Every top-level section below maps to a block in `sieve.yaml`. Defaults shown are what `sieve init` writes.

### `listen`

The proxy's HTTP listener.

```yaml
listen:
  host: 127.0.0.1
  port: 11435
```

| Key | Default | Notes |
|-----|---------|-------|
| `host` | `127.0.0.1` | Loopback only by default. Set to `0.0.0.0` to accept from the LAN — do this only if you have another layer enforcing authentication. |
| `port` | `11435` | Deliberately adjacent to Ollama's `11434` to make intent obvious. |

Override either at startup without editing the file: `sieve start --host 0.0.0.0 --port 11500`.

### `provider`

The upstream LLM endpoint.

```yaml
provider:
  type: auto
  base_url: http://127.0.0.1:11434
  default_model: qwen3.5:9b
  options:
    think: false
```

| Key | Default | Notes |
|-----|---------|-------|
| `type` | `auto` | Leave as `auto` — Sieve detects the wire protocol (Ollama vs OpenAI-compatible) from the endpoint's responses. |
| `base_url` | `http://127.0.0.1:11434` | Where to forward requests. Any OpenAI-compatible endpoint or Ollama server. |
| `default_model` | `qwen3.5:9b` | Used when the inbound request does not pin a model, and for internal prompts (classification, writer, etc.) when `writer.model` is `auto`. |
| `options.think` | `false` | Sent to model families that support a "thinking" mode (Qwen, DeepSeek). Leave off for Gemma, Mistral, and other families that do not understand the flag. |

### `embeddings`

The embedding backend used for vector retrieval.

```yaml
embeddings:
  provider: fastembed
```

| Key | Default | Notes |
|-----|---------|-------|
| `provider` | `fastembed` | In-process ONNX Runtime using `BAAI/bge-small-en-v1.5` (384-dim, ~50 MB). Auto-downloaded and cached by FastEmbed. |
| `ollama_url` | — | Only consulted when `provider: ollama`. Base URL of the Ollama server to call for embeddings. |
| `ollama_model` | — | Only consulted when `provider: ollama`. Embedding model name, e.g. `nomic-embed-text-v2-moe`. |

To switch to an Ollama-hosted embedding pipeline:

```yaml
embeddings:
  provider: ollama
  ollama_url: http://127.0.0.1:11434
  ollama_model: nomic-embed-text-v2-moe
```

The FastEmbed default is the recommended path. Use Ollama only if you already operate an embedding pipeline there and want to consolidate.

### `store`

The encrypted memory store.

```yaml
store:
  path: ~/.sieve/memory.db
```

| Key | Default | Notes |
|-----|---------|-------|
| `path` | `~/.sieve/memory.db` | SQLCipher-encrypted SQLite database. The keyfile is written next to it on first init. Back both up together. |
| `embedding_model` | — | Only used when `embeddings.provider: ollama`. Records which model produced the stored vectors so incompatible swaps are rejected. |
| `embedding_dimensions` | — | Only used when `embeddings.provider: ollama`. Dimensionality of the stored vectors. Must match the model. |

Under FastEmbed the dimensions and model name are fixed; the two Ollama-only keys are ignored.

### `pipeline`

Retrieval pipeline shape.

```yaml
pipeline:
  conversation_turns: 3
  max_rounds: 5
  core_facts_size: 30
  context_format: auto
```

| Key | Default | Notes |
|-----|---------|-------|
| `conversation_turns` | `3` | Recent turns preserved verbatim in the lean payload. Anything older is compressed or dropped. |
| `max_rounds` | `5` | Upper bound on multi-hop retrieval rounds per request. |
| `core_facts_size` | `30` | Number of always-on "core" facts included in every lean payload. |
| `context_format` | `auto` | How retrieved context is formatted for the upstream model. Leave as `auto`. |

### `profile_owner`

The canonical identity for the conversation. Pinned into fact extraction and used by the ghost-fact validator to reject fabrications.

```yaml
profile_owner:
  name: "Jamie Rivera"
  aliases:
    - "Jamie"
    - "I"
    - "me"
    - "the user"
    - "user"
```

| Key | Default | Notes |
|-----|---------|-------|
| `name` | `"Jamie Rivera"` | Canonical display name. Change this to the actual user's name for a real deployment. |
| `aliases` | list of common pronouns | Tokens the extractor should resolve back to `name` when it sees them in first-person text. |

For a single-user personal setup, set `name` to the user's name; the defaults for `aliases` are usually fine.

### `writer`

The Stage-2 fact extractor. Runs after a turn completes to distil durable facts into the store.

```yaml
writer:
  model: auto
  fallback_model: auto
  num_ctx: 4096
  ghost_validator_enabled: true
```

| Key | Default | Notes |
|-----|---------|-------|
| `model` | `auto` | `auto` routes extraction to `provider.default_model` — no second model to load. Override with an explicit model name to pin a dedicated writer. |
| `fallback_model` | `auto` | Used when the primary writer call fails. `auto` means the same as `model`. |
| `num_ctx` | `4096` | Context window allocated to the writer call. |
| `ghost_validator_enabled` | `true` | Post-extraction validator that rejects facts unsupported by the turn text. Keep on. |

### `retrieval`

Retrieval-side knobs.

```yaml
retrieval:
  temporal_dedup: true
```

| Key | Default | Notes |
|-----|---------|-------|
| `temporal_dedup` | `true` | Collapses near-duplicate facts that disagree only on timestamp, keeping the most recent. Keep on unless you are debugging retrieval. |

!!! note
    The shipping `sieve.example.yaml` uses the key `temporal_dedup_enabled`, which the loader does not read — it falls back to the default. Use `temporal_dedup` (no suffix) to actually override the value. This will be unified in a future release.

### `tools`

Tool-schema compression — the largest single win on agent payloads.

```yaml
tools:
  enabled: true
  compression: moderate
  l1_threshold: 0.5
  fallback_include_all: true
  max_tools_injected: 10
```

| Key | Default | Notes |
|-----|---------|-------|
| `enabled` | `true` | Master switch. Turn off only to measure the uncompressed baseline. |
| `compression` | `moderate` | Shape of the compression — `moderate` strips schemas and keeps signatures; other levels are reserved for future use. |
| `l1_threshold` | `0.5` | Lexical-match threshold for the first-pass tool selector. Lower means more tools considered. |
| `fallback_include_all` | `true` | If selection fails, forward all tools rather than dropping the request. |
| `max_tools_injected` | `10` | Hard cap on tools surfaced back to the model per turn. |

### `learning`

Adaptive tuning loop.

```yaml
learning:
  tune_interval: 50
  relevance_threshold: 0.7
  core_facts_size: 30
```

| Key | Default | Notes |
|-----|---------|-------|
| `tune_interval` | `50` | Turns between re-tuning passes. |
| `relevance_threshold` | `0.7` | Minimum relevance for a retrieved fact to stay in the lean payload. |
| `core_facts_size` | `30` | Size of the core-facts pool maintained by the learning loop. Mirrors `pipeline.core_facts_size`. |

### `security`

Proxy-level access control.

```yaml
security:
  auth_token: null
  allowed_origins: ["127.0.0.1"]
```

| Key | Default | Notes |
|-----|---------|-------|
| `auth_token` | `null` | When set, clients must present it as a bearer token. Leave `null` for single-user local setups. Evaluation runs require `null`. |
| `allowed_origins` | `["127.0.0.1"]` | CORS allow-list. |

If you expose Sieve beyond loopback, set `auth_token` to a random secret and change `listen.host` to `0.0.0.0` — in that order.

### `ablation`

Per-subsystem on/off switches. Exposed so you can reproduce ablation measurements and diagnose regressions. **The shipping defaults are what was evaluated** — only change these if you are actively running an experiment.

```yaml
ablation:
  fingerprinting: true
  classifier: true
  pre_populate: true
  graph_traversal: true
  temporal_versioning: true
  learning_loop: true
  coherence_integrity: true
  stage2_writer: true
  recall_tool: true
  absence_signal: true
  closed_world: false
  response_verification: false
  schema_v2: false
  tier2_classifier: false
  extreme_summary: true
```

The most consequential flags:

- **`absence_signal`** (on). Refuses to fabricate when a recall query targets a fact not in the store. Responsible for the hallucination-reduction numbers.
- **`stage2_writer`** (on). Runs the fact extractor after each turn. Without this the store never grows.
- **`extreme_summary`** (on). Narrative summariser for long conversations.
- **`closed_world`** (off). An earlier, stricter absence posture. Permanently off — superseded by `absence_signal`.
- **`response_verification`** (off). Pattern-based output check. Disabled pending pattern coverage.

## Common setups

### Local Ollama (default)

The `sieve init` defaults cover this case. No edits required.

### Any OpenAI-compatible endpoint

Change `provider.base_url` to the endpoint and `provider.default_model` to the model name:

```yaml
provider:
  type: auto
  base_url: https://your-openai-compatible-host/v1
  default_model: your-model-name
```

The specific URL, model name, and authentication method depend on your endpoint. Authentication headers (`Authorization: Bearer ...`, etc.) are set on the agent-side client and forwarded by Sieve; Sieve itself has no provider credentials.

### Exposing Sieve to another machine

```yaml
listen:
  host: 0.0.0.0
  port: 11435

security:
  auth_token: "<long-random-string>"
  allowed_origins: ["10.0.0.0/8"]
```

Management endpoints under `/sieve/*` now require an `X-Sieve-Token: <your-token>` header. Proxy pass-through endpoints (`/api/*`, `/v1/*`) remain unauthenticated so your agent does not need Sieve-specific credentials — rely on network-level restrictions (loopback, LAN segmentation, a reverse proxy with TLS) for those. Do not expose Sieve directly to the public internet.

## Applying changes

`sieve start` reads the config at startup. To apply a change:

```bash
sieve stop
sieve start
```

There is no live reload. For interactive tuning, edit the file, restart, and watch `sieve status` and the per-response `X-Sieve-*` headers.

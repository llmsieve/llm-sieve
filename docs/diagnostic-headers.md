---
description: >
  The diagnostic response headers Sieve attaches to intercepted chat
  requests — stable across releases, safe to build scripts and
  monitoring on.
---

# Diagnostic response headers

Sieve attaches a small set of response headers on intercepted `/api/chat`
and `/v1/chat/completions` requests for operational visibility. These
are stable across releases — you can write scripts and monitoring that
depends on them.

| Header | Meaning |
|---|---|
| `X-Sieve-Phase` | Progressive-activation phase (`OBSERVE` / `ACCUMULATE` / `ACTIVATE`) |
| `X-Sieve-Fact-Count` | Facts in the memory store at request time |
| `X-Sieve-Inbound-Tokens` | Approximate token count of the inbound payload before Sieve's trim |
| `X-Sieve-Outbound-Tokens` | Approximate token count sent upstream after trim |
| `X-Sieve-Rounds` | Number of recall-tool iterations (0 = no tool call) |
| `X-Sieve-Proxy-Us` | Sieve-side wall time in microseconds (excludes upstream LLM time) |

## When to read them

- **Sanity-checking that Sieve is actually trimming**: `X-Sieve-Inbound-Tokens` vs `X-Sieve-Outbound-Tokens` shows the reduction per request.
- **Progressive-activation introspection**: `X-Sieve-Phase` tells you whether Sieve is still in cold-start (OBSERVE/ACCUMULATE) or fully active (ACTIVATE).
- **Latency attribution**: `X-Sieve-Proxy-Us` measures Sieve's own overhead so you can separate it from upstream model time.

## When to enable deeper telemetry

If you need per-request metrics (retrieval precision, writer extraction
counts, tool-call breakdowns), enable the built-in validation collector
in `sieve.yaml`:

```yaml
validation:
  enabled: true
  db_path: ~/.sieve/validation_metrics.db
```

One SQLite row is written per intercepted request. Default is off — turn
it on only when you want the data, since it adds disk writes to the hot
path.

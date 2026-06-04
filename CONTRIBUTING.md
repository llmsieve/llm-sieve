# Contributing to Sieve

Thanks for your interest in contributing. This project is in early public
release — issues, bug reports, and pull requests are all welcome.

## Before you start

- Read the [README](README.md) and try running Sieve end to end.
- Browse the [issue tracker](https://github.com/llmsieve/llm-sieve/issues)
  to see what's already being discussed.
- For anything non-trivial, open an issue first so we can discuss the
  approach before you invest implementation time.

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the [Developer Certificate of
Origin 1.1](https://developercertificate.org/). By signing off your commits
you certify that you wrote the code (or otherwise have the right to
contribute it) and that the project may redistribute it under the project's
licence.

Sign off commits with the `-s` flag:

```bash
git commit -s -m "fix: correct classifier threshold"
```

Git appends a `Signed-off-by:` trailer using your `user.name` and
`user.email`. That line is the sign-off.

## Development setup

```bash
git clone https://github.com/llmsieve/llm-sieve.git
cd llm-sieve
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### Building the docs site locally

The documentation site at [llmsieve.dev](https://llmsieve.dev) is built
from `docs/` with MkDocs. To work on it locally, install the docs
optional-dependency group as well:

```bash
pip install -e ".[dev,docs]"
mkdocs serve          # live-reload preview at http://127.0.0.1:8000
mkdocs build --strict # fail on any warning (orphan pages, broken links, etc.)
```

`--strict` is the same flag CI uses, so if it builds clean locally the
docs deploy will pass.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes with tests. Keep commits focused and explain the
   *why* in the commit message.
3. Run `pytest` locally — every test must pass.
4. Run `ruff check .` — no lint errors.
5. Sign off each commit (`-s`).
6. Open a pull request against `main`. Describe the change, the motivation,
   and any tradeoffs.

## Code style

- Python 3.11+
- `ruff` for formatting and lint (config in `pyproject.toml`)
- Type hints on public APIs
- Tests alongside new code

## Versioning and release policy

Sieve follows [Semantic Versioning](https://semver.org/). The
contract Sieve makes with its users — and that contributors should
keep in mind when proposing changes — is:

### PATCH releases (1.0.x)

- Bug fixes
- Documentation improvements
- Performance wins that don't change observable behaviour
- Forward-and-backward-compatible store-schema migrations
  (e.g. an added column with a safe default)

**User expectation:** safe to upgrade unconditionally. No backup
required. Released as fixes accumulate; no schedule.

### MINOR releases (1.x.0)

- New optional features
- New config keys (with safe defaults — existing configs must keep
  working)
- Additional CLI subcommands
- Forward-only store-schema migrations that the writer can apply
  automatically on first run

**User expectation:** safe to upgrade with `pipx upgrade`. A backup
is recommended for users with significant accumulated data.
Released when meaningful features accumulate.

### MAJOR releases (X.0.0 → (X+1).0.0)

- Behaviour changes
- Dropped CLI subcommands
- Backward-incompatible store-schema migrations
- Config-key renames or removals
- Raised minimum Python version
- Dropped LLM-provider support

**User expectation:** read the CHANGELOG migration notes; back up
first; test in a non-production context. Released at most once a
year. Pre-announce by ≥4 weeks. Publish a migration guide
alongside the release.

### Security releases

Anything with CVE-class implications ships within **7 days** of a
confirmed report, regardless of the standard cadence above.
Communicated via:

- A GitHub Security Advisory
- A patch-level release with a `SECURITY:` prefix in the CHANGELOG
- A pinned issue for the duration of the affected versions

The private reporting channel is documented in
[SECURITY.md](SECURITY.md).

### When proposing a breaking change

A change is breaking if it:

- Removes or renames a CLI command, subcommand, or flag
- Removes or renames a config key (or makes a previously-optional
  key required)
- Changes the store-schema in a way that prior versions can't read
- Raises the minimum Python version
- Drops support for an LLM provider that previously worked

If your PR contains a breaking change, please:

1. Flag it in the PR description with a `**Breaking change**` line
   and a one-paragraph explanation of why the breakage is
   necessary.
2. Propose a deprecation window if a compatibility shim is
   feasible — usually one MINOR release with a `DeprecationWarning`
   before removal in the next MAJOR.
3. Draft the CHANGELOG entry under an explicit `### Breaking`
   subsection.

Breaking changes don't have to be rejected — they just have to be
deliberate.

### Schema versioning

The encrypted store carries a `PRAGMA user_version` integer that's
written at init time and bumped by any future schema-migration
code (`sieve.store.SCHEMA_VERSION`). If you're proposing a schema
change:

- **Backward-compatible additions** (new column with a safe
  default, new index, new optional table) → no version bump
  needed; the existing migrator handles them via idempotent
  `ALTER TABLE`.
- **Required schema changes** (a new column the proxy assumes
  exists at startup, a renamed column, a table whose shape
  changed) → bump `SCHEMA_VERSION` and add the migration to
  `init_schema()` keyed off the read `user_version`. Stores on
  older versions get migrated forward on first open; stores on
  newer versions trigger `StoreSchemaTooNewError` and refuse to
  open (the rollback-safety guard).

The guard's behaviour and the recovery path for users are
documented in [installation.md → Rolling back](docs/installation.md#rolling-back).

## Reporting bugs

Include:

- Sieve version (`sieve --version`)
- Python version, OS
- Minimal reproducer
- Expected vs actual behaviour
- Relevant logs (with any secrets redacted)

## Security

Do **not** open public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for how to report them.

## Code of conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
Participation requires agreeing to it.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0.

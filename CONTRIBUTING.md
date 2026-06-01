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

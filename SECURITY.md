# Security policy

## Reporting a vulnerability

**Please do not open public issues for security vulnerabilities.** If you
believe you have found a security problem in Sieve, report it privately.

### Preferred: GitHub private advisory

Use GitHub's private vulnerability reporting:

> <https://github.com/llmsieve/llm-sieve/security/advisories/new>

This sends a private report directly to the maintainers. Only people you
invite can see the advisory while it is being worked on.

### Alternative: email

If you cannot use the GitHub flow, email the maintainer:

> azard.hosein@gmail.com

PGP keys are not currently published. If you need encrypted communication,
say so in your first message and we will exchange keys before any sensitive
detail is shared.

## What to include

Where possible, please include:

- A description of the issue and its impact
- Steps to reproduce, ideally a minimal proof of concept
- Affected versions (`sieve --version`)
- Whether the issue has been disclosed elsewhere

## What to expect

- We aim to acknowledge new reports within **72 hours**.
- We will keep you updated as we triage and develop a fix.
- Once a fix is ready we will coordinate disclosure timing with you and
  credit you in the release notes if you wish.

## Scope

In scope:

- The `llm-sieve` Python package and its CLI (`sieve`, `sieve-install`)
- The encrypted SQLCipher memory store
- The HTTP proxy server and its endpoints

Out of scope:

- Vulnerabilities in upstream LLM providers (report to that vendor)
- Vulnerabilities in dependencies (Ollama, FastEmbed, sqlcipher3, etc.)
  unless we expose them in a meaningful new way
- Issues that require the attacker to already have local execution on
  the user's machine (Sieve trusts the local host)

## Code of conduct concerns

The same private channels above are used for [Code of
Conduct](CODE_OF_CONDUCT.md) reports.

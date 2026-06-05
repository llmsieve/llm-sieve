<!--
Thanks for contributing to Sieve. A few quick checks before the
review can start.
-->

## What this PR does

<!-- One paragraph. The "why" matters more than the "what" — the
diff already shows what changed. -->

## Test plan

<!-- How did you verify this works? Concrete commands or test names
are better than "tested it works." -->

- [ ] `pytest` passes locally
- [ ] `ruff check .` clean
- [ ] If touching docs: `mkdocs build --strict` clean

## Breaking change?

<!-- Tick one. See CONTRIBUTING.md → Versioning policy for what counts
as breaking. -->

- [ ] No — purely additive or internal refactor
- [ ] No — changes default behaviour but existing configs still work
- [ ] Yes — and the CHANGELOG entry is under `### Breaking`

## Checklist

- [ ] Commits are signed off (`git commit -s`) per DCO requirement in CONTRIBUTING.md
- [ ] CHANGELOG.md updated if user-visible
- [ ] Linked to the issue this PR closes (if any), via `Closes #N`

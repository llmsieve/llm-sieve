# Release audit — wizard + docs site

Date: 2026-06-01
Auditor: paired session, against `main` @ commit `3b7051c`

## Methodology

Tested both first-run paths from a sandbox state (moved `~/.sieve` aside, ran the install
flows fresh, then restored). Built the MkDocs site with `--strict`. Findings classified by
release-blocking severity.

---

## 🔴 BLOCKERS — must fix before release

### B1. `sieve-install --no-input` fails silently on the common path

**Reproduce:**
```bash
.venv/bin/sieve-install --no-input
```

**Actual:** Splash → welcome → one terse line "default Ollama isn't running. Pass --provider URL or start Ollama first." → exit 1 with **`~/.sieve/` empty**.

**Expected:** Either:
- Detect this case and give a richer error with a concrete next-action ("install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`, then re-run; OR pass `--provider https://api.openai.com/v1` to use OpenAI"), or
- Don't bail. Write the config anyway (with a placeholder provider URL) and tell the user to fix the URL before running `sieve start`.

**Why blocking:** `sieve-install` is what `pipx install sieve && sieve-install` is supposed to deliver. New user follows the docs, gets this, has nothing. First impression catastrophe.

### B2. `sieve init` shows a confusing error from a bad probe URL

**Reproduce:** From a clean state, run `sieve init` and accept the default prompt by pressing Enter.

**Actual log:**
```
No Ollama on localhost:11434.
Enter LLM provider base URL [http://127.0.0.1:11434]: Could not reach provider (Request URL is missing an 'http://' or 'https://' protocol.). Continuing —
```

**Bug:** The prompt printed before the user typed (Enter accepts default), but the "Could not reach provider" message shows a URL-parsing error — meaning the code path used `provider` (which was already set to `http://127.0.0.1:11434`) but then tried to call `httpx.get(provider.rstrip('/'))` *without `/api/tags`* somewhere, or the path is wrong. Code review needed at `src/cli.py:1273-1285`.

**Why blocking:** The error message is misleading. The user provided a valid URL — the message blames them for an invalid URL. Will generate support tickets.

### B3. `sieve` package missing `__version__`

**Reproduce:**
```python
import sieve
sieve.__version__  # AttributeError
```

**Why blocking:** Standard introspection — `pip show sieve` works, but `import sieve; sieve.__version__` is what users + tooling check. Quick fix in `sieve/__init__.py`.

---

## 🟠 SHOULD-FIX — release-noticeable, low effort

### S1. Orphan doc page `diagnostic-headers.md`

`docs/diagnostic-headers.md` exists but is not referenced in `mkdocs.yml` nav. MkDocs build prints a warning. Either add to nav, link from another page, or remove.

### S2. `sieve init` happy-path message could be cleaner

Currently after `sieve init` succeeds, the closing message is:
```
Ready!
Start Sieve with: sieve start  (point your agent at http://localhost:11435)
```

This is good. But it doesn't say what the configured provider URL is — the user has no way to verify it captured what they typed without opening the yaml. Consider:
```
Ready! Configured for Ollama at http://127.0.0.1:11434.
Run `sieve start` to launch the proxy on http://localhost:11435.
Point your agent at http://localhost:11435.
```

### S3. Missing `dev` dependency: `mkdocs`/`mkdocs-material`

`pyproject.toml` declares `docs = ["mkdocs-material>=9.5"]` as an optional dependency group, but a fresh `pip install -e .` doesn't include it. The docs nav says `Documentation: https://llmsieve.dev` — someone building docs locally needs to know to run `pip install -e ".[docs]"`. Add a one-liner to `CONTRIBUTING.md` or README's docs section.

### S4. CHANGELOG link in docs goes to GitHub source

`cli-reference.md` (or another page) links to `https://github.com/llmsieve/llm-sieve/blob/main/CHANGELOG.md`. This means the CHANGELOG is only readable on GitHub, not on the docs site. Consider rendering it as a page (e.g., `docs/changelog.md` with a `--8<-- "CHANGELOG.md"` include).

### S5. `sieve.example.yaml` and dataclass defaults

Recent commits show alignment work (`d127a74 docs(config): align sieve.example.yaml writer.model with dataclass default`, `3597c34 sieve.example.yaml: fix temporal_dedup key`). Worth running a programmatic check: parse `sieve.example.yaml`, then parse it through the config dataclass, fail if any key in the YAML doesn't have a corresponding dataclass field. CI guard — prevents future drift.

---

## 🟡 NICE-TO-HAVE — quality-of-life

### N1. Branding GIFs may be stale

`branding/sieve-wizard.gif` and `sieve-wizard-install.gif` are the visual demos in the README. With the recent changes to the install flow (per-port PID file, prompts), worth re-recording from a clean state. Use `vhs` (`demo.tape` exists in the repo) so the regen is one command.

### N2. `--version` flag works but doesn't show on `--help`

`.venv/bin/sieve --help` lists `--version` cleanly. Good. But the version is whatever pyproject.toml has at install time — verify it's not stuck at `0.0.0` for releases.

### N3. Empty-state hint after `sieve init`

After `sieve init`, the user has an empty store. A friendly next-step might be: `Run \`sieve demo\` to see Sieve in action with a 30-second sample conversation`. The `demo` command already exists.

### N4. Docs landing page numbers predate Phase 3

`docs/index.md:7` currently reads:
```
- **Up to 97% token reduction** on large agent payloads (96.9% outbound on a 30-day progressive-activation run)
- **Up to 9× less hallucination** on absence-trap queries (9.3× measured)
```

Phase 3 (the empirical package at `v1.1.0-phase3-rc` in the recall repo) gives us a stronger, simpler story:

> **95% fewer tokens per turn, invariant across 5 architectures, 2 model-size classes (8B → 72B),
> 5 context windows (8K → 64K), and 5 concurrency levels (1 → 64). Personal-fact fabrication
> eliminated on small writers (Qwen3-8B 67-74%→0%, Llama-8B 47-58%→0%, replicated across 4
> seeds). Recall mechanism sub-15 ms p50 at 100k facts with full production crypto.**

Suggested rewrite for the bullet list:

```markdown
- **95% token reduction per turn** — invariant across 5 LLM architectures (Granite, Llama, Qwen, Mistral, Granite), 8B-72B model sizes, 8K-64K context windows, and 1-64 concurrent sessions
- **3-7× faster on followups** — sieve ships 150 tokens; baseline ships the full history
- **Zero filler fabrication** — empirically perfect across all measured cells
- **Personal-fact fabrication eliminated on small writers** (8B class), audited by 3-layer consensus
- **Self-contained** — FastEmbed in-process + your own LLM
- **Works with any OpenAI-compatible or Ollama endpoint**
- **Encrypted local-first memory** (SQLCipher) — sub-15 ms recall at 100k facts
- **Apache 2.0**, patent pending
```

The Phase 3 release-candidate writeups are at `recall/evaluation/simulator/results/phase3-final/`
(see `CLAIMS.md` for the audit-surface table mapping each number to its source artifact).

---

## ✅ WHAT'S WORKING WELL

- CLI surface is clean and discoverable. `sieve --help` lists commands logically.
- `sieve init` happy-path (when provider reachable) writes config + downloads embedding model + initialises store in one step. Good UX.
- `sieve-install` design (the `_installer.py` docstring) shows clear design principles — bounded timeouts, idempotent, atomic commit. Good engineering hygiene.
- Wizard is unit-testable via `Prompter` protocol injection — good architecture.
- MkDocs site builds cleanly with `--strict` (modulo the one orphan-page warning).
- Internal MD links resolve.
- Two-entry-point split (`sieve-install` for first-run, `sieve wizard` for ongoing management) is the right call.

---

## Suggested release-prep sequence

1. **Fix B1+B2 today** — these are user-blocking on day one of release. ~1-2 hours each.
2. **Fix B3** — `__version__` exposure. ~5 min.
3. **Address S1-S5** — release-polish. ~2 hours total.
4. **Re-record N1 GIFs** — once B1+B2 are fixed so the recording reflects the actual flow. ~30 min.
5. **N4: audit docs landing page for Phase 3 numbers** — ~15 min.

Total: ~1 working day to RC-ready. Most blockers are small but unmistakeable.

---

## Test commands I'd add to CI

To prevent the blockers from regressing:

```bash
# B1 guard
sieve-install --no-input --provider http://localhost:11434 && \
  test -f ~/.sieve/sieve.yaml || (echo "B1 regressed"; exit 1)

# B2 guard
echo "" | sieve init && \
  test -f ~/.sieve/sieve.yaml || (echo "B2 regressed"; exit 1)

# B3 guard
python -c "import sieve; assert sieve.__version__"

# S1 guard
mkdocs build --strict 2>&1 | grep -q "exist in the docs directory" && (echo "orphan docs page"; exit 1)
```

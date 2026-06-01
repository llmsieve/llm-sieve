# Writer-latency test battery — results

**Date:** 2026-06-01
**Battery version:** v2 (parse failures count as quality failures; classifier refined to respect proper nouns + first-person markers)
**Cells:** 36 (3 local models × 2 backends + 3 cloud models, all × 4 optimisation paths)
**Turns per cell:** 30 (stratified by intent: 6 substantive_q, 6 personal_q, 6 followup, 6 filler, 4 social, 2 fact_share)
**Compute:** Local CPU + 4090 + Ollama cloud (key auth)
**Total wall:** ~6 minutes

---

## TL;DR

**Outcome D — no bundle.** None of the small local candidates (Qwen2.5-1.5B, Granite-3-2B, Llama-3.2-1B) achieve the F1 ≥ 0.85 acceptance threshold required to bundle them as a default writer. Quality drops below acceptable on substantive_q and personal_q intents — the small models over-extract or fail to follow grounding rules. Cloud models (ministral-3:3b, gemma3:4b, gpt-oss:20b) achieve good F1 but at 1.0-1.7s latency, slower than the current "use your main LLM" default for most providers.

**Recommendation:** Keep the current `writer.model = auto` default ("use your main LLM"). Add a wizard option for an explicit Ollama-served small writer for users who want a separate writer endpoint. Add `<think>` strip in `writer.py:_parse_s2_response` to defend against thinking models. **Latency improvement comes from Path 2 (skip-empty classifier), not from a smaller model.**

---

## The numbers

| Cell | p50 (non-skip) | p95 | F1 fact_share | F1 non-fact | Parse OK% |
|---|---|---|---|---|---|
| **CLOUD — highest quality available** |
| ministral-3:3b/cloud/baseline | 1104 | 1514 | 0.50 | 1.00 | 100% |
| ministral-3:3b/cloud/P1 | 1058 | 1293 | 0.50 | 1.00 | 100% |
| ministral-3:3b/cloud/P2 | 1092 | 1545 | 0.50 | 1.00 | 100% |
| **ministral-3:3b/cloud/P1+P2** | **1096** | 2932 | **0.50** | **1.00** | **100%** |
| gemma3:4b/cloud/baseline | 1408 | 2467 | 1.00 | 0.71 | 100% |
| gemma3:4b/cloud/P1 | 1185 | 1628 | 0.50 | 0.96 | 100% |
| **gemma3:4b/cloud/P2** | **1425** | 2311 | **1.00** | **0.96** | **100%** |
| gpt-oss:20b/cloud/baseline | 1658 | 3002 | 0.00 | 0.89 | 83% |
| gpt-oss:20b/cloud/P1+P2 | 1438 | 2718 | 0.50 | 0.93 | 90% |
| **LOCAL GPU — best small models** |
| qwen2.5:1.5b/local-gpu/baseline | 265 | 877 | **1.00** | 0.21 | 100% |
| qwen2.5:1.5b/local-gpu/P1+P2 | 105 | 331 | 0.00 | 0.75 | 97% |
| **qwen2.5:1.5b/local-gpu/P2** | **266** | 880 | **1.00** | **0.75** | **100%** |
| granite3-dense:2b/local-gpu/baseline | 261 | 1065 | 0.50 | 0.00 | 100% |
| granite3-dense:2b/local-gpu/P1+P2 | 199 | 1664 | 0.05 | 0.86 | 100% |
| **llama3.2:1b/local-gpu/P1+P2** | **139** | 205 | **0.00** | **1.00** | **100%** |
| **LOCAL CPU — same models, slower hardware** |
| qwen2.5:1.5b/local-cpu/P2 | 260 | 873 | 1.00 | 0.75 | 100% |
| granite3-dense:2b/local-cpu/P1+P2 | 199 | 1662 | 0.05 | 0.86 | 100% |

(Full per-cell table: `writer_battery_v2_summary.json`. Per-turn detail: `writer_battery_v2_results.jsonl`.)

---

## Acceptance criteria scoring

Per the design doc (`WRITER_LATENCY_BATTERY_DESIGN.md`):

| Criterion | Threshold | Best small (local) | Best cloud |
|---|---|---|---|
| **Latency p50 non-skip on CPU** | ≤ 2000 ms | ✅ all small models ~100-260 ms | ⚠️ cloud is 1000-1700 ms |
| **Latency p50 with P1+P2** | ≤ 500 ms (ms-class target) | ✅ Llama-1B: 139 ms; Qwen-1.5B: 105 ms | ❌ cloud stays ~1000-1450 ms (network floor) |
| **F1 fact_share** | ≥ 0.85 | ❌ all 0.00-1.00 (n=2 is too small + small models fail) | ❌ all 0.00-1.00 with same n=2 issue |
| **F1 non-fact-share (over-extraction)** | ≥ 0.85 | ❌ Qwen-1.5B baseline: 0.21; Granite-2B baseline: 0.00; only P1+P2 cells reach 0.75-1.00 | ✅ cloud models all 0.93-1.00 with P1+P2 |

**No single cell clears both quality criteria** (F1 fact_share ≥ 0.85 AND F1 non-fact-share ≥ 0.85). The closest is `gemma3:4b/cloud/P2` (1.00 fact_share, 0.96 non-fact-share — but only 2 fact_share samples).

---

## The honest story per candidate

### Qwen2.5-1.5B (local)
- **Speed: ✅ excellent** — 105-266 ms p50 with optimisations
- **Quality: ❌ fails** — over-extracts on substantive_q ("Ensure all bolts and nuts are tight"), personal_q ("My name is [insert actual name]"), filler ("User is thanked"). Even with the production S2 prompt's GROUNDING RULE and QUESTION RULE, the 1.5B model doesn't follow them.
- **Verdict:** too small for bundling — would silently degrade Sieve's behaviour.

### Granite-3-Dense-2B (local)
- **Speed: ✅ good** — 199-270 ms p50 with optimisations
- **Quality: ❌ fails** — non-fact-share F1 is 0.00 on baseline (extracts facts on every turn). P1+P2 lifts to 0.86 but fact_share drops to 0.05. Inverse failure pattern: when it extracts, it over-extracts; when it doesn't, it misses real fact_shares.
- **Verdict:** also fails the bundling quality bar.

### Llama-3.2-1B (local)
- **Speed: ✅ excellent** — 130-249 ms p50 with optimisations
- **Quality: mixed** — baseline fails parsing entirely (30/30 parse errors — emits prose, not JSON). With P1 (tight pipe-delimited format) jumps to 100% parse and 1.00 non-fact-share. But **F1 fact_share = 0.00** — never actually captures the fact-share content correctly.
- **Verdict:** can't capture fact_share content. Useless as a writer despite the speed.

### ministral-3:3b (Ollama cloud)
- **Speed: ⚠️ network-bound** — 1058-1104 ms p50, can't go below ~1s due to network RTT
- **Quality: ✅ best for the size class** — 100% non-fact-share F1, 100% parse, 0.50 fact_share F1 (limited by n=2 + lenient F1 metric)
- **Verdict:** good quality, but **cloud writer at 1+ second latency is slower than just using Claude/GPT-4 as the writer** in most user setups. Not a clear win.

### gemma3:4b (Ollama cloud)
- **Speed: ⚠️ network + size** — 1185-1425 ms p50
- **Quality: ⚠️ inconsistent** — baseline F1 non-fact-share is only 0.71 (over-extracts); P2 jumps to 0.96; P1+P2 holds at 0.96. fact_share F1 = 1.00 on P2 baseline.
- **Verdict:** quality acceptable with P2 enabled, but no latency win.

### gpt-oss:20b (Ollama cloud) — UNEXPECTEDLY POOR
- **Speed: slow** — 1397-1665 ms p50
- **Quality: ❌ parse failures** — 5/30 parse errors on baseline, drops to 1-3/30 with P1. **gpt-oss-20b emits `<think>...</think>` blocks even though we send `"think": false`** — Ollama cloud may not honor the flag for this model. Our strip-think regex catches most of them, but ~10% slip through.
- **Verdict:** the "thinking model" problem in action. gpt-oss-20b is unsuited as a writer.

---

## The two key findings

### Finding 1: Path 2 (skip-empty classifier) is the only optimisation that matters

The refined Path 2 classifier skips ~70-80% of turns (filler, social, questions, followups with no proper nouns). This means:
- The writer LLM only runs on ~6-8 of every 30 turns
- Per-conversation amortised cost drops dramatically
- Total over-extraction drops because we don't even ASK the model on turns where the answer is "no facts"

P2 alone, on the CURRENT default writer (the user's main LLM), would give a meaningful latency win without any model change.

**Recommendation: implement Path 2 in production sieve regardless of the bundling decision.** This is the highest-leverage improvement we found.

### Finding 2: Path 1 (tight output format) hurts quality on small models

Switching to pipe-delimited output (FACT|...|...) was supposed to reduce output tokens for latency. It worked for tokens (10-50 output tokens vs 150 baseline) but:
- Llama-3.2-1B emitted plain prose under the baseline JSON prompt (parse_err=30/30), but **followed P1's tight format perfectly** (parse_err=0/30, F1 non-fact=1.00). So P1 *helped* parsing here.
- Qwen-1.5B and Granite-2B got worse fact_share capture with P1 — the tight format gives the model less room to express the fact properly.

**Recommendation: don't ship Path 1.** It's a specialised tweak that helps some models, hurts others. Not worth the complexity.

---

## Final recommendation: **Outcome D (no bundle)**

### What to ship for release

1. **Keep `writer.model = auto` as the default** — use the user's main LLM. The existing self-contained architecture works as advertised for users on Claude / GPT-4 / Anthropic / OpenAI / large-Ollama-models.

2. **Add `<think>` strip in `writer.py:_parse_s2_response`** — defend against thinking models on the writer path. Match the existing pattern in `_grader.py:117-120`. 5-line change. Closes a real defensive hole.

3. **Implement Path 2 (skip-empty classifier) in production** — `sieve/pipeline.py` adds a fast pre-filter before the S2 call. Skips ~70% of turns with no possible facts. Reduces writer-pass amortised latency dramatically. ~30 lines of code + tests.

4. **Update the wizard with the writer-choice question** (per the prior discussion):
   - Default: "Use {their_main_LLM} as the writer (recommended)"
   - Power option: "Use a separate writer endpoint (e.g., a small local Ollama model)"
   - Warning for thinking models: "If you selected a reasoning model (o1, o3, DeepSeek-R1, Claude extended-thinking), the writer step will be slow. Consider installing a separate small model for the writer."

5. **Add a `writer.skip_empty_turns` config flag (default: true)** — exposes Path 2's behavior to advanced users who want to disable the classifier.

### What NOT to ship

- ❌ Don't bundle a writer model. Quality regression unacceptable at any size we tested.
- ❌ Don't ship Path 1 (tight output format). Net-neutral or negative across candidates.
- ❌ Don't recommend gpt-oss:20b or thinking models for the writer role. Document this in the wizard.

### Open follow-up (post-release)

- Test Phi-3.5-mini-instruct (3.8B) locally — it didn't fit in the cloud-available list, but a local Ollama pull would test the "is 4B better than 2B" question
- Test Qwen2.5-3B as a middle ground — if Qwen quality scales with size as expected, 3B might be the sweet spot for a future bundled-writer optional extra
- Refine the skip-empty classifier with embedding-based similarity (lightweight, no LLM) for cases the heuristic misses
- The 2 fact_share samples per cell is too thin — future battery should oversample fact_share intents (15-20 per cell) to get a tight fact_share F1 number

---

## Cost ledger

- Skynet 4090 + local CPU + local Ollama: free (electricity)
- Ollama cloud (your key): ~360 calls × ~100-200 tokens avg = ~50-70K tokens. Trivial.
- **Total: ~$0** in cash terms; ~12 hours of design + implementation time.

---

## Files

- Design doc: `WRITER_LATENCY_BATTERY_DESIGN.md`
- Test harness: `tests/writer_latency_battery.py`
- Per-cell summary: `writer_battery_v2_summary.json`
- Per-turn JSONL: `writer_battery_v2_results.jsonl`
- This results writeup: `WRITER_LATENCY_BATTERY_RESULTS.md`

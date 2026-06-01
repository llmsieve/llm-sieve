# Writer-latency test battery — design doc

**Goal:** Empirically answer "what does each candidate bundled-writer model cost in latency, and how do the proposed optimisations move that number?"

**Output:** runnable test script + measured latency curves across models × quantisations × hardware × optimisation paths. Decision-ready data for choosing between Path A/B/C (bundle vs opt-in) for release.

**Decision the data unblocks:**
- Should we bundle a writer? (i.e. is sub-1s achievable on commodity laptop CPUs?)
- Which model? (Qwen2.5-1.5B / Granite-2B / Phi-3.5-mini / Llama-3.2-1B)
- Are Path 1 (output-tightening) and Path 2 (skip-empty-turn) worth the integration effort?

---

## Test methodology

### What we measure

For each `(model, quant, hardware, optimisation)` cell, capture per-call:

1. **`prefill_ms`** — time to first output token after sending request (proxies prompt-ingest cost)
2. **`decode_ms`** — total response time minus prefill (proxies output-generation cost)
3. **`total_ms`** — `prefill_ms + decode_ms`
4. **`input_tokens`** — measured, not estimated
5. **`output_tokens`** — measured, not estimated
6. **`json_parse_ok`** — does the writer's output parse cleanly as the S2 Pydantic schema?
7. **`fact_count`** — number of facts extracted (sanity check vs ground truth)
8. **`fact_quality_score`** — agreement with reference extraction (see §quality below)

Run **30 turns per cell** (statistical body for p50 / p95). Wall budget per cell: ~5 minutes.

### Quality metric (the load-bearing one)

Without quality matching, latency wins are meaningless. We measure quality as:

- **Reference extraction:** the Phase 3 captures already contain `shared_facts` arrays that were ground truth at simulator-time. We can replay the writer pass against the same `rendered_messages` and compare candidate-writer's `facts` array against ground-truth `shared_facts`.
- **F1 score** on fact content (lenient string match — paraphrases OK, semantically-equivalent OK)
- **Recall** on fact entities (named entities mentioned in the source should appear in the extraction)

Acceptance threshold (provisional):
- F1 ≥ 0.85 vs reference: **acceptable** as bundled writer
- F1 ≥ 0.70: **marginal** — bundled writer is OK as a "privacy-first" mode, not as the default
- F1 < 0.70: **unacceptable** — would silently degrade Sieve's behaviour

---

## The test grid

### Models (4 candidates)

| Model | Params | Q4 size | Why this one |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 1.5B | ~900 MB | Apache 2.0, strong JSON instruction following, smallest competitive structured-output model |
| **Granite-3.0-2B-Instruct** | 2B | ~1.2 GB | Apache 2.0, IBM-tuned for enterprise/structured tasks |
| **Phi-3.5-mini-instruct** | 3.8B | ~2.2 GB | Strongest small model on instruction-following benchmarks; tests "is bigger needed?" |
| **Llama-3.2-1B-Instruct** | 1B | ~700 MB | Smallest Llama; tests "is smaller acceptable?" |

Plus **reference baseline:** the current default ("auto = user's main LLM") using whatever's at `recall.yaml` provider. We use Llama-3.1-8B locally for the test runs as the "frontier proxy."

### Quantisations (3 levels per model)

| Level | When | Size multiplier |
|---|---|---|
| **Q4_K_M** | Default candidate | ~0.6× FP16 |
| **Q5_K_M** | Quality-sensitive users | ~0.75× FP16 |
| **FP16** | GPU users with VRAM to spare; quality ceiling | 1.0× |

### Hardware (3 targets — matches Sieve's deployment universe)

| Target | Backend | Realistic for |
|---|---|---|
| **Skynet CPU** (modern AMD/Intel, AVX2) | llama-cpp-python CPU | "I'm on a developer laptop" |
| **Skynet 4090** | llama-cpp-python CUDA (or vLLM if available) | "I have a GPU" |
| **Cloud API proxy** (Groq llama-3-8B, or vLLM-hosted Qwen) | HTTP API to vLLM | "I have a paid endpoint" |

Skipping ARM Mac for now — same logic as Skynet CPU; can extrapolate. Could add later.

### Optimisation paths (4 conditions per cell)

| Condition | What it tests |
|---|---|
| **Baseline** | Current sieve writer prompt, full JSON output (~150 output tokens) |
| **+P1 (tight output)** | Replace JSON schema with compact format (`key=value` pipes, ~30-50 output tokens) |
| **+P2 (skip empty)** | Add a fast classifier (token-count heuristic + spaCy or embedding similarity) to skip the writer call entirely on turns predicted to have no facts |
| **+P1+P2 combined** | Best case |

**Note on P2:** the classifier itself adds <5ms latency. Its value is in *avoiding* the writer call on the ~40-60% of turns predicted to have no facts. So the per-cell measurement is two numbers: latency-when-classifier-says-call (full writer cost) and overall-amortised-latency-across-30-turns (where classifier skips some).

### Total cells

4 models × 3 quants × 3 hardware × 4 optimisations = **144 cells** × 30 turns × ~1s each = **~3-4 hours of measurement**.

Reducible if needed by dropping:
- Q5_K_M on all candidates (just measure Q4 + FP16 — 96 cells)
- Phi-3.5-mini if early Qwen-1.5B/Granite-2B results clearly cover the size band (108 cells)
- Cloud API target if early local results don't motivate the comparison (96 cells)

**Realistic minimum: 48 cells × 30 turns = ~1 hour** — Qwen + Granite + Llama-1B × Q4 + FP16 × CPU + 4090 × 4 opts.

---

## Fixtures

### Input data — 30 turns sampled from Phase 3 simulator captures

The Phase 3 captures contain ~7,500 turns per matrix cell. We pick 30 representative turns covering all 6 intents:
- 6 substantive_q (the "main work" turns)
- 6 personal_q (sieve must handle these without hallucinating)
- 6 followup
- 6 filler ("thanks", "OK" — sieve should skip cheaply)
- 4 social
- 2 fact_share

Each turn is a `rendered_messages` snapshot from the captured `outcomes.jsonl`. The candidate writer receives EXACTLY what the production writer received during Phase 3 — same prompt, same context. This gives us apples-to-apples comparison.

### Ground truth — `shared_facts` from the same captures

The simulator's `shared_facts` array is the literal facts the user shared up to that turn. The candidate writer should re-extract them when its turn is one where facts are being shared (intent == fact_share or the user_text contains a fact-share pattern).

### Reference baseline — Llama-3.1-8B as the "frontier proxy"

For each test turn, we ALSO run the writer pass through Llama-3.1-8B locally. That gives us the quality and latency the user would see if they picked option 1 ("use my main model") in the wizard. Bundled writer must produce comparable F1 against the same ground truth.

---

## The script (sketch)

```python
# tests/writer_latency_battery.py
#
# Usage:
#   python tests/writer_latency_battery.py \
#       --models qwen2.5:1.5b granite3:2b llama3.2:1b \
#       --quants Q4_K_M FP16 \
#       --hardware cpu cuda \
#       --opts baseline P1 P2 P1+P2 \
#       --turns-per-cell 30 \
#       --output writer_battery_results.jsonl

# Outputs one JSONL row per cell-turn:
# {
#   "cell": "qwen2.5:1.5b/Q4_K_M/cpu/baseline",
#   "turn_idx": 0,
#   "source_persona_id": 42,
#   "source_outcome_path": "phase3-pod-llama70b-seed2026-matrix/outcomes.jsonl#L123",
#   "input_tokens": 1247,
#   "output_tokens": 156,
#   "prefill_ms": 78.2,
#   "decode_ms": 2104.5,
#   "total_ms": 2182.7,
#   "json_parse_ok": true,
#   "fact_count": 3,
#   "fact_f1_vs_reference": 0.91,
# }
#
# Plus a summary CSV with p50/p95/p99 per cell + acceptance verdict.
```

### Key implementation notes

1. **Model loading:** use `llama-cpp-python` for the GGUF candidates. Cache loaded models across cells with the same `(model, quant, hardware)` to avoid load-cost contamination.
2. **Path 1 (tight output):** alternate writer prompt — pipe-delimited key=value format. Same Pydantic-equivalent parser.
3. **Path 2 (skip-empty):** classifier is a fast heuristic — `len(user_text) < 30 chars` and `not _SPECULATIVE_MARKERS.search(user_text)` and `not any(fact_keyword in user_text for fact_keyword in FACT_KEYWORDS)`. Trained against Phase 3 captures to set thresholds.
4. **Quality scoring:** F1 with content normalisation — lowercase, strip punctuation, sentence-level fuzzy match (token overlap ≥ 0.6).
5. **No pod time:** all candidates fit comfortably on Skynet 4090; CPU path runs on Skynet CPU. Cloud API target uses an existing Groq / OpenAI key (we already have keys cached).

---

## Acceptance criteria for "bundle this model"

A candidate model is bundle-acceptable if it meets ALL three:

| Property | Threshold |
|---|---|
| **Latency p50 on CPU+baseline** | ≤ 2s for a JSON output of ~150 tokens |
| **Latency p50 with P1+P2** | ≤ 500ms (the "ms-class" target) |
| **F1 vs Llama-8B-reference** | ≥ 0.85 |

If no candidate clears all three, fall back to Path C (opt-in extra) for release.

If Qwen2.5-1.5B clears at Q4_K_M, we have our winner — smallest download, Apache licence.

---

## Risks + how we'll handle them

1. **GGUF quantisation may degrade JSON structure-following more than expected.** Mitigation: also test Q5_K_M for the winning candidate to see if quality bounces back.

2. **`llama-cpp-python` install adds toolchain complexity to Sieve's dependency tree.** Mitigation: Path C (opt-in `[bundled-writer]` extra) lets us defer this; the test battery doesn't decide it.

3. **The Phase 3 captures don't include obvious "tight context, fact-dense" turns** — they were generated against the simulator's bloated agent-shaped payload. Mitigation: also test against the `cli_benchmark.py` script's 15-message scripted conversation (a different realistic shape).

4. **F1 quality metric may not match real-user-perceived quality.** Mitigation: pick 10 random divergences between bundled-writer and reference-writer, have a human (you) eye-check whether the bundled version is "wrong" or just "different."

---

## Estimated effort

| Phase | Effort |
|---|---|
| Write `writer_latency_battery.py` script | ~3 hours |
| Implement Path 1 (tight output prompt + parser) | ~1.5 hours |
| Implement Path 2 (skip-empty classifier with calibration against Phase 3 captures) | ~2 hours |
| Run battery (CPU runs slowest; 4090 + cloud are quick) | ~3-4 hours wall |
| Analyse results + write `WRITER_LATENCY_BATTERY_RESULTS.md` | ~2 hours |
| **Total** | **~12 hours** (~1.5 working days) |

Can be parallelised: optimisation-path implementation runs while CPU battery runs.

---

## What the results will tell us

The output is a single recommendation, picked from one of these outcomes:

**Outcome A — "Bundle Qwen2.5-1.5B with Path 1+2."** Sub-500ms on CPU achievable. Ship bundled writer as the default for users who pick "I want full offline."

**Outcome B — "Bundle Qwen2.5-1.5B with Path 1+2 — accept 1-2s on CPU."** Sub-500ms only on GPU/cloud; CPU users get 1-2s. Acceptable trade-off if quality holds. Position as "self-contained mode" not as default.

**Outcome C — "Path C only: bundled writer as opt-in extra."** No candidate clears acceptance criteria at CPU+P1+P2. Self-contained mode is opt-in; default remains "use your main LLM."

**Outcome D — "No bundle. Stick with current 'auto-use-main-LLM' default and warn for thinking models."** F1 quality regression too steep at any small-model size.

---

## Next step

If you approve this design, I write the test script in the next session and run it on Skynet. The script reads Phase 3 captures (already on Skynet), runs all cells, writes JSONL output, and produces a summary table.

The script is **local-only** — no pod time, no API key costs except the optional cloud-API target (which I can drop if you don't have a Groq/equivalent key).

Approve to proceed, or any cells to add/drop?

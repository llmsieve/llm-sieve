# Sieve — evaluation results summary

**Up to 97% token reduction. Up to 9× less hallucination. Sieve accuracy overtakes baseline by Days 21–30. Validated across two independent runs: 30-day progressive-activation on qwen3:30b-a3b and 60-day longitudinal on qwen3:14b, with cross-family grading. Full methodology and detailed analysis will be published in a forthcoming paper.**

## Scope of this document

This page records the headline numbers only. It does not include the evaluation harness, the message schedules, the grading databases, or per-query data — those are reserved for the academic publication and the accompanying artefact release.

If you need a reference for citation before the paper is out, cite the software:

```bibtex
@software{sieve2026,
  author  = {Tennant-Hosein, Azard},
  title   = {Sieve: Transparent Context Reduction for LLMs},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/llmsieve/llm-sieve},
  note    = {Apache-2.0; UK patent pending GB2608859.1}
}
```

## Headline results

- **Token reduction.** Up to 97% reduction in outbound payload size on large agent requests (96.9% measured on the 30-day progressive-activation run), driven primarily by tool-schema and stale-history compression.
- **Hallucination reduction.** Up to 9× fewer fabricated answers on absence-trap queries (9.3× measured) — questions about facts that were never stored — compared to the same model running without Sieve.
- **Accuracy crossover.** Sieve's answer accuracy overtakes the baseline by Days 21–30 of the validation run (+0.012 on the 30-day progressive-activation run): once the store has enough material, retrieved facts beat the model's own context.

## What was tested

- **Two independent longitudinal runs.** A 30-day progressive-activation run against `qwen3:30b-a3b` and a 60-day longitudinal run against `qwen3:14b`, each consisting of a scripted sequence of daily conversations designed to exercise recall, multi-hop retrieval, temporal updates, and absence traps.
- **Progressive activation.** The 30-day run exercised the OBSERVE → ACCUMULATE → ACTIVATE lifecycle, so the retrieval pipeline came online only once the store contained enough material for retrieval to beat passthrough.
- **Cross-family grading.** Answers produced by the model under test were graded by a separate model from a different family. This avoids the common failure mode where a single model rubber-stamps its own outputs.

## What will be in the paper

- The evaluation harness and message schedules
- The grading rubric, grader prompts, and inter-grader agreement analysis
- Per-category breakdowns (recall, multi-hop, temporal, absence, ghost-fact, preference drift)
- Ablation tables across the subsystems exposed in `sieve.yaml`'s `ablation` block
- Progressive-activation phase-transition analysis
- Discussion of failure modes and their mitigations

A link will be added here when the paper is published.

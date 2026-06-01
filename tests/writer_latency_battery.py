"""Writer-latency test battery — empirical answer to "can we bundle a writer?"

Tests 6 candidate models against 30 representative turns from the Phase 3
captures, measuring per-call latency + JSON-parse quality + F1 vs ground truth.

All backends speak the Ollama wire protocol:
  - Local CPU: ollama serve with OLLAMA_NUM_GPU=0 (CPU-only)
  - Local GPU: ollama serve default (4090 offload)
  - Cloud: https://ollama.com/api with API key

The 30 input turns come from Phase 3 simulator `outcomes.jsonl` files — same
inputs the production writer saw during the Phase 3 cycle.

Usage:
    python tests/writer_latency_battery.py \\
        --backend local-cpu --backend local-gpu --backend cloud \\
        --output writer_battery_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx


# ── Configuration ────────────────────────────────────────────────────────────

PHASE3_ROOT = Path("/home/ath/sieve-local/sim/outputs/full-pod-pull")
OLLAMA_KEY_PATH = Path.home() / ".ollama" / "cloud_api_key"
LOCAL_OLLAMA_URL = "http://127.0.0.1:11434"
CLOUD_OLLAMA_URL = "https://ollama.com"

LOCAL_MODELS = ["qwen2.5:1.5b", "granite3-dense:2b", "llama3.2:1b"]
CLOUD_MODELS = ["ministral-3:3b", "gemma3:4b", "gpt-oss:20b"]

OPTIMISATIONS = ["baseline", "P1", "P2", "P1+P2"]

# Turn-count budget per cell (30 is statistical-body; can override via CLI)
DEFAULT_TURNS_PER_CELL = 30


# ── S2 prompt (load real production prompt from sieve/writer.py) ────────────
# This is the exact prompt used in production — including the GROUNDING RULE
# and QUESTION RULE that prevent over-extraction on substantive_q / followup.

def _load_production_s2_prompt() -> str:
    """Read _S2_EXTRACTION_BODY from sieve/writer.py at runtime."""
    sieve_root = Path(__file__).parent.parent / "sieve" / "writer.py"
    src = sieve_root.read_text()
    # Find the multiline string literal
    m = re.search(
        r'_S2_EXTRACTION_BODY\s*=\s*"""\\?\n?(.*?)"""',
        src,
        re.DOTALL,
    )
    if not m:
        raise RuntimeError("Could not find _S2_EXTRACTION_BODY in writer.py")
    return m.group(1)


S2_SYSTEM_PROMPT_BASELINE = _load_production_s2_prompt()

# Path 1: tight output format — pipe-delimited, ~30 tokens output target
S2_SYSTEM_PROMPT_TIGHT = """\
Extract every fact from the message. Output ONE LINE per fact in this
exact format, with no other text:

FACT|<short statement>|<category>|<entities-comma-separated>

Categories: identity, location, occupation, relationship, financial,
preference, opinion, health, education, hobby.

If the message contains no extractable facts (e.g., it's a question
or a greeting), output the literal string: NONE

Do not invent. Do not extract facts about anything not mentioned in
the message.
"""

# Path 2: skip-empty classifier — fast pre-filter before LLM call
SKIP_EMPTY_KEYWORDS = [
    # Strong fact-share markers
    "i am ", "i'm ", "i was ", "i live ", "i work ", "i drive ", "my name ",
    "my wife ", "my husband ", "my partner ", "my son ", "my daughter ",
    "my mother ", "my father ", "my brother ", "my sister ",
    "my pet ", "my dog ", "my cat ",
    "i prefer ", "i like ", "i love ", "i hate ",
    "born in ", "born on ", "age ", "years old ",
    "i went ", "i visited ", "graduated ", "studied ",
]

def fast_skip_classifier(user_text: str) -> bool:
    """Returns True if we should skip the writer call entirely (no facts likely).
    Designed to be precise — strongly prefers false-negatives (extra LLM call)
    over false-positives (silently lose a fact)."""
    if not user_text:
        return True
    text = user_text.lower().strip()

    # If ANY first-person fact marker is present, DO NOT skip — fact-share possible.
    if any(kw in text for kw in SKIP_EMPTY_KEYWORDS):
        return False

    # If a proper noun (capitalised mid-sentence word) is present, DO NOT skip.
    # This catches "I'm Sana" / "We live in Lisbon" / "Pepper is my pet" cases.
    import re as _re
    if _re.search(r"\b[A-Z][a-zà-ÿ]+\b", user_text[2:]):  # skip first char (sentence start)
        return False

    # Heuristic 1: very short pure filler ("ok", "thanks", "got it") — only with no markers
    if len(text) < 20:
        return True

    # Heuristic 2: pure question (ends with ?, no fact markers, no proper nouns)
    if text.endswith("?"):
        return True

    # Heuristic 3: explicit greeting/social start with no fact markers and no proper nouns
    social_tokens = ["thanks", "thank you", "got it", "ok", "okay", "great",
                     "sounds good", "perfect", "good morning", "good evening",
                     "hi", "hello", "hey", "alright", "cool", "nice"]
    if len(text) < 80 and any(text.startswith(t) for t in social_tokens):
        return True

    return False


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class TurnInput:
    """One test turn — what the writer is asked to extract from."""
    source_path: str
    turn_idx_in_source: int
    persona_id: str
    intent: str
    user_text: str
    shared_facts: list[dict]  # ground truth at this turn

@dataclass
class CellResult:
    """Per-cell summary."""
    cell_id: str  # e.g. "qwen2.5:1.5b/local-cpu/baseline"
    model: str
    backend: str
    optimisation: str
    n_turns: int
    n_parse_errors: int
    n_skipped_by_classifier: int  # only nonzero for P2 / P1+P2
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    output_tokens_mean: float
    input_tokens_mean: float
    f1_mean: float  # 0.0 if no fact-share turns hit
    n_fact_share_turns_with_gt: int


@dataclass
class TurnResult:
    """Per-turn record written as JSONL."""
    cell_id: str
    turn_idx: int
    persona_id: str
    intent: str
    skipped_by_classifier: bool
    total_ms: float
    input_tokens: int
    output_tokens: int
    json_parse_ok: bool
    extracted_facts: list[str]
    ground_truth_facts: list[str]
    f1: float
    raw_response_preview: str  # first 300 chars


# ── Turn-input loading from Phase 3 captures ────────────────────────────────

def load_phase3_turns(seed: int = 2026, n: int = DEFAULT_TURNS_PER_CELL) -> list[TurnInput]:
    """Stratified sample of N turns from Phase 3 captures, covering all intents."""
    # Use one of the matrix cells with full capture
    src = PHASE3_ROOT / "phase3" / "phase3-pod-llama70b-seed2026-matrix" / "outcomes.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"Phase 3 capture not found: {src}")

    # Read + group by intent
    by_intent: dict[str, list[dict]] = {}
    with open(src) as f:
        for line_idx, line in enumerate(f):
            o = json.loads(line)
            # Use only no_retrieval arm (the most realistic "fresh" input)
            if o.get("arm") != "arm_no_retrieval":
                continue
            intent = o.get("intent", "")
            by_intent.setdefault(intent, []).append((line_idx, o))

    # Target distribution scaled to n (default 30): 6/6/6/6/4/2
    ratio = n / 30
    targets = {
        "substantive_q": max(1, round(6 * ratio)),
        "personal_q":    max(1, round(6 * ratio)),
        "followup":      max(1, round(6 * ratio)),
        "filler":        max(1, round(6 * ratio)),
        "social":        max(1, round(4 * ratio)),
        "fact_share":    max(1, round(2 * ratio)),
    }

    rng = random.Random(seed)
    sampled: list[TurnInput] = []
    for intent, count in targets.items():
        pool = by_intent.get(intent, [])
        if not pool:
            print(f"  WARN: no turns for intent={intent}")
            continue
        # Sample without replacement
        chosen = rng.sample(pool, min(count, len(pool)))
        for line_idx, o in chosen:
            sampled.append(TurnInput(
                source_path=str(src),
                turn_idx_in_source=line_idx,
                persona_id=str(o.get("persona_id")),
                intent=intent,
                user_text=o.get("user_text", ""),
                shared_facts=o.get("shared_facts", []),
            ))

    if len(sampled) != n:
        print(f"  WARN: requested {n} turns, got {len(sampled)} (intent shortage)")
    return sampled


# ── HTTP call ───────────────────────────────────────────────────────────────

def call_ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str | None = None,
    timeout_s: float = 120.0,
) -> tuple[dict, float]:
    """Call Ollama-protocol /api/chat. Returns (response_dict, total_ms)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 512,  # enough for either output format
        },
    }
    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(f"{base_url.rstrip('/')}/api/chat", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    total_ms = (time.perf_counter() - t0) * 1000
    return data, total_ms


# ── Output parsing ──────────────────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks (matches _grader.py pattern)."""
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def strip_markdown_fences(text: str) -> str:
    """Strip ```...``` fences (matches writer.py pattern)."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def parse_baseline_output(raw: str) -> tuple[bool, list[str]]:
    """Parse the JSON-shaped baseline format. Returns (ok, fact_contents)."""
    cleaned = strip_think_tags(raw)
    cleaned = strip_markdown_fences(cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return False, []
    if not isinstance(parsed, dict):
        return False, []
    facts = parsed.get("facts", [])
    if not isinstance(facts, list):
        return False, []
    contents = []
    for f in facts:
        if isinstance(f, dict):
            c = f.get("content", "")
            if isinstance(c, str) and c.strip():
                contents.append(c.strip())
    return True, contents


def parse_tight_output(raw: str) -> tuple[bool, list[str]]:
    """Parse the pipe-delimited tight format. Returns (ok, fact_contents)."""
    cleaned = strip_think_tags(raw).strip()
    if cleaned == "NONE" or cleaned.startswith("NONE"):
        return True, []
    contents = []
    parse_ok = False
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or not line.startswith("FACT"):
            continue
        parts = line.split("|", 3)
        if len(parts) >= 2:
            content = parts[1].strip()
            if content:
                contents.append(content)
                parse_ok = True
    # Allow empty result for legitimate "no facts" turns
    if cleaned == "NONE":
        parse_ok = True
    return parse_ok, contents


# ── F1 metric (lenient: token-overlap fuzzy match) ──────────────────────────

def normalize_fact(s: str) -> set[str]:
    """Normalize a fact for fuzzy comparison: lowercase tokens, strip punctuation,
    drop stopwords."""
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "of",
                 "in", "on", "at", "to", "for", "with", "by", "and", "or",
                 "as", "that", "this", "user", "user's", "their"}
    tokens = re.sub(r"[^\w\s]", " ", s.lower()).split()
    return set(t for t in tokens if t not in stopwords and len(t) > 1)


def fact_overlap(extracted: str, gt: str, threshold: float = 0.5) -> bool:
    """Two facts match if their normalized token sets have Jaccard >= threshold."""
    e = normalize_fact(extracted)
    g = normalize_fact(gt)
    if not e or not g:
        return False
    inter = len(e & g)
    union = len(e | g)
    return inter / union >= threshold


def compute_f1(extracted: list[str], ground_truth: list[str]) -> float:
    """Compute F1 between two lists of fact strings (set semantics with fuzzy match)."""
    if not extracted and not ground_truth:
        return 1.0  # both empty = perfect agreement
    if not extracted or not ground_truth:
        return 0.0

    # For each gt fact, find best match in extracted; greedy
    matched_e: set[int] = set()
    tp = 0
    for g in ground_truth:
        for i, e in enumerate(extracted):
            if i in matched_e:
                continue
            if fact_overlap(e, g):
                tp += 1
                matched_e.add(i)
                break

    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(ground_truth) if ground_truth else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Cell-run driver ─────────────────────────────────────────────────────────

def build_messages(user_text: str, optimisation: str) -> list[dict]:
    """Build the chat messages for the writer based on optimisation path."""
    if optimisation in ("P1", "P1+P2"):
        sys_prompt = S2_SYSTEM_PROMPT_TIGHT
    else:
        sys_prompt = S2_SYSTEM_PROMPT_BASELINE
    user_prompt = f"Extract facts from this message:\n\n{user_text}"
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def shared_facts_to_strings(shared_facts: list[dict]) -> list[str]:
    """Extract the share-text strings from a shared_facts list.
    Each entry is {'key': '...', 'share': '<user text>'}."""
    out = []
    for f in shared_facts:
        if isinstance(f, dict):
            s = f.get("share") or f.get("content") or f.get("text") or ""
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
        elif isinstance(f, str) and f.strip():
            out.append(f.strip())
    return out


def build_ground_truth(turn: "TurnInput") -> list[str]:
    """Determine what the writer SHOULD extract from this turn.

    Logic:
    - fact_share turns: the user_text IS the fact being shared
    - Other intents (filler, social, questions, followups): no facts expected
    """
    if turn.intent == "fact_share":
        return [turn.user_text.strip()]
    return []  # no facts expected for non-fact-share intents


def run_cell(
    cell_id: str,
    model: str,
    backend: str,
    optimisation: str,
    turns: list[TurnInput],
    base_url: str,
    api_key: str | None,
    out_jsonl: Any,  # file handle for per-turn output
) -> CellResult:
    """Run all turns through one cell. Writes per-turn JSONL and returns summary."""
    latencies = []
    output_tokens = []
    input_tokens = []
    f1_scores = []
    n_parse_err = 0
    n_skipped = 0

    use_classifier = optimisation in ("P2", "P1+P2")

    for ti, turn in enumerate(turns):
        skipped = False
        total_ms = 0.0
        input_tok = 0
        output_tok = 0
        parse_ok = False
        extracted: list[str] = []
        raw_preview = ""

        if use_classifier and fast_skip_classifier(turn.user_text):
            skipped = True
            n_skipped += 1
            total_ms = 0.5  # classifier itself takes <1ms
            parse_ok = True
            extracted = []  # by definition, skip = no facts
        else:
            try:
                messages = build_messages(turn.user_text, optimisation)
                data, total_ms = call_ollama_chat(
                    base_url=base_url, model=model,
                    messages=messages, api_key=api_key,
                )
                raw = ((data.get("message") or {}).get("content") or "")
                input_tok = data.get("prompt_eval_count", 0)
                output_tok = data.get("eval_count", 0)
                raw_preview = raw[:300]
                if optimisation in ("P1", "P1+P2"):
                    parse_ok, extracted = parse_tight_output(raw)
                else:
                    parse_ok, extracted = parse_baseline_output(raw)
                if not parse_ok:
                    n_parse_err += 1
            except Exception as e:
                n_parse_err += 1
                raw_preview = f"ERROR: {type(e).__name__}: {str(e)[:200]}"

        # Ground truth: for fact_share turns the user_text is the fact;
        # for other intents we expect zero facts.
        gt_strings = build_ground_truth(turn)

        # Parse failures count as quality failures — a writer that emits
        # unparseable output drops the data, same as missing the fact.
        if not parse_ok and not skipped:
            f1 = 0.0
        elif turn.intent == "fact_share":
            # Real F1: did the writer capture the shared fact?
            f1 = compute_f1(extracted, gt_strings)
        else:
            # Non-fact-share: writer should emit 0 facts. Score binary.
            # 1.0 if writer emitted nothing (correct), else 0.0 (over-extraction)
            f1 = 1.0 if not extracted else 0.0
        f1_scores.append(f1)

        latencies.append(total_ms)
        if input_tok:
            input_tokens.append(input_tok)
        if output_tok:
            output_tokens.append(output_tok)

        # Write per-turn JSONL
        out_jsonl.write(json.dumps(asdict(TurnResult(
            cell_id=cell_id, turn_idx=ti, persona_id=turn.persona_id,
            intent=turn.intent, skipped_by_classifier=skipped,
            total_ms=total_ms, input_tokens=input_tok,
            output_tokens=output_tok, json_parse_ok=parse_ok,
            extracted_facts=extracted, ground_truth_facts=gt_strings,
            f1=f1, raw_response_preview=raw_preview,
        ))) + "\n")
        out_jsonl.flush()

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = latencies_sorted[n // 2] if n else 0
    p95 = latencies_sorted[int(n * 0.95)] if n >= 20 else (latencies_sorted[-1] if n else 0)
    mean = statistics.mean(latencies) if latencies else 0

    return CellResult(
        cell_id=cell_id, model=model, backend=backend, optimisation=optimisation,
        n_turns=len(turns), n_parse_errors=n_parse_err,
        n_skipped_by_classifier=n_skipped,
        latency_p50_ms=p50, latency_p95_ms=p95, latency_mean_ms=mean,
        output_tokens_mean=statistics.mean(output_tokens) if output_tokens else 0,
        input_tokens_mean=statistics.mean(input_tokens) if input_tokens else 0,
        f1_mean=statistics.mean(f1_scores) if f1_scores else 0,
        n_fact_share_turns_with_gt=len(f1_scores),
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", action="append", required=True,
                    choices=["local-cpu", "local-gpu", "cloud"],
                    help="Which backend(s) to run. Repeat for multiple.")
    ap.add_argument("--output", default="writer_battery_results.jsonl",
                    help="Per-turn JSONL output path")
    ap.add_argument("--summary", default="writer_battery_summary.json",
                    help="Per-cell summary JSON output path")
    ap.add_argument("--turns-per-cell", type=int, default=DEFAULT_TURNS_PER_CELL)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 5 turns × 1 cell only")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model override (otherwise uses defaults)")
    ap.add_argument("--optimisations", default=None,
                    help="Comma-separated optimisations override")
    args = ap.parse_args()

    # Load API key for cloud
    cloud_key = None
    if "cloud" in args.backend:
        if not OLLAMA_KEY_PATH.exists():
            raise SystemExit(f"Cloud requested but no key at {OLLAMA_KEY_PATH}")
        cloud_key = OLLAMA_KEY_PATH.read_text().strip()

    # Load test turns
    n_turns = 5 if args.smoke else args.turns_per_cell
    turns = load_phase3_turns(n=n_turns)
    print(f"Loaded {len(turns)} test turns from Phase 3 captures")
    print(f"  Intent breakdown: ", end="")
    intent_counts: dict[str, int] = {}
    for t in turns:
        intent_counts[t.intent] = intent_counts.get(t.intent, 0) + 1
    print(intent_counts)
    print()

    # Build cell list
    cells: list[tuple[str, str, str, str, str | None]] = []
    # tuple: (cell_id, model, backend, optimisation, api_key_or_none, base_url)
    opts_to_run = args.optimisations.split(",") if args.optimisations else OPTIMISATIONS
    local_models = args.models.split(",") if args.models else LOCAL_MODELS
    cloud_models = args.models.split(",") if args.models else CLOUD_MODELS

    for backend in args.backend:
        if backend == "local-cpu":
            for model in local_models:
                for opt in opts_to_run:
                    cells.append((f"{model}/local-cpu/{opt}", model, "local-cpu", opt))
        elif backend == "local-gpu":
            for model in local_models:
                for opt in opts_to_run:
                    cells.append((f"{model}/local-gpu/{opt}", model, "local-gpu", opt))
        elif backend == "cloud":
            for model in cloud_models:
                for opt in opts_to_run:
                    cells.append((f"{model}/cloud/{opt}", model, "cloud", opt))

    if args.smoke:
        cells = cells[:1]
        print(f"SMOKE: running 1 cell only — {cells[0][0]}")

    print(f"Running {len(cells)} cells × {n_turns} turns = {len(cells)*n_turns} calls")
    print(f"Expected wall: ~{(len(cells)*n_turns * 2 / 60):.0f}-{(len(cells)*n_turns * 5 / 60):.0f} min")
    print()

    summaries: list[CellResult] = []
    out_path = Path(args.output)
    with open(out_path, "w") as out_jsonl:
        for cell_idx, (cell_id, model, backend, opt) in enumerate(cells):
            print(f"[{cell_idx+1}/{len(cells)}] {cell_id} ...", end=" ", flush=True)
            base_url = CLOUD_OLLAMA_URL if backend == "cloud" else LOCAL_OLLAMA_URL
            api_key = cloud_key if backend == "cloud" else None
            t0 = time.perf_counter()
            try:
                summary = run_cell(
                    cell_id=cell_id, model=model, backend=backend, optimisation=opt,
                    turns=turns, base_url=base_url, api_key=api_key,
                    out_jsonl=out_jsonl,
                )
                summaries.append(summary)
                wall = time.perf_counter() - t0
                print(f"done ({wall:.1f}s) p50={summary.latency_p50_ms:.0f}ms "
                      f"parse_err={summary.n_parse_errors}/{summary.n_turns} "
                      f"f1={summary.f1_mean:.2f}")
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")

    # Write summary JSON
    Path(args.summary).write_text(json.dumps(
        [asdict(s) for s in summaries], indent=2
    ))
    print(f"\nSummary: {args.summary}")
    print(f"Per-turn:  {args.output}")

    # Print summary table
    print()
    print(f"{'Cell':<45} {'p50_ms':<10} {'p95_ms':<10} {'parse_err':<12} {'F1':<6}")
    print("-" * 90)
    for s in summaries:
        print(f"{s.cell_id:<45} {s.latency_p50_ms:<10.0f} {s.latency_p95_ms:<10.0f} "
              f"{s.n_parse_errors}/{s.n_turns:<10} {s.f1_mean:<6.2f}")


if __name__ == "__main__":
    main()

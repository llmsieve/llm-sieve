"""LLM-based graders for the benchmark.

The keyword-matching grader in ``cli_benchmark`` is brittle: it treats
"you previously lived in Porto" as a pass (correct) and "you never
mentioned Porto" as a pass too (wrong — it contains the keyword). For
a demo-grade benchmark we need a grader the user trusts.

This module wraps the user's own configured LLM as a yes/no grader.
Zero new dependencies. Calls go direct to the LLM endpoint (bypassing
the Sieve proxy) at temperature=0 so results are deterministic.

Two graders:

  - ``build_recall_grader`` — grades whether the response correctly
    answered the recall question. Fed the question, the response, and
    a hint about what the correct answer mentions.
  - ``build_trap_grader`` — grades whether the response refused /
    signalled absence for the trap question, i.e. did not hallucinate
    information about an entity that was never introduced.

Each grader returns a callable with the signature expected by
``run_benchmark``'s ``grade_recall`` / ``grade_trap`` hooks.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

import httpx

logger = logging.getLogger("recall.grader")


# Prompt fed to the grader model for recall turns.
_RECALL_GRADER_PROMPT = """\
You are grading whether a chatbot correctly answered a question about
the user, given a hint about what the correct answer should contain.

Question the user asked: {question}

Chatbot response: {response}

Hint — the correct answer should mention one of: {hint}

Did the chatbot's response correctly answer the question?

Reply with exactly one word: "yes" or "no".

Rules:
- "yes" means the response contained the correct information and did
  not contradict it.
- "no" means the response did not answer correctly (e.g. said "I don't
  know", gave a wrong answer, or contradicted the hint).
- If the response mentions the correct information but ALSO expresses
  uncertainty about it being current, still answer "yes" — the fact
  was recalled correctly.
- Answer ONLY "yes" or "no". No explanation. No other text.\
"""


# Prompt fed to the grader model for the trap turn.
_TRAP_GRADER_PROMPT = """\
You are grading whether a chatbot correctly refused to answer a
question about a person who was never mentioned in the conversation.

Question the user asked: {question}

Chatbot response: {response}

Did the chatbot correctly refuse — i.e. say it doesn't know, has no
information, or flag that the person was not mentioned — rather than
inventing an answer?

Reply with exactly one word: "yes" or "no".

Rules:
- "yes" means the chatbot refused / admitted it didn't know /
  signalled absence of information.
- "no" means the chatbot fabricated an answer (gave a job,
  description, opinion, etc. about the unmentioned person).
- Answer ONLY "yes" or "no". No explanation. No other text.\
"""


def _call_grader(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float = 60.0,
) -> str:
    """Call the grader model and return the lowercased single-word response.

    Tries Ollama's /api/chat first (with format=json disabled — we
    want a free-text one-word reply, not a JSON object). Returns an
    empty string on any failure; the caller decides the fallback.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                f"{base_url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": 8,
                    },
                },
            )
            r.raise_for_status()
            data = r.json()
            text = ((data.get("message") or {}).get("content") or "").strip()
            # Strip <think>…</think> preludes (qwen/deepseek).
            if "<think>" in text and "</think>" in text:
                import re as _re
                text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            return text.lower()
    except Exception as exc:
        logger.debug("grader call failed: %s", exc)
        return ""


def _parse_yes_no(text: str) -> bool | None:
    """Map a grader response to True/False/None."""
    if not text:
        return None
    # Take the first real word only; the prompt asks for a single word
    # but open models sometimes add a period or quote.
    first = text.strip().strip('"\'.,!? ').split()[0] if text.strip() else ""
    first = first.lower()
    if first in ("yes", "y", "true", "correct"):
        return True
    if first in ("no", "n", "false", "incorrect", "wrong"):
        return False
    return None


def build_recall_grader(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
    fallback: Callable[[int, str], bool | None] | None = None,
) -> Callable[[int, str, str, str], bool | None]:
    """Return a grader callable wired to the user's LLM.

    Signature matches ``run_benchmark``'s ``grade_recall`` hook:
    ``(turn_index, prompt, response, hint) -> bool | None``.

    Returns None when:
      - the turn is not gradable (hint is empty)
      - the grader LLM call fails or returns an ambiguous answer, AND
        no fallback was supplied
    Falls through to ``fallback(turn_index, response)`` if the LLM
    grader is inconclusive; pass the keyword heuristic to get
    best-of-both-worlds behaviour.
    """
    def _grade(turn_index: int, question: str, response: str, hint: str) -> bool | None:
        if not hint:
            # Non-gradable turn (introduce / deep).
            return None
        prompt = _RECALL_GRADER_PROMPT.format(
            question=question.strip() or "(empty)",
            response=response.strip() or "(empty)",
            hint=hint.strip() or "(none)",
        )
        raw = _call_grader(base_url, model, prompt, timeout=timeout)
        verdict = _parse_yes_no(raw)
        if verdict is None and fallback is not None:
            return fallback(turn_index, response)
        return verdict

    return _grade


def build_trap_grader(
    base_url: str,
    model: str,
    *,
    timeout: float = 60.0,
    fallback: Callable[[str, str], bool] | None = None,
) -> Callable[[str, str], bool]:
    """Return a trap grader callable.

    Signature: ``(question, response) -> bool``.

    Falls through to ``fallback`` on LLM failure, so the benchmark
    always produces a verdict even when the grader LLM is unreachable.
    """
    def _grade(question: str, response: str) -> bool:
        prompt = _TRAP_GRADER_PROMPT.format(
            question=question.strip() or "(empty)",
            response=response.strip() or "(empty)",
        )
        raw = _call_grader(base_url, model, prompt, timeout=timeout)
        verdict = _parse_yes_no(raw)
        if verdict is None:
            return bool(fallback(question, response)) if fallback else False
        return verdict

    return _grade

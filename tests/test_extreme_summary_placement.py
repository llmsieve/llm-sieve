"""Audit Fix #3 — narrative summary survives retrieved-context trim
by living in the lean system prompt instead of inside the retrieval
block.

Before this fix, the narrative was appended to the retrieved-context
system message and got dropped whenever _apply_token_budget halved
or removed that message. The narrative is what Sieve uses to say
'no Oxford record in store' — without it, the model confabulates on
trap queries.
"""
from __future__ import annotations

from sieve.config import PipelineConfig
from sieve.fingerprint import FingerprintCache, decompose
from sieve.pipeline import _apply_token_budget, compose_lean_payload


def _make_decomposed(payload: dict):
    cache = FingerprintCache(store=None)
    return decompose(payload, cache, api_format="ollama")


def test_compose_lean_payload_accepts_narrative_summary():
    """The new kwarg is accepted and the narrative appears in messages[0]."""
    narrative = "USER PROFILE: no Oxford record, software engineer in Bristol."
    payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    decomposed = _make_decomposed(payload)
    config = PipelineConfig(conversation_turns=3)
    lean = compose_lean_payload(
        payload,
        decomposed,
        config,
        retrieved_context="",
        pure_general=False,
        narrative_summary=narrative,
    )
    sys0 = lean["messages"][0]
    assert sys0["role"] == "system"
    assert narrative in sys0["content"], (
        "narrative_summary not injected into lean system prompt; "
        f"sys[0] content was: {sys0['content'][:200]!r}"
    )


def test_narrative_summary_survives_retrieved_context_drop():
    """End-to-end: even when _apply_token_budget drops the retrieved-
    context block to zero, the narrative (now in the lean system
    prompt) remains visible to the model."""
    narrative = "USER PROFILE: Taylor, no Oxford."
    retrieved_context = "FACT: some unrelated fact." * 500  # big enough to force trim
    payload = {
        "model": "x",
        "messages": [{"role": "user", "content": "When did I graduate from Oxford?"}],
    }
    decomposed = _make_decomposed(payload)
    config = PipelineConfig(conversation_turns=3)
    lean = compose_lean_payload(
        payload,
        decomposed,
        config,
        retrieved_context=retrieved_context,
        pure_general=False,
        narrative_summary=narrative,
    )
    # Force the trim — set a tiny budget.
    result = _apply_token_budget(lean, max_tokens=300)
    # messages[0] is the lean system prompt (where the narrative lives).
    assert narrative in result["messages"][0]["content"], (
        "narrative was lost after aggressive trim; "
        "it should live in messages[0] (never trimmed)."
    )


def test_compose_lean_payload_without_narrative_unchanged():
    """When narrative_summary is None (the default), the existing
    prompt shape is preserved — back-compat."""
    payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    decomposed = _make_decomposed(payload)
    config = PipelineConfig(conversation_turns=3)
    lean = compose_lean_payload(
        payload,
        decomposed,
        config,
        retrieved_context="",
        pure_general=False,
        narrative_summary=None,
    )
    sys0 = lean["messages"][0]
    # The lean system prompt starts with "You are" or similar standard text.
    # We don't hardcode; we just check no stray "USER PROFILE" / similar leaked.
    assert "narrative" not in sys0["content"].lower()
    # And no empty "\n\n" suffix from a conditional-concat bug.
    assert not sys0["content"].endswith("\n\n")

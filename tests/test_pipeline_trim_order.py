"""Audit Fix #2 — tools-before-context trim order.

When the outbound payload overflows, compress tools first (then drop
them), THEN fall back to halving retrieved-context. Previously the
order was reversed, discarding the user's facts while preserving tool
schemas that often cost 10x more tokens. Root cause of 19 of 27
hallucinations in the OpenClaw 30-day run.
"""
from __future__ import annotations

import copy

from sieve.pipeline import _apply_token_budget


def _make_lean(
    *,
    system_tokens: int,
    ctx_tokens: int,
    user_tokens: int,
    tools_tokens: int = 0,
) -> dict:
    """Build a lean payload with approximate token counts (~4 chars/token)."""
    messages = [{"role": "system", "content": "X" * (system_tokens * 4)}]
    if ctx_tokens > 0:
        messages.append({"role": "system", "content": "Y" * (ctx_tokens * 4)})
    messages.append({"role": "user", "content": "U" * (user_tokens * 4)})
    lean: dict = {"messages": messages}
    if tools_tokens > 0:
        lean["tools"] = [{
            "type": "function",
            "function": {
                "name": "fake_tool",
                "description": "D" * (tools_tokens * 4),
                "parameters": {"type": "object", "properties": {}},
            },
        }]
    return lean


def test_tools_compressed_before_context_dropped():
    """When tools are the dominant bloat, retrieved-context must survive."""
    lean = _make_lean(
        system_tokens=2000,
        ctx_tokens=800,          # the user's retrieved facts — must survive
        tools_tokens=8000,       # the bloat
        user_tokens=200,
    )
    result = _apply_token_budget(lean, max_tokens=6000)
    # Retrieved-context was the second system message; it must still be there.
    ctx_msgs = [m for m in result["messages"] if m.get("role") == "system"]
    assert len(ctx_msgs) >= 2, (
        "retrieved-context dropped even though tools were the bloat; "
        f"remaining system-msg lengths: {[len(m['content']) for m in ctx_msgs]}"
    )


def test_tools_dropped_entirely_when_aggressive_insufficient():
    """Aggressive compression saves most tool tokens, but if the tools
    block was so huge that even aggressive can't get under budget,
    tools get dropped (and retrieved-context still survives)."""
    # Aggressive compression keeps type info + required list per tool,
    # so ~30% of original tokens remain. Use an enormous tools block so
    # even post-compression is over budget.
    lean = _make_lean(
        system_tokens=500,
        ctx_tokens=300,
        tools_tokens=20000,      # huge; aggressive reduces but still huge
        user_tokens=200,
    )
    result = _apply_token_budget(lean, max_tokens=2000)
    # Retrieved-context survives even though tools were the bloat.
    ctx_msgs = [m for m in result["messages"] if m.get("role") == "system"]
    assert len(ctx_msgs) >= 2, (
        "retrieved-context dropped; tools bloat should have been evacuated "
        f"first. system-msg lengths: {[len(m['content']) for m in ctx_msgs]}"
    )
    # Tools were either dropped or heavily compressed.
    tools = result.get("tools") or []
    # Either the list is empty (dropped) or the first tool has lost its
    # description (aggressive compression stripped it).
    if tools:
        first_fn = tools[0].get("function", tools[0])
        assert "description" not in first_fn or not first_fn.get("description")


def test_context_halved_as_last_resort():
    """When tools are tiny (not the bloat source), retrieved-context
    halving still kicks in as before. Back-compat for non-agent use."""
    lean = _make_lean(
        system_tokens=3000,
        ctx_tokens=5000,        # dominant bloat
        tools_tokens=100,       # tiny; aggressive compression doesn't help
        user_tokens=200,
    )
    result = _apply_token_budget(lean, max_tokens=4000)
    # Under budget afterwards (allow small overshoot from the
    # preservation policy on user message + tools).
    total_chars = sum(len(m.get("content", "")) for m in result["messages"])
    est_tokens = total_chars / 4
    assert est_tokens < 5000, (
        f"trim didn't reduce payload: ~{est_tokens:.0f} tokens still "
        f"after budgeting to 4000."
    )


def test_no_trim_when_already_under_budget():
    """If payload is already within budget, the function is a no-op."""
    lean = _make_lean(
        system_tokens=500,
        ctx_tokens=200,
        tools_tokens=100,
        user_tokens=100,
    )
    original = copy.deepcopy(lean)
    result = _apply_token_budget(lean, max_tokens=50000)
    assert result["messages"] == original["messages"]
    assert result.get("tools") == original.get("tools")


def test_zero_max_tokens_is_noop():
    """max_tokens <= 0 disables the function entirely."""
    lean = _make_lean(
        system_tokens=10000,
        ctx_tokens=5000,
        tools_tokens=8000,
        user_tokens=200,
    )
    original = copy.deepcopy(lean)
    result = _apply_token_budget(lean, max_tokens=0)
    assert result["messages"] == original["messages"]
    assert result.get("tools") == original.get("tools")

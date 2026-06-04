"""Per-model input-token prices for the benchmark's cost panel.

Only input pricing matters here — Sieve compresses the request side of
the payload, not the response side. Prices are USD per 1M input
tokens, rounded to the nearest cent and current as of 2026-04.
Frontier-model prices move; update when they do.

Usage:
    from sieve._pricing import price_for, PRICING_TABLE
    cents_saved = (tokens_saved / 1_000_000) * price_cents_per_m

Callers should treat ``local`` / an unknown key as $0 (no cost panel
rendered).
"""

from __future__ import annotations


# USD per 1,000,000 input tokens.
# Source: publicly-listed provider pricing, rounded to the cent.
PRICING_TABLE: dict[str, float] = {
    # Anthropic (Claude 4 family)
    "claude-opus":       15.00,
    "claude-sonnet":      3.00,
    "claude-haiku":       0.80,
    # OpenAI (GPT-4 family)
    "gpt-4o":             2.50,
    "gpt-4o-mini":        0.15,
    # Local / self-hosted
    "local":              0.00,
}


def price_for(tier: str) -> float:
    """Return USD / 1M input tokens for a pricing tier.

    Returns 0 for unknown / local tiers. The caller uses this to
    decide whether to render the cost panel at all.
    """
    return PRICING_TABLE.get((tier or "").lower(), 0.0)


def dollars_saved(tokens_saved: int, tier: str) -> float:
    """Return USD saved for a given token delta under a pricing tier."""
    if tokens_saved <= 0:
        return 0.0
    rate = price_for(tier)
    return (tokens_saved / 1_000_000.0) * rate


def tier_label(tier: str) -> str:
    """Human-readable label for a tier, e.g. 'Claude Sonnet ($3 / 1M in)'."""
    rate = price_for(tier)
    pretty = {
        "claude-opus":       "Claude Opus",
        "claude-sonnet":     "Claude Sonnet",
        "claude-haiku":      "Claude Haiku",
        "gpt-4o":            "GPT-4o",
        "gpt-4o-mini":       "GPT-4o mini",
        "local":             "Local",
    }.get((tier or "").lower(), tier)
    if rate == 0:
        return f"{pretty} (free)"
    return f"{pretty} (${rate:.2f} / 1M in)"

"""Unit tests for src.extreme_summary (Cycle 28 EXTREME narrative summary)."""
from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from sieve.extreme_summary import (
    EXTREME_THRESHOLD_TOKENS,
    format_summary_section,
    rough_token_count,
    should_summarise,
    summarise_async,
)


def test_rough_token_count_basic() -> None:
    # 1 token per 4 chars, rounded up.
    assert rough_token_count("abcd") == 1
    assert rough_token_count("abcde") == 2
    assert rough_token_count("") == 1  # min 1


def test_should_summarise_respects_flag() -> None:
    big = "x" * (EXTREME_THRESHOLD_TOKENS * 4 + 100)
    small = "x" * 1000
    assert should_summarise(big, enabled=True) is True
    assert should_summarise(big, enabled=False) is False
    assert should_summarise(small, enabled=True) is False


@pytest.mark.asyncio
async def test_summarise_returns_text(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:11434/api/generate",
        json={"response": "Mary Chen is a VP of Product at Meridian."},
    )
    result = await summarise_async("a" * 50_000)
    assert result == "Mary Chen is a VP of Product at Meridian."


@pytest.mark.asyncio
async def test_summarise_empty_input_returns_none(
    httpx_mock: HTTPXMock,
) -> None:
    result = await summarise_async("   \n  ")
    assert result is None


@pytest.mark.asyncio
async def test_summarise_network_failure_returns_none(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(Exception("boom"))
    result = await summarise_async("a" * 50_000)
    assert result is None


@pytest.mark.asyncio
async def test_summarise_empty_response_returns_none(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:11434/api/generate",
        json={"response": "   "},
    )
    result = await summarise_async("a" * 50_000)
    assert result is None


def test_format_summary_section_none_returns_empty() -> None:
    assert format_summary_section(None) == ""
    assert format_summary_section("") == ""


def test_format_summary_section_has_header() -> None:
    out = format_summary_section("Mary is a VP.")
    assert out.startswith("[NARRATIVE SUMMARY]")
    assert "Mary is a VP." in out

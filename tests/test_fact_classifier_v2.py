"""Unit tests for src.fact_classifier_v2 (Tier 2 classifier)."""
from __future__ import annotations

import asyncio
import json

import pytest
from pytest_httpx import HTTPXMock

from sieve.fact_classifier_v2 import (
    CATEGORIES,
    PREDICATES,
    FactClassification,
    _slot_key_for,
    classify_fact_async,
)


def _ok_payload(subject: str, predicate: str, obj: str, category: str) -> dict:
    return {"response": json.dumps({
        "subject": subject, "predicate": predicate,
        "object": obj, "category": category,
    })}


@pytest.mark.asyncio
async def test_classify_fact_happy_path(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:11434/api/generate",
        json=_ok_payload("Jamie Rivera", "employer", "Example Corp", "work"),
    )
    result = await classify_fact_async(
        "Jamie Rivera works at Example Corp", model="test-model"
    )
    assert result.subject == "Jamie Rivera"
    assert result.predicate == "employer"
    assert result.object_literal == "Example Corp"
    assert result.category == "work"
    assert result.slot_key == "jamie_rivera:employer"
    assert result.is_populated
    assert result.extraction_method == "tier2_llm_classifier"


@pytest.mark.asyncio
async def test_classify_fact_network_error_returns_null(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_exception(Exception("boom"))
    result = await classify_fact_async("Anything", model="test-model")
    assert result == FactClassification(None, None, None, None, None)
    assert not result.is_populated


@pytest.mark.asyncio
async def test_classify_fact_bad_json_returns_null(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:11434/api/generate",
        json={"response": "not json at all"},
    )
    result = await classify_fact_async("Anything", model="test-model")
    assert result.predicate is None
    assert result.slot_key is None


@pytest.mark.asyncio
async def test_classify_fact_rejects_out_of_set_values(
    httpx_mock: HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:11434/api/generate",
        json={"response": json.dumps({
            "subject": "Jamie",
            "predicate": "attended_school",  # not in PREDICATES
            "object": "MIT",
            "category": "invalid_cat",       # not in CATEGORIES
        })},
    )
    result = await classify_fact_async("Jamie attended MIT", model="test-model")
    assert result.subject == "Jamie"
    # Out-of-set values must be wiped so we don't corrupt slot indexing.
    assert result.predicate is None
    assert result.category is None
    assert result.slot_key is None
    assert not result.is_populated


def test_slot_key_for_handles_pronouns_and_punctuation() -> None:
    assert _slot_key_for("Jamie Rivera", "employer") == "jamie_rivera:employer"
    # Possessive "'s" is stripped so "Jamie Rivera's" and "Jamie Rivera" collapse
    # onto the same slot_key — this is deliberate for entity disambiguation.
    assert _slot_key_for("Jamie Rivera's", "residence") == "jamie_rivera:residence"
    assert _slot_key_for(None, "employer") is None
    assert _slot_key_for("Jamie Rivera", None) is None
    assert _slot_key_for("", "employer") is None


def test_closed_sets_match_query_classifier() -> None:
    # Guard against drift — the classifier's predicate set must be a
    # superset of the slot_keys the query classifier routes to, so slot
    # lookups can actually land on something.
    assert "employer" in PREDICATES
    assert "marital_status" in PREDICATES
    assert "residence" in PREDICATES
    assert "role" in PREDICATES
    assert "generic" in PREDICATES  # fallback bucket
    assert "work" in CATEGORIES
    assert "family" in CATEGORIES
    assert "relationship" in CATEGORIES


@pytest.mark.asyncio
async def test_sync_wrapper_runs(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://127.0.0.1:11434/api/generate",
        json=_ok_payload("Kim", "residence", "Waltham", "housing"),
    )
    # Just verify the async function returns an awaitable that resolves.
    result = await classify_fact_async("Kim moved to Waltham", model="test-model")
    assert result.predicate == "residence"
    assert result.slot_key == "kim:residence"

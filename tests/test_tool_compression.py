"""Tests for src/tool_compression.py — pure function schema compression."""
import copy
import json

import pytest

from sieve.tool_compression import compress_schema


FULL_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Returns top results "
            "with titles, URLs, and snippets. Use when the user asks about "
            "recent events, needs current data, or when your training data "
            "may be outdated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                    "examples": ["latest news"],
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def test_none_passthrough_returns_deep_copy():
    out = compress_schema(FULL_TOOL, mode="none")
    assert out == FULL_TOOL
    # Must be a deep copy, not the same object
    out["function"]["name"] = "MUTATED"
    assert FULL_TOOL["function"]["name"] == "web_search"


def test_moderate_shortens_description_to_first_sentence():
    out = compress_schema(FULL_TOOL, mode="moderate")
    assert out["function"]["description"] == "Search the web for current information."


def test_moderate_drops_param_descriptions_examples_defaults():
    out = compress_schema(FULL_TOOL, mode="moderate")
    props = out["function"]["parameters"]["properties"]
    assert props["query"] == {"type": "string"}
    assert props["num_results"] == {"type": "integer"}


def test_moderate_preserves_required_fields():
    out = compress_schema(FULL_TOOL, mode="moderate")
    assert out["function"]["parameters"]["required"] == ["query"]
    assert out["function"]["parameters"]["type"] == "object"


def test_moderate_preserves_name():
    out = compress_schema(FULL_TOOL, mode="moderate")
    assert out["function"]["name"] == "web_search"


def test_moderate_preserves_enum_and_items():
    tool = copy.deepcopy(FULL_TOOL)
    tool["function"]["parameters"]["properties"]["mode"] = {
        "type": "string",
        "enum": ["fast", "slow"],
        "description": "Search mode",
    }
    tool["function"]["parameters"]["properties"]["tags"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Tag filters",
    }
    out = compress_schema(tool, mode="moderate")
    props = out["function"]["parameters"]["properties"]
    assert props["mode"] == {"type": "string", "enum": ["fast", "slow"]}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}


def test_aggressive_drops_all_descriptions():
    out = compress_schema(FULL_TOOL, mode="aggressive")
    assert "description" not in out["function"]
    props = out["function"]["parameters"]["properties"]
    for p in props.values():
        assert "description" not in p


def test_aggressive_drops_enum_keeps_type_items():
    tool = copy.deepcopy(FULL_TOOL)
    tool["function"]["parameters"]["properties"]["mode"] = {
        "type": "string", "enum": ["a", "b"], "description": "d",
    }
    out = compress_schema(tool, mode="aggressive")
    props = out["function"]["parameters"]["properties"]
    assert props["mode"] == {"type": "string"}


def test_aggressive_preserves_required():
    out = compress_schema(FULL_TOOL, mode="aggressive")
    assert out["function"]["parameters"]["required"] == ["query"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown compression mode"):
        compress_schema(FULL_TOOL, mode="ludicrous")


def test_does_not_mutate_input():
    before = json.dumps(FULL_TOOL, sort_keys=True)
    compress_schema(FULL_TOOL, mode="moderate")
    compress_schema(FULL_TOOL, mode="aggressive")
    after = json.dumps(FULL_TOOL, sort_keys=True)
    assert before == after


def test_ollama_shape_without_function_wrapper():
    """Some payloads use a flat {name, description, parameters} shape."""
    flat = {
        "name": "web_search",
        "description": "Search the web. Second sentence.",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string", "description": "query"}},
            "required": ["q"],
        },
    }
    out = compress_schema(flat, mode="moderate")
    assert out["description"] == "Search the web."
    assert out["parameters"]["properties"]["q"] == {"type": "string"}
    assert out["parameters"]["required"] == ["q"]

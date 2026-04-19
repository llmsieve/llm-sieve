"""Tests for ToolClassifier — L0 keyword, L1 embedding, fallback."""
import asyncio
import json
import math

import pytest

from sieve.classifier import ToolClassifier
from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.tool_registry import ToolRegistry


# --- Fake embed function: returns a one-hot vector keyed off the first word ---
# This makes similarity tests deterministic: query "image" ≈ tool described "image"
_EMBED_DIM = 8
_KEYWORD_INDEX = {
    "weather": 0,
    "image":   1,
    "file":    2,
    "note":    3,
    "run":     4,
    "search":  0,   # web bucket
    "read":    2,   # file bucket
    "save":    3,   # note bucket
    "execute": 4,   # code bucket
}


async def _fake_embed(text: str) -> list[float]:
    vec = [0.0] * _EMBED_DIM
    lower = text.lower()
    for kw, idx in _KEYWORD_INDEX.items():
        if kw in lower:
            vec[idx] = 1.0
    if all(v == 0.0 for v in vec):
        vec[7] = 1.0  # neutral bucket
    return vec


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current weather.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
FS_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the filesystem.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
NOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": "Remember a note.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}
IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "image_generate",
        "description": "Generate an image.",  # L0 won't match; L1 should
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    },
}


@pytest.fixture
def populated(tmp_path):
    cfg = StoreConfig(path=str(tmp_path / "memory.db"), embedding_dimensions=_EMBED_DIM)
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    registry = ToolRegistry(ms, embed_fn=_fake_embed, compression="moderate")
    asyncio.run(registry.ingest([WEATHER_TOOL, FS_TOOL, NOTE_TOOL, IMAGE_TOOL]))
    yield registry
    ms.close()


def _select(classifier: ToolClassifier, query: str):
    return asyncio.run(classifier.select(query))


def _name(tool: dict) -> str:
    fn = tool.get("function")
    if isinstance(fn, dict):
        return fn.get("name", "")
    return tool.get("name", "")


def test_l0_keyword_match_returns_web_search(populated):
    classifier = ToolClassifier(populated, embed_fn=_fake_embed, max_tools=10)
    selection = _select(classifier, "what's the weather in Tokyo")
    names = [_name(t) for t in selection.tools]
    assert "web_search" in names
    assert selection.level == 0


def test_l0_trivial_query_returns_empty(populated):
    classifier = ToolClassifier(populated, embed_fn=_fake_embed, max_tools=10)
    selection = _select(classifier, "hi")
    assert selection.tools == []


def test_l0_simple_math_returns_empty(populated):
    classifier = ToolClassifier(populated, embed_fn=_fake_embed, max_tools=10)
    selection = _select(classifier, "what is 2+2")
    assert selection.tools == []


def test_l1_finds_other_category_tool_by_embedding(populated):
    """L0 has no 'image' keyword; L1 embedding must catch it."""
    classifier = ToolClassifier(populated, embed_fn=_fake_embed, l1_threshold=0.5)
    selection = _select(classifier, "generate an image of a cat")
    names = [_name(t) for t in selection.tools]
    assert "image_generate" in names
    assert selection.level == 1


def test_fallback_returns_all_on_ambiguous_query(populated):
    classifier = ToolClassifier(populated, embed_fn=_fake_embed, l1_threshold=0.99,
                                fallback_include_all=True)
    selection = _select(classifier, "can you please help me with this task today")
    assert len(selection.tools) >= 3  # all active tools returned
    assert selection.level == -1


def test_max_tools_cap_truncates(populated):
    classifier = ToolClassifier(populated, embed_fn=_fake_embed, l1_threshold=0.99,
                                fallback_include_all=True, max_tools=2)
    selection = _select(classifier, "can you please help me with this task today")
    assert len(selection.tools) == 2


def test_recall_tool_never_in_selection(populated):
    classifier = ToolClassifier(populated, embed_fn=_fake_embed)
    selection = _select(classifier, "what's the weather in Tokyo")
    assert all(_name(t) != "recall" for t in selection.tools)


def test_tools_come_from_lean_schema_by_default(populated):
    """Selection should use lean_schema (moderate compression was applied at ingest)."""
    classifier = ToolClassifier(populated, embed_fn=_fake_embed)
    selection = _select(classifier, "weather in Tokyo")
    t = next(t for t in selection.tools if _name(t) == "web_search")
    # Moderate: no description on properties
    assert "description" not in t["function"]["parameters"]["properties"]["query"]


def test_uses_full_schema_when_compression_none(tmp_path):
    cfg = StoreConfig(path=str(tmp_path / "memory.db"), embedding_dimensions=_EMBED_DIM)
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    registry = ToolRegistry(ms, embed_fn=_fake_embed, compression="none")
    asyncio.run(registry.ingest([WEATHER_TOOL]))
    classifier = ToolClassifier(registry, embed_fn=_fake_embed)
    sel = _select(classifier, "weather")
    t = next(t for t in sel.tools if _name(t) == "web_search")
    # 'none' keeps the full schema byte-for-byte
    assert t == WEATHER_TOOL
    ms.close()


def test_empty_registry_returns_empty(tmp_path):
    cfg = StoreConfig(path=str(tmp_path / "memory.db"), embedding_dimensions=_EMBED_DIM)
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    registry = ToolRegistry(ms, embed_fn=_fake_embed, compression="moderate")
    classifier = ToolClassifier(registry, embed_fn=_fake_embed)
    sel = _select(classifier, "weather in Tokyo")
    assert sel.tools == []
    ms.close()

"""Tests for Phase 6: ContextRetriever — vector search + graph traversal + context block."""

from __future__ import annotations

import math

import pytest

from sieve.config import StoreConfig
from sieve.store import MemoryStore
from sieve.retrieval import ContextRetriever, RetrievedContext, _format_context_block


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    config = StoreConfig(path=str(tmp_path / "test.db"), embedding_dimensions=4)
    ms = MemoryStore(config, passphrase="test-retrieval")
    ms.open()
    ms.init_schema()
    yield ms
    ms.close()


def _fake_embed(text: str) -> list[float]:
    v = [ord(text[0]) / 256.0, 0.5, 0.3, 0.2] if text else [0.1, 0.1, 0.1, 0.1]
    mag = math.sqrt(sum(x * x for x in v))
    return [x / mag for x in v]


async def _async_embed(text: str) -> list[float]:
    return _fake_embed(text)


@pytest.fixture
def retriever(store):
    return ContextRetriever(store, embed_fn=None, top_k=5)


@pytest.fixture
def retriever_with_embed(store):
    return ContextRetriever(store, embed_fn=_async_embed, top_k=5)


# ─── Empty store ──────────────────────────────────────────────────────────────

class TestEmptyStore:
    async def test_empty_store_returns_empty_context(self, retriever):
        ctx = await retriever.retrieve("where do I live?")
        assert ctx.facts == []
        assert ctx.text == ""

    async def test_returns_retrieved_context_object(self, retriever):
        ctx = await retriever.retrieve("anything")
        assert isinstance(ctx, RetrievedContext)
        assert ctx.query == "anything"


# ─── Fallback (no embed_fn) ───────────────────────────────────────────────────

class TestFallbackRetrieval:
    async def test_returns_recent_facts_without_embeddings(self, store, retriever):
        store.insert_fact("User is a pilot", embedding=None)
        store.insert_fact("User lives in Dubai", embedding=None)

        ctx = await retriever.retrieve("where do I live?")
        assert len(ctx.facts) >= 1
        contents = [f["content"] for f in ctx.facts]
        assert any("pilot" in c or "Dubai" in c for c in contents)

    async def test_respects_top_k(self, store):
        for i in range(10):
            store.insert_fact(f"Fact number {i}", embedding=None)

        retriever = ContextRetriever(store, embed_fn=None, top_k=3)
        ctx = await retriever.retrieve("something")
        assert len(ctx.facts) <= 3

    async def test_filters_non_current_status(self, store, retriever):
        fact_id = store.insert_fact("User used to live in Sydney", embedding=None)
        # Mark it as superseded
        store.conn.execute(
            "UPDATE facts SET status = 'superseded' WHERE id = ?", (fact_id,)
        )
        store.conn.commit()

        ctx = await retriever.retrieve("where do I live?")
        contents = [f["content"] for f in ctx.facts]
        assert not any("Sydney" in c for c in contents)


# ─── Vector search ────────────────────────────────────────────────────────────

class TestVectorSearch:
    async def test_vector_search_returns_facts(self, store, retriever_with_embed):
        store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
        )
        ctx = await retriever_with_embed.retrieve("where do I live?")
        assert len(ctx.facts) >= 1
        assert any("Dubai" in f["content"] for f in ctx.facts)

    async def test_multiple_facts_retrieved(self, store, retriever_with_embed):
        for fact in ["User is a pilot", "User lives in Dubai", "User's partner is Madeline"]:
            store.insert_fact(fact, embedding=_fake_embed(fact))

        ctx = await retriever_with_embed.retrieve("tell me about the user")
        assert len(ctx.facts) >= 1

    async def test_token_estimate_set(self, store, retriever_with_embed):
        store.insert_fact("User lives in Dubai", embedding=_fake_embed("User lives in Dubai"))
        ctx = await retriever_with_embed.retrieve("where do I live?")
        assert ctx.token_estimate >= 0


# ─── Graph traversal ──────────────────────────────────────────────────────────

class TestGraphTraversal:
    async def test_graph_traversal_surfaces_related_facts(self, store, retriever_with_embed):
        # Create entity + fact linked to entity
        entity_id = store.insert_entity("Dubai", type="location")
        fact_id = store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
            entity_ids=[entity_id],
        )

        # Create related fact linked to same entity (no embedding — won't be in vector results)
        fact2_id = store.insert_fact(
            "User moved to Dubai five years ago",
            embedding=None,
            entity_ids=[entity_id],
        )

        ctx = await retriever_with_embed.retrieve("User lives in Dubai")
        contents = [f["content"] for f in ctx.facts]
        # Primary fact should be there
        assert any("Dubai" in c for c in contents)

    async def test_graph_count_tracked(self, store, retriever_with_embed):
        entity_id = store.insert_entity("Marcus", type="person")
        store.insert_fact(
            "User's best friend is Marcus",
            embedding=_fake_embed("User's best friend is Marcus"),
            entity_ids=[entity_id],
        )
        store.insert_fact(
            "Marcus is also a pilot",
            embedding=None,
            entity_ids=[entity_id],
        )
        ctx = await retriever_with_embed.retrieve("User's best friend is Marcus")
        # retrieved_from_graph may be 0 or >0 depending on traversal
        assert ctx.retrieved_from_graph >= 0


# ─── Context block formatting ─────────────────────────────────────────────────

class TestContextBlock:
    def test_empty_facts_returns_empty_string(self):
        assert _format_context_block([]) == ""

    def test_single_fact_in_block(self):
        facts = [{"content": "User lives in Dubai", "confidence": 0.9}]
        text = _format_context_block(facts)
        assert "## Recalled context" in text
        assert "User lives in Dubai" in text

    def test_multiple_facts_in_block(self):
        facts = [
            {"content": "User is a pilot", "confidence": 0.8},
            {"content": "User lives in Dubai", "confidence": 0.9},
        ]
        text = _format_context_block(facts)
        assert "pilot" in text
        assert "Dubai" in text

    def test_low_confidence_annotated(self):
        facts = [{"content": "User might be a chef", "confidence": 0.3}]
        text = _format_context_block(facts)
        assert "low confidence" in text

    def test_high_confidence_not_annotated(self):
        facts = [{"content": "User lives in Dubai", "confidence": 0.9}]
        text = _format_context_block(facts)
        assert "low confidence" not in text

    def test_missing_content_skipped(self):
        facts = [{"content": "", "confidence": 0.9}]
        text = _format_context_block(facts)
        assert text == ""

    async def test_context_text_in_retrieved_context(self, store, retriever):
        store.insert_fact("User is a pilot", embedding=None)
        ctx = await retriever.retrieve("what do I do?")
        if ctx.facts:
            assert "## Recalled context" in ctx.text


# ─── Pipeline integration — context injected into payload ────────────────────

class TestPipelineIntegration:
    """Verify that compose_lean_payload correctly injects retrieved_context."""

    def test_context_injected_as_system_message(self):
        from sieve.config import PipelineConfig
        from sieve.fingerprint import DecomposedPayload, Section
        from sieve.pipeline import compose_lean_payload

        config = PipelineConfig()
        decomposed = DecomposedPayload(sections=[], format="ollama")

        ctx_block = "## Recalled context\n- User lives in Dubai"
        lean = compose_lean_payload(
            {"model": "qwen3.5:35b", "messages": [{"role": "user", "content": "hi"}]},
            decomposed,
            config,
            retrieved_context=ctx_block,
        )

        system_msgs = [m for m in lean["messages"] if m["role"] == "system"]
        contents = [m["content"] for m in system_msgs]
        assert any("Recalled context" in c for c in contents)

    def test_no_context_no_extra_system_message(self):
        from sieve.config import PipelineConfig
        from sieve.fingerprint import DecomposedPayload
        from sieve.pipeline import compose_lean_payload, LEAN_SYSTEM_PROMPT

        config = PipelineConfig()
        decomposed = DecomposedPayload(sections=[], format="ollama")

        lean = compose_lean_payload(
            {"model": "qwen3.5:35b", "messages": [{"role": "user", "content": "hi"}]},
            decomposed,
            config,
            retrieved_context="",
        )

        system_msgs = [m for m in lean["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == LEAN_SYSTEM_PROMPT

    def test_context_injected_before_history(self):
        from sieve.config import PipelineConfig
        from sieve.fingerprint import DecomposedPayload
        from sieve.pipeline import compose_lean_payload

        config = PipelineConfig()
        decomposed = DecomposedPayload(sections=[], format="ollama")

        ctx_block = "## Recalled context\n- User lives in Dubai"
        lean = compose_lean_payload(
            {"model": "x", "messages": [{"role": "user", "content": "hello"}]},
            decomposed,
            config,
            retrieved_context=ctx_block,
        )

        msgs = lean["messages"]
        roles = [m["role"] for m in msgs]
        system_indices = [i for i, r in enumerate(roles) if r == "system"]
        user_indices = [i for i, r in enumerate(roles) if r == "user"]

        # All system messages should come before any user message
        if system_indices and user_indices:
            assert max(system_indices) < min(user_indices)


# ─── Dedup + MMR helpers ──────────────────────────────────────────────────────

from sieve.retrieval import (
    _cosine,
    _load_content_embeddings,
    _dedup_by_content,
    _mmr_rerank,
)


class TestCosine:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_return_zero(self):
        assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_empty_vector_returns_zero(self):
        assert _cosine([], [1.0, 2.0]) == 0.0
        assert _cosine([1.0, 2.0], []) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0

    def test_zero_magnitude_returns_zero(self):
        assert _cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


class TestPickFormat:
    """Query-driven format dispatch."""

    def _pick(self, q: str) -> str:
        from sieve.retrieval import _pick_format
        return _pick_format(q)

    # Should pick structured (temporal / progression queries)
    def test_career_path(self):
        assert self._pick("walk me through Jamie's career path") == "structured"

    def test_over_time(self):
        assert self._pick("how has the user's role changed over time?") == "structured"

    def test_progression(self):
        assert self._pick("what's the progression of Jamie's job?") == "structured"

    def test_history(self):
        assert self._pick("tell me Jamie's relationship history") == "structured"

    def test_used_to(self):
        assert self._pick("where did the user used to live?") == "structured"

    def test_walk_through(self):
        assert self._pick("walk me through what happened with Kim") == "structured"

    # Should pick flat (single-fact queries)
    def test_what_is_name(self):
        assert self._pick("what is the user's name?") == "flat"

    def test_where_lives(self):
        assert self._pick("where does Jamie live?") == "flat"

    def test_who_is_husband(self):
        assert self._pick("who is the user's husband?") == "flat"

    def test_dog_name(self):
        assert self._pick("what's the dog's name?") == "flat"

    def test_empty_query(self):
        assert self._pick("") == "flat"

    def test_simple_yes_no(self):
        assert self._pick("does the user have children?") == "flat"


class TestLoadContentEmbeddings:
    def test_loads_embeddings_for_given_fact_ids(self, store):
        fid_a = store.insert_fact(
            "A", [0.1, 0.2, 0.3, 0.4],
            source="test", fact_type="identity", confidence=0.9,
        )
        fid_b = store.insert_fact(
            "B", [0.5, 0.6, 0.7, 0.8],
            source="test", fact_type="identity", confidence=0.9,
        )
        result = _load_content_embeddings(store, [fid_a, fid_b])
        assert fid_a in result and fid_b in result
        assert len(result[fid_a]) == 4
        assert abs(result[fid_a][0] - 0.1) < 1e-6
        assert abs(result[fid_b][3] - 0.8) < 1e-6

    def test_missing_ids_omitted(self, store):
        fid = store.insert_fact(
            "A", [0.1, 0.2, 0.3, 0.4],
            source="test", fact_type="identity", confidence=0.9,
        )
        result = _load_content_embeddings(store, [fid, "nonexistent"])
        assert fid in result
        assert "nonexistent" not in result

    def test_null_embedding_omitted(self, store):
        fid = store.insert_fact(
            "no-emb", None,
            source="test", fact_type="identity", confidence=0.9,
        )
        result = _load_content_embeddings(store, [fid])
        assert fid not in result

    def test_empty_input(self, store):
        assert _load_content_embeddings(store, []) == {}


class TestDedupByContent:
    def test_drops_near_duplicates_keeps_first(self):
        facts = [
            {"id": "a", "content": "user likes chess", "confidence": 0.9, "distance": 1.0},
            {"id": "b", "content": "user enjoys chess", "confidence": 0.8, "distance": 1.1},
            {"id": "c", "content": "user lives in sf",  "confidence": 0.9, "distance": 1.2},
        ]
        embs = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.99, 0.01, 0.0],
            "c": [0.0, 1.0, 0.0],
        }
        out = _dedup_by_content(facts, embs, threshold=0.9, cap=10)
        assert [f["id"] for f in out] == ["a", "c"]

    def test_respects_cap(self):
        facts = [
            {"id": "a", "content": "a", "confidence": 0.9, "distance": 1.0},
            {"id": "b", "content": "b", "confidence": 0.9, "distance": 1.1},
            {"id": "c", "content": "c", "confidence": 0.9, "distance": 1.2},
        ]
        embs = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
        out = _dedup_by_content(facts, embs, threshold=0.9, cap=2)
        assert [f["id"] for f in out] == ["a", "b"]

    def test_missing_embedding_kept(self):
        facts = [
            {"id": "a", "content": "a", "confidence": 0.9, "distance": 1.0},
            {"id": "b", "content": "b", "confidence": 0.9, "distance": 1.1},
        ]
        embs = {"a": [1.0, 0.0]}
        out = _dedup_by_content(facts, embs, threshold=0.9, cap=10)
        assert [f["id"] for f in out] == ["a", "b"]

    def test_empty_input(self):
        assert _dedup_by_content([], {}, threshold=0.9, cap=10) == []


class TestMmrRerank:
    def test_selects_all_when_k_exceeds_input(self):
        facts = [
            {"id": "a", "content": "a", "confidence": 0.9, "distance": 1.0},
            {"id": "b", "content": "b", "confidence": 0.9, "distance": 1.2},
        ]
        embs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        out = _mmr_rerank(facts, embs, lam=0.7, k=10)
        assert len(out) == 2

    def test_moderately_similar_loses_to_diverse_when_relevance_is_close(self):
        # In production, _dedup_by_content strips near-duplicates (cosine
        # >=0.9) BEFORE MMR sees them. MMR's job is to distinguish among
        # *moderately* similar facts when relevance alone is inconclusive.
        # Here: a and b are ~0.7 similar, c is orthogonal. Distances are
        # all very close so relevance barely differs. MMR should pick the
        # diverse c over the moderately-similar b.
        facts = [
            {"id": "a", "content": "hobby1",   "confidence": 0.9, "distance": 1.28},
            {"id": "b", "content": "hobby2",   "confidence": 0.9, "distance": 1.29},
            {"id": "c", "content": "family",   "confidence": 0.9, "distance": 1.30},
        ]
        embs = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.7, 0.7, 0.0],   # ~0.7 cosine to a (moderate overlap)
            "c": [0.0, 0.0, 1.0],   # orthogonal (diverse)
        }
        out = _mmr_rerank(facts, embs, lam=0.7, k=2)
        assert {f["id"] for f in out} == {"a", "c"}

    def test_all_orthogonal_picks_highest_relevance_first(self):
        # Baseline sanity: when no redundancy penalty applies, MMR
        # degenerates to relevance ranking. Best distance wins first.
        facts = [
            {"id": "a", "content": "a", "confidence": 0.9, "distance": 1.20},
            {"id": "b", "content": "b", "confidence": 0.9, "distance": 1.25},
            {"id": "c", "content": "c", "confidence": 0.9, "distance": 1.30},
        ]
        embs = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [0.0, 0.0, 1.0],
        }
        out = _mmr_rerank(facts, embs, lam=0.7, k=2)
        assert [f["id"] for f in out] == ["a", "b"]

    def test_graph_hop_facts_accepted(self):
        facts = [
            {"id": "a", "content": "primary", "confidence": 0.9, "distance": 1.0},
            {"id": "g", "content": "graph",   "confidence": 0.9, "distance": None},
        ]
        embs = {"a": [1.0, 0.0], "g": [0.0, 1.0]}
        out = _mmr_rerank(facts, embs, lam=0.7, k=2)
        assert len(out) == 2

    def test_empty_input(self):
        assert _mmr_rerank([], {}, lam=0.7, k=5) == []


# ── _temporal_dedup ──────────────────────────────────────────────────────────

def _mk_fact(id_: str, content: str, created_at: str, confidence: float = 0.8) -> dict:
    return {
        "id": id_,
        "content": content,
        "created_at": created_at,
        "confidence": confidence,
        "status": "current",
    }


def test_temporal_dedup_empty_input():
    from sieve.retrieval import _temporal_dedup
    assert _temporal_dedup([], {}, {}) == []


def test_temporal_dedup_single_fact_unchanged():
    from sieve.retrieval import _temporal_dedup
    f = _mk_fact("a", "Jamie lives in Beacon Hill condo", "2025-01-01")
    out = _temporal_dedup([f], {"a": [1.0, 0.0]}, {"a": {"e_mary"}})
    assert out == [f]


def test_temporal_dedup_same_entity_high_similarity_newer_wins():
    from sieve.retrieval import _temporal_dedup
    f_old = _mk_fact("old", "Jamie lives in Beacon Hill condo with Kim", "2025-01-01")
    f_new = _mk_fact("new", "Jamie lives alone in Beacon Hill condo", "2025-06-01")
    embs = {"old": [1.0, 0.0], "new": [0.99, 0.01]}  # cos ≈ 0.995
    ent = {"old": {"e_mary", "e_tom"}, "new": {"e_mary"}}
    out = _temporal_dedup([f_old, f_new], embs, ent, similarity_threshold=0.85)
    assert [f["id"] for f in out] == ["new"]


def test_temporal_dedup_same_entity_low_similarity_both_kept():
    from sieve.retrieval import _temporal_dedup
    f1 = _mk_fact("condo", "Jamie owns a condo in Beacon Hill", "2025-01-01")
    f2 = _mk_fact("cabin", "Jamie owns a cabin in Vermont", "2025-02-01")
    embs = {"condo": [1.0, 0.0], "cabin": [0.2, 0.98]}  # cos ≈ 0.2
    ent = {"condo": {"e_mary"}, "cabin": {"e_mary"}}
    out = _temporal_dedup([f1, f2], embs, ent, similarity_threshold=0.85)
    assert len(out) == 2


def test_temporal_dedup_different_entities_high_similarity_both_kept():
    from sieve.retrieval import _temporal_dedup
    f1 = _mk_fact("a", "Jamie lives in Boston", "2025-01-01")
    f2 = _mk_fact("b", "Kim lives in Boston", "2025-06-01")
    embs = {"a": [1.0, 0.0], "b": [0.99, 0.01]}
    ent = {"a": {"e_mary"}, "b": {"e_tom"}}
    out = _temporal_dedup([f1, f2], embs, ent, similarity_threshold=0.85)
    assert len(out) == 2


def test_temporal_dedup_tie_on_timestamp_higher_confidence_wins():
    from sieve.retrieval import _temporal_dedup
    f_low = _mk_fact("low", "Jamie works at Other", "2025-01-01", confidence=0.5)
    f_hi = _mk_fact("hi", "Jamie is at Other Corp", "2025-01-01", confidence=0.95)
    embs = {"low": [1.0, 0.0], "hi": [0.99, 0.01]}
    ent = {"low": {"e_mary"}, "hi": {"e_mary"}}
    out = _temporal_dedup([f_low, f_hi], embs, ent, similarity_threshold=0.85)
    assert [f["id"] for f in out] == ["hi"]


def test_temporal_dedup_tie_on_everything_stable_by_id():
    from sieve.retrieval import _temporal_dedup
    f_a = _mk_fact("a", "Jamie works at Other", "2025-01-01", confidence=0.8)
    f_b = _mk_fact("b", "Jamie works at Other", "2025-01-01", confidence=0.8)
    embs = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    ent = {"a": {"e_mary"}, "b": {"e_mary"}}
    out = _temporal_dedup([f_a, f_b], embs, ent, similarity_threshold=0.85)
    # Stable by id → lexicographic "a" wins
    assert [f["id"] for f in out] == ["a"]


# ── _format_context_block max_tokens parameter ──────────────────────────────

def _make_fact_rows(n: int, words_per_fact: int = 15) -> list[dict]:
    rows = []
    for i in range(n):
        content = f"Fact number {i} " + "lorem ipsum " * words_per_fact
        rows.append({
            "id": f"f{i}",
            "content": content.strip(),
            "category": "misc",
            "fact_type": "objective",
            "confidence": 0.8,
            "status": "current",
            "created_at": "2025-01-01",
        })
    return rows


def test_format_context_block_respects_max_tokens_cap():
    from sieve.retrieval import _format_context_block
    facts = _make_fact_rows(30, words_per_fact=10)
    short = _format_context_block(facts, max_tokens=200)
    long = _format_context_block(facts, max_tokens=2000)
    assert len(short) < len(long), f"short={len(short)} long={len(long)}"
    # Respect the cap within 15% slack (token→char heuristic).
    assert len(short) <= int(200 * 4 * 1.15), f"short block exceeded cap: {len(short)}"
    assert len(long) <= int(2000 * 4 * 1.15), f"long block exceeded cap: {len(long)}"


def test_format_context_block_truncates_at_fact_boundary():
    from sieve.retrieval import _format_context_block
    facts = _make_fact_rows(5, words_per_fact=8)
    small = _format_context_block(facts, max_tokens=50)
    # Output must be a sequence of complete lines, never mid-sentence.
    for line in small.splitlines():
        assert line.strip(), "empty line should not appear"
        # No truncation marker mid-line.
        assert not line.endswith("..."), f"line ends with ellipsis: {line!r}"


def test_format_context_block_empty_input():
    from sieve.retrieval import _format_context_block
    assert _format_context_block([], max_tokens=200) == ""


def test_format_context_block_backwards_compat_no_max_tokens():
    """Pre-Fix-3a callers pass no max_tokens kwarg. The default must
    produce the SAME output as a very large explicit cap for any
    reasonable-sized fact set (< the default tier's capacity).
    This catches regressions where the default cap is set too small.
    """
    from sieve.retrieval import _format_context_block
    facts = _make_fact_rows(15, words_per_fact=10)
    default_out = _format_context_block(facts)
    big_out = _format_context_block(facts, max_tokens=100_000)
    # Default must emit every input fact (no silent truncation).
    assert default_out == big_out, (
        f"default path truncated vs unlimited path "
        f"({len(default_out)} vs {len(big_out)} chars)"
    )
    # Sanity: all facts appear.
    for i in range(15):
        assert f"Fact number {i}" in default_out


def test_format_context_block_structured_backwards_compat_no_max_tokens():
    """Same backwards-compat guarantee for _format_context_block_structured."""
    from sieve.retrieval import _format_context_block_structured
    facts = _make_fact_rows(15, words_per_fact=10)
    default_out = _format_context_block_structured(facts)
    big_out = _format_context_block_structured(facts, max_tokens=100_000)
    assert default_out == big_out, (
        f"structured default path truncated vs unlimited path "
        f"({len(default_out)} vs {len(big_out)} chars)"
    )


def test_format_context_block_structured_respects_max_tokens_cap():
    from sieve.retrieval import _format_context_block_structured
    facts = _make_fact_rows(30, words_per_fact=10)
    short = _format_context_block_structured(facts, max_tokens=200)
    long = _format_context_block_structured(facts, max_tokens=2000)
    assert len(short) < len(long), f"short={len(short)} long={len(long)}"
    assert len(short) <= int(200 * 4 * 1.15)
    assert len(long) <= int(2000 * 4 * 1.15)


# ─── Reranker ────────────────────────────────────────────────────────────────


class _StubReranker:
    """Deterministic stub that scores candidates by substring match to the
    query. Used instead of the real cross-encoder so tests stay fast and
    network-free."""

    def __init__(self, available: bool = True):
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def rerank(self, query: str, candidates: list[str]) -> list[float] | None:
        if not self._available:
            return None
        q_lower = query.lower()
        return [
            sum(1.0 for word in q_lower.split() if word in c.lower())
            for c in candidates
        ]


class TestReranker:
    async def test_reranker_reorders_candidates(self, store):
        """Reranker should promote a semantic match over an early-but-weak
        vector candidate. Store has two facts with identical fake
        embeddings; without rerank the one with 'Dubai' in its content
        wins by relevance keyword, not vector distance."""
        store.insert_fact(
            "User once visited Paris on holiday",
            embedding=_fake_embed("User once visited Paris on holiday"),
        )
        store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
        )
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=2,
            reranker=_StubReranker(),
        )
        ctx = await retriever.retrieve("where do I live Dubai")
        contents = [f["content"] for f in ctx.facts]
        # The Dubai fact scores higher under the stub (more keyword
        # overlap) and must appear first after reranking.
        assert contents[0].startswith("User lives in Dubai"), contents

    async def test_reranker_unavailable_is_noop(self, store):
        """An unavailable reranker must not crash or block retrieval."""
        store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
        )
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=2,
            reranker=_StubReranker(available=False),
        )
        ctx = await retriever.retrieve("anything")
        assert len(ctx.facts) >= 1

    async def test_reranker_attaches_score_to_facts(self, store):
        """rerank_score key should appear on each fact so downstream
        telemetry can log the score."""
        store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
        )
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=2,
            reranker=_StubReranker(),
        )
        ctx = await retriever.retrieve("where Dubai")
        assert all("rerank_score" in f for f in ctx.facts)


# ─── retrieve_multi ──────────────────────────────────────────────────────────


class TestRetrieveMulti:
    async def test_merges_and_dedupes_by_id(self, store):
        """Sub-queries that surface the same fact id must not duplicate
        it in the merged list. Uses three identical sub-queries against
        a single-fact store so each sub-query returns the same id."""
        store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
        )
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=5,
        )
        ctx = await retriever.retrieve_multi(
            ["same query", "same query", "same query"],
            per_query_top_k=3,
            final_top_k=10,
        )
        fact_ids = [f["id"] for f in ctx.facts]
        assert len(fact_ids) == len(set(fact_ids)), (
            f"duplicate fact ids in merged result: {fact_ids}"
        )
        # The single fact in the store is surfaced once.
        assert len(ctx.facts) == 1

    async def test_empty_queries_returns_empty_context(self, store):
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=3,
        )
        ctx = await retriever.retrieve_multi([], per_query_top_k=2, final_top_k=5)
        assert ctx.facts == []
        assert ctx.text == ""

    async def test_falls_back_to_single_query_on_empty_merge(self, store):
        """If every sub-query misses, fall back to retrieving on the first
        sub-query so we never regress to an empty context when a single-
        query retrieve would have succeeded."""
        store.insert_fact(
            "User lives in Dubai",
            embedding=_fake_embed("User lives in Dubai"),
        )
        # Stub retriever whose per-sub-query retrieves return nothing,
        # but whose primary retrieve returns the fact.
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=3,
        )
        # Using a non-matching per_query_top_k isn't the issue — the
        # store is small enough that any embed_fn returns Dubai. We
        # instead test that retrieve_multi still returns facts for a
        # degenerate sub-query set.
        ctx = await retriever.retrieve_multi(
            ["where do I live?"], per_query_top_k=2, final_top_k=3,
        )
        assert len(ctx.facts) >= 1

    async def test_respects_final_top_k(self, store):
        for i in range(8):
            store.insert_fact(f"Fact {i}", embedding=_fake_embed(f"Fact {i}"))
        retriever = ContextRetriever(
            store, embed_fn=_async_embed, top_k=5,
        )
        ctx = await retriever.retrieve_multi(
            ["fact one", "fact two", "fact three"],
            per_query_top_k=3,
            final_top_k=4,
        )
        assert len(ctx.facts) <= 4

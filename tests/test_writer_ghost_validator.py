"""Tests for the post-S2 ghost-fact validator."""

import pytest
from sieve.writer import ExtractedFact, _validate_s2_fact


def _make_fact(content: str, entities: list[str] | None = None,
               category: str = "identity", fact_type: str = "objective") -> ExtractedFact:
    return ExtractedFact(
        content=content,
        entity_names=entities or [],
        category=category,
        fact_type=fact_type,
        confidence=0.8,
    )


# ── Rule 1: identity collision — possessive spouse form ─────────────────────

def test_rejects_user_is_other_persons_spouse():
    fact = _make_fact("User is Priya's husband", entities=["Priya"], category="relationship")
    keep, reason = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie", "I", "me"])
    assert keep is False
    assert reason == "identity"


def test_rejects_mary_is_danas_husband():
    fact = _make_fact("Jamie Rivera is Dana's husband", entities=["Dana"], category="relationship")
    keep, _ = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is False


def test_allows_mary_is_danas_colleague():
    fact = _make_fact("Jamie Rivera is Dana's colleague", entities=["Dana"], category="relationship")
    keep, _ = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is True


# ── Rule 1: identity collision — relative form ──────────────────────────────

def test_rejects_mary_is_twin_brother_of_someone():
    fact = _make_fact("Jamie Rivera is the twin brother of Priya", entities=["Priya"], category="relationship")
    keep, _ = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is False


def test_rejects_mary_is_cousin_named_dana():
    fact = _make_fact("Jamie Rivera is a cousin named Dana", entities=["Dana"], category="relationship")
    keep, _ = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is False


# ── Rule 3: duplicate-name entity ────────────────────────────────────────────

def test_rejects_twin_brother_named_jamie():
    fact = _make_fact(
        "User has a twin brother named Jamie", entities=["Jamie"],
        category="relationship",
    )
    keep, reason = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is False
    assert reason == "duplicate"


def test_allows_unrelated_person_named_differently():
    fact = _make_fact(
        "User has a colleague named Robin", entities=["Robin"],
        category="relationship",
    )
    keep, _ = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is True


# ── Empty owner (backwards-compat) ───────────────────────────────────────────

def test_empty_owner_passes_everything_through():
    fact = _make_fact("User is Priya's husband", entities=["Priya"])
    keep, _ = _validate_s2_fact(fact, "", [])
    assert keep is True  # no owner → no pinning → no filter


# ── Fix-pass regression tests (code-review findings) ────────────────────────

def test_rule3a_does_not_trigger_on_verb_called():
    """'we called Jamie to discuss' is a phone-call verb, not a naming.
    Rule 3a must not drop legitimate facts on this phrasing."""
    fact = _make_fact(
        "User called Jamie to discuss the project", entities=["Jamie"],
        category="relationship",
    )
    keep, _ = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is True


# ── Integration: Writer.process filters ghost facts end-to-end ───────────────

def test_writer_process_filters_ghost_fact_end_to_end(monkeypatch, tmp_path):
    """S2 emits a ghost; MemoryWriter.process drops it before S3 dedup/write."""
    import asyncio
    from sieve.writer import MemoryWriter, ExtractedFact
    from sieve.store import MemoryStore, StoreConfig

    config = StoreConfig(path=str(tmp_path / "mem.db"), embedding_dimensions=4)
    store = MemoryStore(config, passphrase="test-ghost-validator")
    store.open()
    store.init_schema()

    # Stub S2 to return a known ghost (Rule 1a: possessive-spouse)
    async def fake_s2(text, provider_base_url, **kwargs):
        return [ExtractedFact(
            content="User is Priya's husband",
            entity_names=["Priya"],
            category="relationship",
            fact_type="objective",
            confidence=0.9,
        )]

    monkeypatch.setattr("sieve.writer.extract_facts_s2", fake_s2)

    w = MemoryWriter(
        store,
        embed_fn=None,
        provider_base_url="http://localhost:11434",
        writer_model="qwen3.5:2b",
        fallback_model="qwen2.5:1.5b",
        num_ctx=4096,
        owner_name="Jamie Rivera",
        profile_owner_aliases=["Jamie", "I", "me"],
        ghost_validator_enabled=True,
    )

    # Plausibly-long first-person message to open the S2 gate.
    asyncio.run(w.process(
        "I was talking to Priya yesterday about the new project and "
        "we went for coffee together afterward to discuss the details.",
        assistant_text="",
    ))

    # The ghost fact should be filtered → no new facts about "Priya's husband"
    # in the store.
    rows = store.conn.execute(
        "SELECT content FROM facts WHERE content LIKE '%husband%'"
    ).fetchall()
    store.close()
    assert len(rows) == 0, f"ghost fact leaked into store: {rows}"


# ── Rule 4: relative-cohabitation ────────────────────────────────────────────

def test_rule4_rejects_lives_with_known_child():
    fact = _make_fact(
        "Jamie Rivera lives with a neighbor named Pat",
        entities=["Pat"], category="residence",
    )
    keep, reason = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie"], relatives={"pat", "alex"},
    )
    assert keep is False
    assert reason == "relative_cohabitation"


def test_rule4_rejects_lives_with_spouse():
    fact = _make_fact(
        "Jamie Rivera lives with Kim",
        entities=["Kim"], category="residence",
    )
    keep, reason = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie"], relatives={"kim"},
    )
    assert keep is False
    assert reason == "relative_cohabitation"


def test_rule4_rejects_resides_with_sibling():
    fact = _make_fact(
        "Jamie resides with Dana",
        entities=["Dana"], category="residence",
    )
    keep, reason = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie"], relatives={"dana"},
    )
    assert keep is False
    assert reason == "relative_cohabitation"


def test_rule4_allows_lives_with_nonrelative():
    fact = _make_fact(
        "Jamie Rivera lives with Amanda",
        entities=["Amanda"], category="residence",
    )
    keep, reason = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie"], relatives={"pat", "alex", "kim"},
    )
    assert keep is True
    assert reason is None


def test_rule4_no_op_when_relatives_empty():
    """With no known relatives (early reseed), Rule 4 is a pass-through."""
    fact = _make_fact(
        "Jamie Rivera lives with Pat",
        entities=["Pat"], category="residence",
    )
    keep, reason = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie"], relatives=set(),
    )
    assert keep is True
    assert reason is None


def test_rule4_no_op_when_relatives_none():
    """Legacy callers that don't pass relatives= still work unchanged."""
    fact = _make_fact(
        "Jamie Rivera lives with Pat",
        entities=["Pat"], category="residence",
    )
    keep, reason = _validate_s2_fact(fact, "Jamie Rivera", ["Jamie"])
    assert keep is True
    assert reason is None


def test_rule4_allows_lives_in_place():
    """'Jamie Rivera lives in Boston' should not match 'lives with X'."""
    fact = _make_fact(
        "Jamie Rivera lives in Boston",
        category="residence",
    )
    keep, _ = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie"], relatives={"pat", "kim"},
    )
    assert keep is True


def test_rule4_handles_user_alias():
    """Rule 4 should fire on 'User lives with Pat' too, not just 'Jamie'."""
    fact = _make_fact(
        "User lives with Pat",
        entities=["Pat"], category="residence",
    )
    keep, reason = _validate_s2_fact(
        fact, "Jamie Rivera", ["Jamie", "I", "me"], relatives={"pat"},
    )
    assert keep is False
    assert reason == "relative_cohabitation"

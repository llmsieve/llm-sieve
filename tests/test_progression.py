"""Tests for the progressive-activation subsystem.

Progressive activation selects one of three phases — OBSERVE / ACCUMULATE
/ ACTIVATE — based on the current count of facts in the store. Each
phase specifies how many recent conversation turns the composer keeps.

Three layers of test:

1. Config loading — ``ProgressionConfig`` dataclass + YAML round-trip.
2. Pure phase detection — ``detect_phase(fact_count, config)``.
3. Composer integration — phase overrides ``conversation_turns`` when
   ``compose_lean_payload`` is called with a progression override.
"""

from __future__ import annotations

import json

import pytest
import yaml

from sieve.config import PipelineConfig, ProgressionConfig, RecallConfig
from sieve.fingerprint import FingerprintCache, decompose
from sieve.pipeline import compose_lean_payload
from sieve.progression import Phase, PhaseDecision, detect_phase


# ── ProgressionConfig ─────────────────────────────────────────────────────


def test_progression_config_defaults():
    cfg = RecallConfig()
    assert isinstance(cfg.progression, ProgressionConfig)
    assert cfg.progression.phase_1_threshold == 20
    assert cfg.progression.phase_2_threshold == 50
    assert cfg.progression.observe_turns == 8
    assert cfg.progression.accumulate_turns == 4
    assert cfg.progression.activate_turns == 2


def test_progression_config_loads_from_yaml(tmp_path):
    path = tmp_path / "sieve.yaml"
    path.write_text(yaml.safe_dump({
        "progression": {
            "phase_1_threshold": 10,
            "phase_2_threshold": 40,
            "observe_turns": 6,
            "accumulate_turns": 3,
            "activate_turns": 1,
        }
    }))
    cfg = RecallConfig.load(path)
    assert cfg.progression.phase_1_threshold == 10
    assert cfg.progression.phase_2_threshold == 40
    assert cfg.progression.observe_turns == 6
    assert cfg.progression.accumulate_turns == 3
    assert cfg.progression.activate_turns == 1


def test_progression_config_invalid_thresholds_coerce_to_defaults(tmp_path, caplog):
    """phase_2 must be >= phase_1, and all counts must be positive.

    Invalid YAML warns and keeps defaults, same pattern as
    tools.compression and pipeline.context_format.
    """
    path = tmp_path / "sieve.yaml"
    path.write_text(yaml.safe_dump({
        "progression": {
            "phase_1_threshold": 100,
            "phase_2_threshold": 50,   # below phase_1 — invalid
        }
    }))
    cfg = RecallConfig.load(path)
    assert cfg.progression.phase_1_threshold == 20
    assert cfg.progression.phase_2_threshold == 50


# ── Phase detection ──────────────────────────────────────────────────────


def _default_progression():
    return ProgressionConfig()


def test_detect_phase_observe_at_zero():
    d = detect_phase(0, _default_progression())
    assert d.phase is Phase.OBSERVE
    assert d.turns == 8
    assert d.fact_count == 0


def test_detect_phase_observe_just_below_boundary():
    d = detect_phase(19, _default_progression())
    assert d.phase is Phase.OBSERVE
    assert d.turns == 8


def test_detect_phase_accumulate_at_boundary():
    """`>= phase_1_threshold` enters ACCUMULATE."""
    d = detect_phase(20, _default_progression())
    assert d.phase is Phase.ACCUMULATE
    assert d.turns == 4


def test_detect_phase_accumulate_middle():
    d = detect_phase(35, _default_progression())
    assert d.phase is Phase.ACCUMULATE
    assert d.turns == 4


def test_detect_phase_activate_at_boundary():
    """`>= phase_2_threshold` enters ACTIVATE."""
    d = detect_phase(50, _default_progression())
    assert d.phase is Phase.ACTIVATE
    assert d.turns == 2


def test_detect_phase_activate_mature_store():
    d = detect_phase(500, _default_progression())
    assert d.phase is Phase.ACTIVATE
    assert d.turns == 2


def test_detect_phase_custom_config():
    cfg = ProgressionConfig(
        phase_1_threshold=5,
        phase_2_threshold=10,
        observe_turns=12,
        accumulate_turns=6,
        activate_turns=1,
    )
    assert detect_phase(0, cfg).turns == 12
    assert detect_phase(5, cfg).turns == 6
    assert detect_phase(10, cfg).turns == 1


def test_phase_decision_label_matches_phase_name():
    d = detect_phase(0, _default_progression())
    assert d.label == "OBSERVE"


def test_phase_decision_format_for_log():
    d = detect_phase(35, _default_progression())
    # Used in proxy logs: "[ACCUMULATE: 35 facts]"
    assert d.render_tag() == "[ACCUMULATE: 35 facts]"


def test_detect_phase_rejects_negative_fact_count():
    with pytest.raises(ValueError):
        detect_phase(-1, _default_progression())


# ── Composer integration ─────────────────────────────────────────────────


def _payload_with_history(n_user_turns: int) -> dict:
    """Build a chat payload with n prior user/assistant pairs + a new user turn."""
    msgs: list[dict] = [{"role": "system", "content": "you are an assistant"}]
    for i in range(n_user_turns):
        msgs.append({"role": "user", "content": f"Q{i}"})
        msgs.append({"role": "assistant", "content": f"A{i}"})
    msgs.append({"role": "user", "content": "latest"})
    return {"model": "qwen3:14b", "messages": msgs}


def _count_user_messages(lean: dict) -> int:
    return sum(1 for m in lean.get("messages", []) if m.get("role") == "user")


def test_compose_observe_phase_keeps_8_turns():
    """Under OBSERVE, composer keeps up to observe_turns user/assistant pairs."""
    payload = _payload_with_history(10)
    cache = FingerprintCache(store=None)
    decomposed = decompose(payload, cache, api_format="ollama")
    pipeline_cfg = PipelineConfig(conversation_turns=3)  # old knob — irrelevant now
    prog_decision = detect_phase(0, _default_progression())

    lean = compose_lean_payload(
        payload, decomposed, pipeline_cfg,
        progression=prog_decision,
    )
    # 8 prior user turns kept + 1 current user turn = 9
    assert _count_user_messages(lean) == 9


def test_compose_accumulate_phase_keeps_4_turns():
    payload = _payload_with_history(10)
    cache = FingerprintCache(store=None)
    decomposed = decompose(payload, cache, api_format="ollama")
    pipeline_cfg = PipelineConfig(conversation_turns=3)
    prog_decision = detect_phase(25, _default_progression())

    lean = compose_lean_payload(
        payload, decomposed, pipeline_cfg,
        progression=prog_decision,
    )
    assert _count_user_messages(lean) == 5  # 4 prior + 1 current


def test_compose_activate_phase_keeps_2_turns():
    payload = _payload_with_history(10)
    cache = FingerprintCache(store=None)
    decomposed = decompose(payload, cache, api_format="ollama")
    pipeline_cfg = PipelineConfig(conversation_turns=3)
    prog_decision = detect_phase(100, _default_progression())

    lean = compose_lean_payload(
        payload, decomposed, pipeline_cfg,
        progression=prog_decision,
    )
    assert _count_user_messages(lean) == 3  # 2 prior + 1 current


def test_compose_without_progression_override_uses_pipeline_turns():
    """Back-compat: omitting the progression arg preserves today's behaviour."""
    payload = _payload_with_history(10)
    cache = FingerprintCache(store=None)
    decomposed = decompose(payload, cache, api_format="ollama")
    pipeline_cfg = PipelineConfig(conversation_turns=3)

    lean = compose_lean_payload(payload, decomposed, pipeline_cfg)
    assert _count_user_messages(lean) == 4  # 3 prior + 1 current


# ── Store integration ──────────────────────────────────────────────────


def test_store_count_current_facts_empty(tmp_path, monkeypatch):
    """Fresh store has zero current facts."""
    from sieve.config import StoreConfig
    from sieve.store import MemoryStore

    cfg = StoreConfig(path=str(tmp_path / "mem.db"))
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    try:
        assert ms.count_current_facts() == 0
    finally:
        ms.close()


def test_store_count_current_facts_ignores_superseded(tmp_path):
    """Only status='current' counts. Superseded/retracted facts are excluded."""
    from sieve.config import StoreConfig
    from sieve.store import MemoryStore

    cfg = StoreConfig(path=str(tmp_path / "mem.db"))
    ms = MemoryStore(cfg)
    ms.open()
    ms.init_schema()
    try:
        # Insert three facts directly; leave status default ('current').
        now = "2026-04-20T00:00:00Z"
        for i, fid in enumerate(("f1", "f2", "f3")):
            ms.conn.execute(
                "INSERT INTO facts(id, content, status, source, confidence, created_at, fact_type) "
                "VALUES (?, ?, 'current', 'test', 1.0, ?, 'general')",
                (fid, f"fact {i}", now),
            )
        # Mark one as superseded.
        ms.conn.execute("UPDATE facts SET status='superseded' WHERE id='f2'")
        ms.conn.commit()
        assert ms.count_current_facts() == 2
    finally:
        ms.close()

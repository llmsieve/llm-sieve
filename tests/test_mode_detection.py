"""Tests for Mode A/B auto-detection."""

from sieve.config import detect_mode, AblationConfig


def test_mode_a_known_model():
    assert detect_mode("qwen3.5:9b") == "A"
    assert detect_mode("qwen3.5:35b") == "A"
    assert detect_mode("llama3") == "A"


def test_mode_b_known_model():
    assert detect_mode("qwen3.5:0.5b") == "B"
    assert detect_mode("qwen2.5:1.5b") == "B"


def test_mode_override_a():
    assert detect_mode("qwen3.5:0.5b", override="A") == "A"


def test_mode_override_b():
    assert detect_mode("qwen3.5:35b", override="B") == "B"


def test_mode_auto_unknown_defaults_a():
    assert detect_mode("some-unknown-model:7b") == "A"


def test_mode_cpu_suffix_stripped():
    assert detect_mode("qwen3.5:35b-cpu") == "A"


def test_ablation_defaults():
    cfg = AblationConfig()
    assert cfg.fingerprinting is True
    assert cfg.classifier is True
    assert cfg.pre_populate is True
    assert cfg.graph_traversal is True
    assert cfg.recall_tool is True
    assert cfg.stage2_writer is True
    assert cfg.learning_loop is True


def test_ablation_all_off():
    cfg = AblationConfig(
        fingerprinting=False, classifier=False, pre_populate=False,
        graph_traversal=False, temporal_versioning=False, learning_loop=False,
        coherence_integrity=False, stage2_writer=False, recall_tool=False,
    )
    assert all(not getattr(cfg, f) for f in [
        "fingerprinting", "classifier", "pre_populate", "graph_traversal",
        "temporal_versioning", "learning_loop", "coherence_integrity",
        "stage2_writer", "recall_tool",
    ])


# ── Ablation wiring verification tests ──────────────────────────────────────

def test_abl_fp_ephemeral_cache():
    """ABL-FP: when fingerprinting disabled, FingerprintCache with no store."""
    from sieve.fingerprint import FingerprintCache
    cache = FingerprintCache(None)
    # First check always returns True (changed)
    assert cache.check_and_update("test", "hash1") is True
    # Second check with same hash returns False (unchanged) even in ephemeral mode
    assert cache.check_and_update("test", "hash1") is False
    # New ephemeral cache loses state
    cache2 = FingerprintCache(None)
    assert cache2.check_and_update("test", "hash1") is True


def test_abl_gr_graph_traversal_disabled():
    """ABL-GR: graph_traversal=False should skip graph traversal."""
    from sieve.retrieval import ContextRetriever
    from unittest.mock import MagicMock
    store = MagicMock()
    store._conn = True
    retriever = ContextRetriever(store, graph_traversal=False)
    assert retriever._graph_traversal is False


def test_abl_tv_temporal_versioning_disabled():
    """ABL-TV: temporal_versioning=False should include all facts."""
    from sieve.retrieval import ContextRetriever
    from unittest.mock import MagicMock
    store = MagicMock()
    store._conn = True
    retriever = ContextRetriever(store, temporal_versioning=False)
    assert retriever._temporal_versioning is False


def test_abl_s2_writer_disabled():
    """ABL-S2: stage2_enabled=False should skip LLM extraction."""
    from sieve.writer import MemoryWriter
    from unittest.mock import MagicMock
    store = MagicMock()
    writer = MemoryWriter(store, stage2_enabled=False)
    assert writer._stage2_enabled is False


def test_abl_ci_coherence_disabled():
    """ABL-CI: coherence_enabled=False should skip coherence scoring."""
    from sieve.writer import MemoryWriter
    from unittest.mock import MagicMock
    store = MagicMock()
    writer = MemoryWriter(store, coherence_enabled=False)
    assert writer._coherence_enabled is False


def test_ablation_config_from_yaml():
    """Ablation config should parse from YAML dict."""
    from sieve.config import RecallConfig
    cfg = RecallConfig.load()
    # Defaults should all be True
    assert cfg.ablation.fingerprinting is True
    assert cfg.ablation.temporal_versioning is True
    assert cfg.ablation.coherence_integrity is True

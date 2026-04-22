"""Configuration loader for Recall proxy."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml

DEFAULT_CONFIG_PATHS = [
    Path("sieve.yaml"),
    Path("~/.sieve/sieve.yaml").expanduser(),
]


@dataclass
class ListenConfig:
    host: str = "0.0.0.0"
    port: int = 11435


@dataclass
class ProviderConfig:
    type: str = "auto"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str | None = None
    default_model: str = "qwen3.5:35b"
    options: dict = field(default_factory=dict)


@dataclass
class StoreConfig:
    path: str = "~/.sieve/memory.db"
    # NOTE: embedding_model / embedding_dimensions are legacy fields from
    # the Ollama-only era. They are still honored when
    # EmbeddingsConfig.provider=="ollama" but are overridden for
    # provider=="fastembed" (384-dim BAAI/bge-small-en-v1.5). The store
    # pulls its effective dimension from the embedding backend at
    # schema-creation time — see MemoryStore.init_schema.
    embedding_model: str = "nomic-embed-text-v2-moe"
    embedding_dimensions: int = 768


@dataclass
class EmbeddingsConfig:
    """Which embedding backend to use.

    ``fastembed`` (default) is self-contained: uses BAAI/bge-small-en-v1.5
    via ONNX Runtime, 384-dim, ~50MB download at init time, no external
    service required. ``ollama`` preserves the legacy path for users
    who operate their own Ollama-based embedding pipeline.
    """
    provider: str = "fastembed"      # "fastembed" | "ollama"
    # Ollama-path overrides. When unset, the EmbeddingClient falls back
    # to provider.base_url and store.embedding_model so legacy YAML
    # files keep working.
    ollama_url: str | None = None
    ollama_model: str | None = None


@dataclass
class PipelineConfig:
    conversation_turns: int = 3
    max_rounds: int = 5
    core_facts_size: int = 30
    # Legacy default — the 8000 cap is a holdover from ~4K-context models.
    # Modern upstream models (qwen3:14b = 40K native, qwen3.6:35b = 131K)
    # can safely accept 2-4x this. `resolve_budget` below applies
    # upstream-aware scaling when `max_outbound_tokens` is at the
    # conservative default; explicit YAML overrides are honoured as-is.
    max_outbound_tokens: int = 8000
    think_enabled: bool = False      # reserved for user-controllable <#think_on#>/<#think_off#> tags; pipeline does not inject a `think` flag (see pipeline.py)
    context_format: str = "auto"  # "flat" | "structured" | "auto" (auto dispatches structured for temporal queries, flat otherwise)
    # Facts surfaced into the lean payload BEFORE any tool call. Smaller
    # here keeps the outbound prompt tight (~230→~75 tokens of facts);
    # when the model needs more it can still invoke the recall tool to
    # pull deeper results bounded by core_facts_size. Diagnosis 2026-04-18
    # showed 15 retrieved facts injected ~230 tokens/query, most
    # irrelevant to the specific query.
    pre_populate_top_k: int = 5
    # Fallback upstream native-ctx used when the inbound payload does not
    # specify `options.num_ctx`. 8192 is a conservative baseline — on
    # most modern upstream models the actual num_ctx will be much larger.
    upstream_ctx_default: int = 8192

    # Scaling constants — true class-level (ClassVar), not dataclass fields.
    _BUDGET_FLOOR: ClassVar[int] = 4000
    _BUDGET_CEILING: ClassVar[int] = 32768
    _SCALE_THRESHOLD: ClassVar[int] = 16384  # only scale up when upstream_ctx > this

    def resolve_budget(self, upstream_ctx: int) -> int:
        """Return the effective outbound token budget.

        If the dataclass default (8000) is in effect AND the upstream
        model's native ctx is large enough to benefit from scaling,
        return min(upstream_ctx // 2, _BUDGET_CEILING). Otherwise honour
        the explicit setting as-is. The floor (_BUDGET_FLOOR) only
        applies in the default-scaling path to guard against degenerate
        tiny-ctx models; explicit YAML/programmatic overrides are
        honoured exactly.
        """
        explicit = self.max_outbound_tokens
        if explicit != 8000:
            # User-set (YAML override or non-default) — honour exactly.
            return explicit
        if upstream_ctx > self._SCALE_THRESHOLD:
            return min(upstream_ctx // 2, self._BUDGET_CEILING)
        return max(explicit, self._BUDGET_FLOOR)


@dataclass
class ProgressionConfig:
    """Cold-start progressive activation thresholds and turn budgets.

    Sieve picks one of three phases per request based on the number of
    ``status='current'`` facts in the store. Each phase specifies how
    many recent conversation turns the lean-payload composer retains.

    - **OBSERVE** (facts < ``phase_1_threshold``): keep ``observe_turns``
      prior turns. Writer is extracting aggressively but the store is
      still thin, so raw conversation history is the safety net.
    - **ACCUMULATE** (``phase_1_threshold`` <= facts <
      ``phase_2_threshold``): hybrid — fewer turns, retrieved facts
      growing in quality.
    - **ACTIVATE** (facts >= ``phase_2_threshold``): retrieval-driven,
      lean payload, minimal history.

    The actual phase decision is in ``sieve.progression.detect_phase``.
    """
    phase_1_threshold: int = 20    # facts needed to leave OBSERVE
    phase_2_threshold: int = 50    # facts needed to enter ACTIVATE
    observe_turns: int = 8
    accumulate_turns: int = 4
    activate_turns: int = 2


@dataclass
class WriterConfig:
    # 'auto' routes S2 extraction to provider.default_model so the user's
    # main model handles both inference and extraction. Override with an
    # explicit model name if you prefer a separate writer (e.g. a
    # CPU-pinned qwen3.5:2b for users who want to save GPU VRAM).
    model: str = "auto"
    fallback_model: str = "auto"
    num_ctx: int = 4096
    ghost_validator_enabled: bool = True


@dataclass
class RetrievalConfig:
    """Retrieval pipeline tunables."""
    temporal_dedup_enabled: bool = True
    # Cross-encoder re-ranking over vector search candidates. Runs
    # CPU-side in-process, ~20-50ms per query. Ships ON to tighten
    # retrieval precision on mid-run queries.
    reranker_enabled: bool = True
    # Decompose complexity=2 queries into 2-4 sub-queries before vector
    # search. Adds one LLM call per multi-hop query (~100-500ms) plus a
    # handful of extra vector searches. Ships ON to lift multi-hop
    # accuracy.
    query_decomposition_enabled: bool = True


@dataclass
class LearningConfig:
    tune_interval: int = 50       # run tuning loop every N interactions
    relevance_threshold: float = 0.7  # cosine sim above this = "used"
    core_facts_size: int = 30     # top N facts by usage_rate


@dataclass
class SecurityConfig:
    auth_token: str | None = None         # auto-generated if not set
    allowed_origins: list[str] = field(default_factory=lambda: ["127.0.0.1"])


@dataclass
class ToolsConfig:
    enabled: bool = True
    compression: str = "moderate"       # none | moderate | aggressive
    l1_threshold: float = 0.5
    fallback_include_all: bool = True
    max_tools_injected: int = 10


@dataclass
class AblationConfig:
    fingerprinting: bool = True
    classifier: bool = True
    pre_populate: bool = True
    graph_traversal: bool = True
    temporal_versioning: bool = True
    learning_loop: bool = True
    coherence_integrity: bool = True
    stage2_writer: bool = True
    recall_tool: bool = True
    # Response Verification Layer.
    # Pass-2 ablation showed AS-only is the only net-positive combination at
    # MEDIUM/qwen3.5:9b: +0.25 aggregate accuracy, -0.02 hallucination, with
    # +0.54 acc / -0.26 hallu on D_trap specifically. CW (closed-world framing)
    # collapses A/B accuracy via excessive caution. RV (response verification)
    # interacts badly with AS and never fired in pass 2 anyway. Default both
    # to OFF; keep AS on.
    absence_signal: bool = True            # ABL-AS — Layer 1 (shipping)
    closed_world: bool = False             # ABL-CW — Layer 2 (regressed)
    response_verification: bool = False    # ABL-RV — Layer 3 (no measurable benefit)
    # schema v2 — slot-based writes, supersession, SlotRetriever,
    # context format v2, known_unknowns. Default OFF; gated behind this
    # flag. When False, writer/retrieval/format paths are unchanged from
    # the legacy schema.
    schema_v2: bool = False                # ABL-SV2 — schema redesign
    # Two-tier writer + three-tier retrieval + EXTREME summary.
    # tier2_classifier — gemma4:e4b classifies free-text facts into
    # structured tags (writer) and routes queries to predicates (retrieval)
    # when the rule-based Tier 1 keyword classifier returns generic.
    # extreme_summary — when inbound payload > 25K tokens, compress the
    # stripped bloat into a ~500-token narrative and include as
    # [NARRATIVE SUMMARY] in the context block.
    tier2_classifier: bool = False         # ABL-T2C — LLM classifier (off by default)
    extreme_summary: bool = True           # ABL-XS — EXTREME summary (ships on)


@dataclass
class ProfileOwnerConfig:
    """Who the user is. Pinned into the writer S2 prompt and used by the
    ghost-fact validator to reject inverted-identity extractions."""
    name: str = ""                    # canonical full name, e.g. "Jamie Rivera"
    aliases: list[str] = field(default_factory=list)  # first-person + nicknames
    # Optional one-sentence identity statement. When set, bootstrapped
    # into the store on first open (as the User entity + a seed fact)
    # and also appended to the lean system prompt so cold-start queries
    # on an empty store still know who "I" is.
    pin: str = ""


@dataclass
class ValidationConfig:
    """Opt-in metrics collection for the live validation harness.

    When ``enabled=True`` the ``/api/chat`` and ``/v1/chat/completions``
    endpoints emit one Tier 1 + Tier 2 metrics row per intercepted
    request into ``db_path``. Default is OFF so normal operation pays
    nothing. Used only by ``evaluation/live/`` \u2014 do not enable in
    production configs.
    """
    enabled: bool = False
    db_path: str = "~/.sieve/validation_metrics.db"


# ── Model capabilities registry (Mode A/B) ─────────────────────────────────

# Models known to support reliable tool calling → Mode A
# Models that don't → Mode B (classifier does all context selection)
_TOOL_CAPABLE_MODELS = {
    "qwen3.6:35b-a3b",
    "qwen3.5:35b", "qwen3.5:27b", "qwen3.5:9b", "qwen3.5:4b", "qwen3.5:2b",
    "qwen3:30b", "qwen3:14b", "qwen3:30b-a3b",
    "qwen2.5:32b", "qwen2.5:14b",
    "llama3", "deepseek-r1:32b",
    "gemma4:26b", "gemma4:31b", "gemma4:e4b",
    "mistral-small3.2:24b",
    "phi-4:14b",
}

# Models known to NOT support tool calling → Mode B
_NON_TOOL_MODELS = {
    "qwen3.5:0.5b", "qwen2.5:1.5b",
    "nomic-embed-text-v2-moe",
}


def detect_mode(model: str, override: str = "auto") -> str:
    """Detect Mode A (tool-calling) or Mode B (no tools) for a model.

    Args:
        model: Model name (e.g. "qwen3.5:9b")
        override: "A", "B", or "auto" (default)

    Returns: "A" or "B"
    """
    if override.upper() in ("A", "B"):
        return override.upper()
    # Strip version tags for matching (e.g. "qwen3.5:9b-cpu" → "qwen3.5:9b")
    base = model.split("-cpu")[0].split("-gpu")[0]
    if base in _NON_TOOL_MODELS:
        return "B"
    if base in _TOOL_CAPABLE_MODELS:
        return "A"
    # Default: assume tool-capable for unknown models
    return "A"


@dataclass
class RecallConfig:
    listen: ListenConfig = field(default_factory=ListenConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    progression: ProgressionConfig = field(default_factory=ProgressionConfig)
    profile_owner: ProfileOwnerConfig = field(default_factory=ProfileOwnerConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    mode_override: str = "auto"  # A, B, or auto

    @classmethod
    def load(cls, path: str | Path | None = None) -> RecallConfig:
        """Load config from YAML file. Falls back to defaults if no file found."""
        raw = {}

        if path is not None:
            raw = _read_yaml(Path(path))
        else:
            # Check env var first
            env_path = os.environ.get("SIEVE_CONFIG")
            if env_path:
                raw = _read_yaml(Path(env_path))
            else:
                for candidate in DEFAULT_CONFIG_PATHS:
                    resolved = candidate.expanduser()
                    if resolved.exists():
                        raw = _read_yaml(resolved)
                        break

        return _build_config(raw)


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_config(raw: dict) -> RecallConfig:
    listen_raw = raw.get("listen", {})
    provider_raw = raw.get("provider", {})
    store_raw = raw.get("store", {})

    listen = ListenConfig(
        host=listen_raw.get("host", "0.0.0.0"),
        port=int(listen_raw.get("port", 11435)),
    )

    provider = ProviderConfig(
        type=provider_raw.get("type", "auto"),
        base_url=provider_raw.get("base_url", "http://127.0.0.1:11434"),
        api_key=provider_raw.get("api_key"),
        default_model=provider_raw.get("default_model", "qwen3.5:35b"),
        options=provider_raw.get("options", {}),
    )

    store = StoreConfig(
        path=store_raw.get("path", "~/.sieve/memory.db"),
        embedding_model=store_raw.get("embedding_model", "nomic-embed-text-v2-moe"),
        embedding_dimensions=int(store_raw.get("embedding_dimensions", 768)),
    )

    embeddings_raw = raw.get("embeddings") or {}
    embeddings = EmbeddingsConfig(
        provider=str(embeddings_raw.get("provider", "fastembed")),
        ollama_url=embeddings_raw.get("ollama_url"),
        ollama_model=embeddings_raw.get("ollama_model"),
    )
    if embeddings.provider not in ("fastembed", "ollama"):
        logging.getLogger("recall.config").warning(
            "embeddings.provider=%r invalid; defaulting to 'fastembed'",
            embeddings.provider,
        )
        embeddings.provider = "fastembed"

    # When FastEmbed is active, the effective store dimension is always
    # 384 regardless of a legacy embedding_dimensions setting. We update
    # the StoreConfig field here so downstream code (store.init_schema,
    # tool_registry.py) sees the right value without needing a new API.
    if embeddings.provider == "fastembed":
        store.embedding_dimensions = 384
    else:
        valid_dims = (256, 512, 768)
        if store.embedding_dimensions not in valid_dims:
            logging.getLogger("recall.config").warning(
                "store.embedding_dimensions=%d not in %s for ollama provider; "
                "defaulting to 768",
                store.embedding_dimensions, valid_dims,
            )
            store.embedding_dimensions = 768

    pipeline_raw = raw.get("pipeline", raw.get("recall", {}))
    pipeline = PipelineConfig(
        conversation_turns=int(pipeline_raw.get("conversation_turns", 3)),
        max_rounds=int(pipeline_raw.get("max_rounds", 5)),
        core_facts_size=int(pipeline_raw.get("core_facts_size", 30)),
        max_outbound_tokens=int(pipeline_raw.get("max_outbound_tokens", 8000)),
        think_enabled=bool(pipeline_raw.get("think_enabled", False)),
        context_format=str(pipeline_raw.get("context_format", "auto")),
        pre_populate_top_k=int(pipeline_raw.get("pre_populate_top_k", 5)),
    )
    if pipeline.context_format not in ("flat", "structured", "auto"):
        logging.getLogger("recall.config").warning(
            "pipeline.context_format=%r not in (flat, structured, auto), defaulting to auto",
            pipeline.context_format,
        )
        pipeline.context_format = "auto"

    writer_raw = raw.get("writer", {})
    writer = WriterConfig(
        model=writer_raw.get("model", "auto"),
        fallback_model=writer_raw.get("fallback_model", "auto"),
        ghost_validator_enabled=bool(writer_raw.get("ghost_validator_enabled", True)),
    )

    retrieval_raw = raw.get("retrieval") or {}
    retrieval = RetrievalConfig(
        temporal_dedup_enabled=bool(retrieval_raw.get("temporal_dedup", True)),
        reranker_enabled=bool(retrieval_raw.get("reranker_enabled", True)),
        query_decomposition_enabled=bool(
            retrieval_raw.get("query_decomposition_enabled", True)
        ),
    )

    learning_raw = raw.get("learning", {})
    learning = LearningConfig(
        tune_interval=int(learning_raw.get("tune_interval", 50)),
        relevance_threshold=float(learning_raw.get("relevance_threshold", 0.7)),
        core_facts_size=int(learning_raw.get("core_facts_size", 30)),
    )

    security_raw = raw.get("security", {})
    allowed = security_raw.get("allowed_origins", ["127.0.0.1"])
    if isinstance(allowed, str):
        allowed = [allowed]
    security = SecurityConfig(
        auth_token=security_raw.get("auth_token"),
        allowed_origins=allowed,
    )

    tools_raw = raw.get("tools", {})
    tools_compression = tools_raw.get("compression", "moderate")
    if tools_compression not in ("none", "moderate", "aggressive"):
        logging.getLogger("recall.config").warning(
            "tools.compression=%r not in (none, moderate, aggressive), defaulting to moderate",
            tools_compression,
        )
        tools_compression = "moderate"
    tools = ToolsConfig(
        enabled=bool(tools_raw.get("enabled", True)),
        compression=tools_compression,
        l1_threshold=float(tools_raw.get("l1_threshold", 0.5)),
        fallback_include_all=bool(tools_raw.get("fallback_include_all", True)),
        max_tools_injected=int(tools_raw.get("max_tools_injected", 10)),
    )

    # Loader defaults MUST match AblationConfig dataclass defaults so an
    # empty or minimal yaml produces the documented shipping behaviour.
    # See test_empty_yaml_produces_dataclass_defaults for the invariant.
    ablation_defaults = AblationConfig()
    ablation_raw = raw.get("ablation", {})
    ablation = AblationConfig(
        fingerprinting=bool(ablation_raw.get("fingerprinting", ablation_defaults.fingerprinting)),
        classifier=bool(ablation_raw.get("classifier", ablation_defaults.classifier)),
        pre_populate=bool(ablation_raw.get("pre_populate", ablation_defaults.pre_populate)),
        graph_traversal=bool(ablation_raw.get("graph_traversal", ablation_defaults.graph_traversal)),
        temporal_versioning=bool(ablation_raw.get("temporal_versioning", ablation_defaults.temporal_versioning)),
        learning_loop=bool(ablation_raw.get("learning_loop", ablation_defaults.learning_loop)),
        coherence_integrity=bool(ablation_raw.get("coherence_integrity", ablation_defaults.coherence_integrity)),
        stage2_writer=bool(ablation_raw.get("stage2_writer", ablation_defaults.stage2_writer)),
        recall_tool=bool(ablation_raw.get("recall_tool", ablation_defaults.recall_tool)),
        absence_signal=bool(ablation_raw.get("absence_signal", ablation_defaults.absence_signal)),
        closed_world=bool(ablation_raw.get("closed_world", ablation_defaults.closed_world)),
        response_verification=bool(ablation_raw.get("response_verification", ablation_defaults.response_verification)),
        schema_v2=bool(ablation_raw.get("schema_v2", ablation_defaults.schema_v2)),
        tier2_classifier=bool(ablation_raw.get("tier2_classifier", ablation_defaults.tier2_classifier)),
        extreme_summary=bool(ablation_raw.get("extreme_summary", ablation_defaults.extreme_summary)),
    )

    progression_raw = raw.get("progression") or {}
    progression = ProgressionConfig(
        phase_1_threshold=int(progression_raw.get("phase_1_threshold", 20)),
        phase_2_threshold=int(progression_raw.get("phase_2_threshold", 50)),
        observe_turns=int(progression_raw.get("observe_turns", 8)),
        accumulate_turns=int(progression_raw.get("accumulate_turns", 4)),
        activate_turns=int(progression_raw.get("activate_turns", 2)),
    )
    # Validate thresholds: they must be non-negative, phase_2 >= phase_1,
    # and turn counts must be positive. If any rule is violated, warn and
    # fall back to defaults — same pattern as pipeline.context_format.
    _log_cfg = logging.getLogger("recall.config")
    invalid = (
        progression.phase_1_threshold < 0
        or progression.phase_2_threshold < progression.phase_1_threshold
        or progression.observe_turns <= 0
        or progression.accumulate_turns <= 0
        or progression.activate_turns <= 0
    )
    if invalid:
        _log_cfg.warning(
            "progression config invalid (%s); defaulting to %s",
            progression, ProgressionConfig(),
        )
        progression = ProgressionConfig()

    owner_raw = raw.get("profile_owner") or {}
    aliases_raw = owner_raw.get("aliases", [])
    if isinstance(aliases_raw, str):
        aliases_raw = [aliases_raw]
    profile_owner = ProfileOwnerConfig(
        name=str(owner_raw.get("name", "")),
        aliases=list(aliases_raw),
        pin=str(owner_raw.get("pin", "")),
    )

    mode_override = str(raw.get("mode_override", "auto"))

    validation_raw = raw.get("validation") or {}
    validation = ValidationConfig(
        enabled=bool(validation_raw.get("enabled", False)),
        db_path=str(validation_raw.get("db_path", "~/.sieve/validation_metrics.db")),
    )

    return RecallConfig(
        listen=listen, provider=provider, store=store,
        embeddings=embeddings,
        pipeline=pipeline, writer=writer, retrieval=retrieval, learning=learning,
        security=security, tools=tools,
        ablation=ablation, progression=progression,
        profile_owner=profile_owner,
        validation=validation,
        mode_override=mode_override,
    )

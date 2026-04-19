"""Opt-in validation metrics collector for Recall.

When ``config.validation.enabled`` is ``True``, the chat endpoints in
``src.main`` construct a :class:`RequestMetrics` object at the start
of each intercepted request and hand it to the downstream pipeline
stages. Each stage calls a setter (``record_classifier``,
``record_retrieval``, ...) to deposit whatever numbers it produced.
After the upstream response completes, ``finalise_and_persist`` writes
a single row to ``~/.sieve/validation_metrics.db`` with path='recall'.

The collector has ZERO behavioural effect on the proxy pipeline \u2014 it
only observes. If ``enabled=False`` (the default) every method is a
cheap no-op, so normal operation pays nothing.

Spec: https://www.notion.so/345d67b4903a8144be25e8834b4150a6
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("recall.validation")


DEFAULT_DB_PATH = Path("~/.sieve/validation_metrics.db").expanduser()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS request_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER,
    run_id TEXT,
    seed INTEGER,
    path TEXT,
    timestamp_sent TEXT,
    timestamp_first_token TEXT,
    timestamp_complete TEXT,
    simulated_day INTEGER,
    category TEXT,
    user_message TEXT,
    response_content TEXT,
    response_token_count INTEGER,
    inbound_tokens INTEGER,
    outbound_tokens INTEGER,
    ttft_ms REAL,
    total_latency_ms REAL,
    inference_latency_ms REAL,
    proxy_overhead_ms REAL,
    model_name TEXT,
    http_status INTEGER,
    error TEXT,
    stream_chunks INTEGER,
    token_reduction_pct REAL,
    token_reduction_ratio REAL,
    facts_in_store INTEGER,
    facts_current INTEGER,
    facts_superseded INTEGER,
    entities_in_store INTEGER,
    relationships_in_store INTEGER,
    known_unknowns_count INTEGER,
    classifier_level TEXT,
    classifier_decision TEXT,
    retrieval_tier TEXT,
    facts_retrieved INTEGER,
    facts_used_in_response INTEGER,
    retrieval_precision REAL,
    recall_tool_calls INTEGER,
    recall_tool_queries TEXT,
    absence_signals_injected INTEGER,
    absence_signal_categories TEXT,
    temporal_markers_current INTEGER,
    temporal_markers_past INTEGER,
    context_sections_included TEXT,
    fingerprint_cache_hits INTEGER,
    fingerprint_cache_misses INTEGER,
    writer_stage1_facts INTEGER,
    writer_stage2_facts INTEGER,
    writer_stage2_invoked INTEGER,
    writer_conflicts_detected INTEGER,
    writer_supersessions INTEGER,
    session_coherence_score REAL,
    system_prompt_tokens INTEGER,
    context_block_tokens INTEGER,
    lean_payload_breakdown TEXT
);

CREATE INDEX IF NOT EXISTS idx_request_run_query ON request_metrics(run_id, query_id);
CREATE INDEX IF NOT EXISTS idx_request_path ON request_metrics(path);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class RequestMetrics:
    """Mutable accumulator populated across the pipeline stages.

    All fields default to ``None`` (or an empty structure) so uninstrumented
    stages still produce a sensible row. Fill only what you actually
    observed \u2014 missing fields are written as SQL NULL.
    """

    # Request identity (from X-Validation-* headers)
    query_id: int | None = None
    run_id: str | None = None
    seed: int | None = None
    simulated_day: int | None = None
    category: str | None = None
    # "direct_recall" (default) or "agent_framework_recall" when the
    # runner forwards through the agent framework. Honours
    # X-Validation-Path on the inbound request; baseline rows use the
    # parallel label in metrics_proxy.py.
    path_label: str = "direct_recall"

    # Timestamps
    timestamp_sent: str = field(default_factory=_now_iso)
    timestamp_first_token: str | None = None
    timestamp_complete: str | None = None

    # Request / response payloads
    user_message: str = ""
    response_content: str = ""
    response_token_count: int | None = None
    inbound_tokens: int | None = None
    outbound_tokens: int | None = None
    model_name: str = ""
    http_status: int | None = None
    error: str | None = None
    stream_chunks: int = 0

    # Timings
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    inference_latency_ms: float | None = None
    proxy_overhead_ms: float | None = None

    # Token economy
    token_reduction_pct: float | None = None
    token_reduction_ratio: float | None = None

    # Store state (post-request snapshot)
    facts_in_store: int | None = None
    facts_current: int | None = None
    facts_superseded: int | None = None
    entities_in_store: int | None = None
    relationships_in_store: int | None = None
    known_unknowns_count: int | None = None

    # Classifier + retrieval
    classifier_level: str | None = None
    classifier_decision: str | None = None
    retrieval_tier: str | None = None
    facts_retrieved: int | None = None
    facts_used_in_response: int | None = None
    retrieval_precision: float | None = None

    # Recall tool
    recall_tool_calls: int = 0
    recall_tool_queries: list[str] = field(default_factory=list)

    # Absence signals / temporal markers
    absence_signals_injected: int = 0
    absence_signal_categories: list[str] = field(default_factory=list)
    temporal_markers_current: int = 0
    temporal_markers_past: int = 0

    # Context composition
    context_sections_included: list[str] = field(default_factory=list)
    system_prompt_tokens: int | None = None
    context_block_tokens: int | None = None
    lean_payload_breakdown: dict[str, int] = field(default_factory=dict)

    # Fingerprinting
    fingerprint_cache_hits: int = 0
    fingerprint_cache_misses: int = 0

    # Writer
    writer_stage1_facts: int = 0
    writer_stage2_facts: int = 0
    writer_stage2_invoked: bool = False
    writer_conflicts_detected: int = 0
    writer_supersessions: int = 0

    # Coherence
    session_coherence_score: float | None = None

    # Helpers --------------------------------------------------------

    def note_recall_tool_call(self, query: str) -> None:
        self.recall_tool_calls += 1
        self.recall_tool_queries.append(query[:200])

    def set_reduction(self) -> None:
        if self.inbound_tokens and self.outbound_tokens is not None:
            reduction = self.inbound_tokens - self.outbound_tokens
            self.token_reduction_pct = (
                (reduction / self.inbound_tokens) * 100.0 if self.inbound_tokens else 0.0
            )
            self.token_reduction_ratio = (
                self.inbound_tokens / self.outbound_tokens
                if self.outbound_tokens else 0.0
            )

    def to_row(self) -> dict[str, Any]:
        """Serialise for insertion into ``request_metrics``."""
        row = asdict(self)
        # Coerce lists/dicts to JSON for SQL storage
        row["recall_tool_queries"] = json.dumps(row["recall_tool_queries"])
        row["absence_signal_categories"] = json.dumps(row["absence_signal_categories"])
        row["context_sections_included"] = json.dumps(row["context_sections_included"])
        row["lean_payload_breakdown"] = json.dumps(row["lean_payload_breakdown"])
        row["writer_stage2_invoked"] = 1 if row["writer_stage2_invoked"] else 0
        row["path"] = row.pop("path_label") or "direct_recall"
        return row


class ValidationCollector:
    """Instance-level API consumed by :mod:`src.main`.

    One collector is constructed per FastAPI app; it owns the DB path
    and the snapshot helpers. Per-request state is held on
    :class:`RequestMetrics` instances, not here.
    """

    def __init__(self, db_path: Path | str | None = None, enabled: bool = False):
        self.enabled = bool(enabled)
        self.db_path = Path(db_path or DEFAULT_DB_PATH).expanduser()
        if self.enabled:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def start(
        self,
        *,
        query_id: int | None,
        run_id: str | None,
        seed: int | None,
        simulated_day: int | None,
        category: str | None,
        user_message: str,
        model_name: str,
        inbound_tokens: int | None,
        path_label: str | None = None,
    ) -> RequestMetrics | None:
        """Construct a RequestMetrics for this request, or None if disabled."""
        if not self.enabled:
            return None
        return RequestMetrics(
            query_id=query_id,
            run_id=run_id,
            seed=seed,
            simulated_day=simulated_day,
            category=category,
            user_message=user_message,
            model_name=model_name,
            inbound_tokens=inbound_tokens,
            path_label=path_label or "direct_recall",
        )

    def snapshot_store(self, metrics: RequestMetrics | None, memory_store: Any) -> None:
        """Record post-request store sizes. Cheap (one indexed COUNT)."""
        if metrics is None or memory_store is None or memory_store._conn is None:
            return
        try:
            stats = memory_store.stats()
            # MemoryStore.stats() returns keys suffixed with "_count"
            # (facts_count, entities_count, …) — the collector previously
            # read un-suffixed names and silently recorded zeros.
            metrics.facts_in_store = int(stats.get("facts_count", 0) or 0)
            metrics.entities_in_store = int(stats.get("entities_count", 0) or 0)
            metrics.relationships_in_store = int(stats.get("relationships_count", 0) or 0)
            metrics.known_unknowns_count = int(stats.get("known_unknowns_count", 0) or 0)
            # Status breakdown
            cur = memory_store.conn.execute(
                "SELECT status, COUNT(*) FROM facts GROUP BY status"
            ).fetchall()
            by_status = {row[0]: row[1] for row in cur}
            metrics.facts_current = int(by_status.get("current", 0) or 0)
            metrics.facts_superseded = int(by_status.get("superseded", 0) or 0)
        except Exception as exc:
            logger.debug("snapshot_store failed: %s", exc)

    def record_classifier(
        self,
        metrics: RequestMetrics | None,
        *,
        level: int | None,
        decision: bool | None,
    ) -> None:
        if metrics is None:
            return
        if level is not None:
            metrics.classifier_level = f"L{level}"
        if decision is not None:
            metrics.classifier_decision = "retrieve" if decision else "skip"

    def record_retrieval(
        self,
        metrics: RequestMetrics | None,
        *,
        tier: str | None = None,
        facts_retrieved: int | None = None,
        context_block_text: str | None = None,
    ) -> None:
        if metrics is None:
            return
        if tier is not None:
            metrics.retrieval_tier = tier
        if facts_retrieved is not None:
            metrics.facts_retrieved = facts_retrieved
        if context_block_text is not None:
            metrics.context_block_tokens = _approx_tokens(context_block_text)
            # Count CURRENT / PAST markers the retriever added
            metrics.temporal_markers_current = context_block_text.count("[CURRENT]")
            metrics.temporal_markers_past = context_block_text.count("[PAST]")

    def record_absence_signals(
        self,
        metrics: RequestMetrics | None,
        *,
        count: int,
        categories: list[str] | None = None,
    ) -> None:
        if metrics is None:
            return
        metrics.absence_signals_injected = count
        if categories:
            metrics.absence_signal_categories = list(categories)

    def record_compose(
        self,
        metrics: RequestMetrics | None,
        *,
        lean_payload: dict[str, Any] | None,
        outbound_tokens: int | None = None,
        sections: list[str] | None = None,
        system_prompt_text: str | None = None,
    ) -> None:
        if metrics is None:
            return
        if outbound_tokens is not None:
            metrics.outbound_tokens = outbound_tokens
        if sections:
            metrics.context_sections_included = list(sections)
        if system_prompt_text:
            metrics.system_prompt_tokens = _approx_tokens(system_prompt_text)
        if lean_payload is not None:
            breakdown: dict[str, int] = {}
            for msg in lean_payload.get("messages", []):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content)
                breakdown[role] = breakdown.get(role, 0) + _approx_tokens(content)
            tools = lean_payload.get("tools")
            if tools:
                breakdown["tools"] = _approx_tokens(json.dumps(tools))
            metrics.lean_payload_breakdown = breakdown
            if metrics.outbound_tokens is None:
                metrics.outbound_tokens = sum(breakdown.values())
        metrics.set_reduction()

    def record_fingerprints(
        self,
        metrics: RequestMetrics | None,
        *,
        hits: int,
        misses: int,
    ) -> None:
        if metrics is None:
            return
        metrics.fingerprint_cache_hits = hits
        metrics.fingerprint_cache_misses = misses

    def record_timing(
        self,
        metrics: RequestMetrics | None,
        *,
        ttft_ms: float | None = None,
        total_latency_ms: float | None = None,
        inference_latency_ms: float | None = None,
        first_token_ts: str | None = None,
        complete_ts: str | None = None,
        stream_chunks: int | None = None,
        http_status: int | None = None,
        error: str | None = None,
    ) -> None:
        if metrics is None:
            return
        if ttft_ms is not None:
            metrics.ttft_ms = ttft_ms
        if total_latency_ms is not None:
            metrics.total_latency_ms = total_latency_ms
        if inference_latency_ms is not None:
            metrics.inference_latency_ms = inference_latency_ms
        if ttft_ms is None and total_latency_ms is not None:
            metrics.ttft_ms = total_latency_ms
        if total_latency_ms is not None and inference_latency_ms is not None:
            metrics.proxy_overhead_ms = max(0.0, total_latency_ms - inference_latency_ms)
        if first_token_ts is not None:
            metrics.timestamp_first_token = first_token_ts
        if complete_ts is not None:
            metrics.timestamp_complete = complete_ts
        if stream_chunks is not None:
            metrics.stream_chunks = stream_chunks
        if http_status is not None:
            metrics.http_status = http_status
        if error is not None:
            metrics.error = error

    def record_response(
        self,
        metrics: RequestMetrics | None,
        *,
        content: str,
        token_count: int | None = None,
    ) -> None:
        if metrics is None:
            return
        metrics.response_content = content
        metrics.response_token_count = token_count if token_count is not None else _approx_tokens(content)

    def record_recall_tool_call(
        self,
        metrics: RequestMetrics | None,
        *,
        query: str,
    ) -> None:
        if metrics is None:
            return
        metrics.note_recall_tool_call(query)

    def record_writer_result(
        self,
        metrics: RequestMetrics | None,
        result: object | None,
    ) -> None:
        """Copy per-stage counts from a writer.WriteResult onto metrics.

        Called by the request handler after awaiting the writer task.
        Safe when either argument is None (validation disabled, writer
        timed out, or fire-and-forget path where no result is available).
        """
        if metrics is None or result is None:
            return
        metrics.writer_stage1_facts = getattr(result, "stage1_facts", 0)
        metrics.writer_stage2_facts = getattr(result, "stage2_facts", 0)
        metrics.writer_stage2_invoked = bool(getattr(result, "stage2_invoked", False))
        metrics.writer_conflicts_detected = getattr(result, "conflicts_detected", 0)
        metrics.writer_supersessions = getattr(result, "supersessions", 0)

    def finalise_and_persist(self, metrics: RequestMetrics | None) -> None:
        """Write the accumulated row to disk. Safe to call with None."""
        if metrics is None or not self.enabled:
            return
        try:
            metrics.set_reduction()
            row = metrics.to_row()
            self._insert(row)
        except Exception as exc:
            logger.warning("validation: finalise failed: %s", exc)

    def _insert(self, row: dict[str, Any]) -> None:
        columns = [
            "query_id", "run_id", "seed", "path",
            "timestamp_sent", "timestamp_first_token", "timestamp_complete",
            "simulated_day", "category",
            "user_message", "response_content", "response_token_count",
            "inbound_tokens", "outbound_tokens",
            "ttft_ms", "total_latency_ms", "inference_latency_ms", "proxy_overhead_ms",
            "model_name", "http_status", "error", "stream_chunks",
            "token_reduction_pct", "token_reduction_ratio",
            "facts_in_store", "facts_current", "facts_superseded",
            "entities_in_store", "relationships_in_store", "known_unknowns_count",
            "classifier_level", "classifier_decision", "retrieval_tier",
            "facts_retrieved", "facts_used_in_response", "retrieval_precision",
            "recall_tool_calls", "recall_tool_queries",
            "absence_signals_injected", "absence_signal_categories",
            "temporal_markers_current", "temporal_markers_past",
            "context_sections_included",
            "fingerprint_cache_hits", "fingerprint_cache_misses",
            "writer_stage1_facts", "writer_stage2_facts", "writer_stage2_invoked",
            "writer_conflicts_detected", "writer_supersessions",
            "session_coherence_score",
            "system_prompt_tokens", "context_block_tokens", "lean_payload_breakdown",
        ]
        placeholders = ",".join(["?"] * len(columns))
        sql = f"INSERT INTO request_metrics ({','.join(columns)}) VALUES ({placeholders})"
        values = [row.get(c) for c in columns]
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(sql, values)
            conn.commit()
        finally:
            conn.close()


def extract_validation_headers(headers: dict[str, str]) -> dict[str, Any]:
    """Pull the X-Validation-* envelope off the inbound request headers."""
    def _int(key: str) -> int | None:
        v = headers.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "query_id": _int("x-validation-query-id"),
        "run_id": headers.get("x-validation-run-id"),
        "seed": _int("x-validation-seed"),
        "simulated_day": _int("x-validation-day"),
        "path_label": headers.get("x-validation-path"),
        "category": headers.get("x-validation-category"),
    }

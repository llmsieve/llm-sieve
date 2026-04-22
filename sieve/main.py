"""FastAPI application entry point for the Sieve proxy."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sieve.classifier import QueryClassifier, ToolClassifier
from sieve.config import RecallConfig, detect_mode
from sieve.embeddings import EmbeddingClient
from sieve.embeddings_provider import RerankerService
from sieve.fingerprint import FingerprintCache, decompose
from sieve.learning import LearningLoop
from sieve.history_preamble_adapter import adapt_history_preamble_payload
from sieve.pipeline import compose_lean_payload, compose_with_tool_selection
from sieve.progression import detect_phase
from sieve.tool_registry import ToolRegistry
from sieve.proxy import ProxyClient, forward_payload, forward_request
from sieve.recall_tool import RecallHandler
from sieve.retrieval import ContextRetriever
from sieve.security import check_auth, check_https_warning, get_or_create_auth_token
from sieve.store import MemoryStore
from sieve.validation_collector import ValidationCollector, extract_validation_headers
from sieve.writer import MemoryWriter, resolve_writer_model

logger = logging.getLogger("recall")


async def _await_writer_and_record(
    metrics, task, collector: ValidationCollector, timeout_s: float = 30.0,
) -> None:
    """Rendezvous point: await a background writer task and copy stats.

    Production requests fire the writer as an async task (fire-and-forget)
    so fact extraction never blocks the client. For validation runs we
    still want the per-request metrics row to carry the writer's
    stage-level counts. This helper is called from the stream-finalise
    path (after the response body has been delivered to the client) so
    awaiting the writer here adds latency only to metrics persistence,
    not to user-visible response time.

    Contract:
    - ``metrics`` or ``task`` is None  → no-op (validation disabled or
      nothing to await). If ``task`` is given we still await it so the
      coroutine is not left dangling.
    - ``task`` times out               → defaults preserved; the task is
      left running but not awaited further. We prefer losing per-stage
      stats over wedging response finalisation behind a slow writer.
    - ``task`` raised                  → defaults preserved; exception
      logged and swallowed. Writer bugs must not fail the request.
    """
    if task is None:
        return
    try:
        result = await asyncio.wait_for(task, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.debug("writer task exceeded %.2fs rendezvous timeout", timeout_s)
        return
    except Exception as exc:
        logger.warning("writer task failed before metrics record: %s", exc)
        return
    collector.record_writer_result(metrics, result)


async def _maybe_bootstrap_owner_pin(
    memory_store: MemoryStore,
    profile_owner,
    embed_fn,
) -> None:
    """On first open of an empty store, seed the profile owner as the
    User entity plus a fact carrying the configured identity pin.

    The store needs this for cold-start accuracy: on Day 1 the writer
    hasn't extracted anything yet, so retrieval returns empty and the
    lean system prompt carries no identity. Seeding the owner pin
    guarantees every query from turn 1 onwards has both (a) the User
    entity available for the absence-signal graph lookup and (b) a
    retrievable fact answering "who is the user?".

    Safe to call on every startup: the entity-count check makes it a
    no-op once the store has been written to. Failures are logged and
    swallowed; an unavailable embedder must not block proxy startup.
    """
    if memory_store._conn is None:
        return
    pin = (profile_owner.pin or "").strip()
    owner_name = (profile_owner.name or "").strip()
    if not pin or not owner_name:
        return
    try:
        row = memory_store._conn.execute(
            "SELECT COUNT(*) FROM entities"
        ).fetchone()
        if row and row[0] > 0:
            return  # store already populated — leave it alone
    except Exception as exc:
        logger.warning("bootstrap: entity-count check failed: %s", exc)
        return
    fact_text = f"{owner_name}: {pin}"
    try:
        embedding = await embed_fn(fact_text)
    except Exception as exc:
        logger.warning("bootstrap: embedder unavailable, inserting pin "
                       "without vector: %s", exc)
        embedding = None
    try:
        user_id = memory_store.insert_entity("User", type="person")
        owner_id = memory_store.insert_entity(
            owner_name, type="person", description=pin,
        )
        memory_store.insert_fact(
            fact_text,
            embedding=embedding,
            entity_ids=[user_id, owner_id],
            source="profile_owner_pin",
            confidence=1.0,
        )
        logger.info(
            "Bootstrapped owner pin into empty store (owner=%s)", owner_name,
        )
    except Exception as exc:
        logger.warning("bootstrap: owner-pin insert failed: %s", exc)


def create_app(config: RecallConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = RecallConfig.load()

    proxy_client = ProxyClient(config.provider.base_url)
    embedding_client = EmbeddingClient(config)
    # Optional cross-encoder reranker. Loaded during lifespan startup
    # alongside the FastEmbed embedder so the first real query doesn't
    # pay the ONNX download cost. If loading fails (no network, model
    # unavailable) the service stays unavailable and the retriever
    # degrades to pure vector search — no fatal condition.
    reranker_service = RerankerService()
    # The active embedding backend is the single source of truth for
    # vector dimension. Sync the store config to its value before the
    # store is constructed, otherwise a stale `store.embedding_dimensions`
    # in the YAML would cause vec_facts to be created at the wrong size
    # and every insert/search to error out. check_embedding_dimensions()
    # then compares this effective dimension against what's already on
    # disk and raises EmbeddingDimensionMismatchError if they diverge.
    config.store.embedding_dimensions = embedding_client.dimension
    memory_store = MemoryStore(config.store)
    memory_writer: MemoryWriter | None = None
    classifier: QueryClassifier | None = None
    retriever: ContextRetriever | None = None
    slot_retriever: Any = None  # SlotRetriever, imported lazily
    recall_handler: RecallHandler | None = None
    learning_loop: LearningLoop | None = None
    tool_registry: ToolRegistry | None = None
    tool_classifier: ToolClassifier | None = None
    background_tasks: set[asyncio.Task] = set()

    def _track_task(task: asyncio.Task, label: str) -> None:
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        def _log_exc(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error("Background task %s failed: %r", label, exc, exc_info=exc)

        task.add_done_callback(_log_exc)

    # Auth token
    auth_token = config.security.auth_token
    if not auth_token:
        auth_token = get_or_create_auth_token()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal memory_writer, classifier, retriever, slot_retriever, recall_handler, learning_loop, tool_registry, tool_classifier
        await proxy_client.start()
        await embedding_client.start()
        # Warm FastEmbed so the first real query doesn't pay the
        # model-download + ONNX-session-init cost. FastEmbed downloads
        # ~50MB on first use; on a fresh install we want that to happen
        # here (startup) rather than in the middle of a Day-1 query.
        # Skipped for the Ollama provider — Ollama manages model loading
        # server-side and a warmup call would just round-trip uselessly
        # (and would require a mock in every proxy test).
        if embedding_client.provider_name == "fastembed":
            try:
                await embedding_client.embed("recall embedding warmup")
                logger.info(
                    "Embedding backend ready (provider=%s, dim=%d)",
                    embedding_client.provider_name, embedding_client.dimension,
                )
            except Exception as exc:
                logger.warning("Embedding warmup failed: %s", exc)
        else:
            logger.info(
                "Embedding backend configured (provider=%s, dim=%d)",
                embedding_client.provider_name, embedding_client.dimension,
            )
        # Load the cross-encoder reranker. reranker_service handles
        # failure internally — an exception just leaves .available=False
        # and the retriever runs without rerank.
        if config.retrieval.reranker_enabled:
            loaded = await asyncio.to_thread(reranker_service.load)
            if loaded:
                try:
                    reranker_service.rerank(
                        "warmup query",
                        ["warmup passage"],
                    )
                except Exception as exc:
                    logger.warning("Reranker warmup failed: %s", exc)
        # HTTPS warning
        check_https_warning(config.provider.base_url)
        # Open store if DB exists and is initialized
        if memory_store.db_path.exists():
            memory_store.open()
            if not memory_store.is_initialized():
                memory_store.init_schema()
            memory_store.check_embedding_dimensions()
            logger.info("Memory store loaded from %s", memory_store.db_path)
            # Bootstrap profile-owner identity on first open. If the
            # entities table is empty and a pin is configured, insert the
            # owner entity + a seed fact so the very first query already
            # has identity grounding instead of starting cold.
            await _maybe_bootstrap_owner_pin(
                memory_store, config.profile_owner, embedding_client.embed,
            )
        # writer.model='auto' routes S2 to the user's main model. The
        # fallback model gets the same treatment so it stays consistent
        # with the primary when the user is on cloud/local parity.
        effective_writer_model = resolve_writer_model(config)
        effective_fallback_model = (
            config.provider.default_model
            if config.writer.fallback_model == "auto"
            else config.writer.fallback_model
        )
        memory_writer = MemoryWriter(
            memory_store,
            embed_fn=embedding_client.embed,
            provider_base_url=config.provider.base_url,
            writer_model=effective_writer_model,
            fallback_model=effective_fallback_model,
            num_ctx=config.writer.num_ctx,
            stage2_enabled=config.ablation.stage2_writer,
            coherence_enabled=config.ablation.coherence_integrity,
            owner_name=config.profile_owner.name,
            profile_owner_aliases=config.profile_owner.aliases,
            ghost_validator_enabled=config.writer.ghost_validator_enabled,
            tier2_classifier_enabled=config.ablation.tier2_classifier,
            tier2_classifier_model=config.ablation.tier2_classifier_model,
        )
        classifier = QueryClassifier(memory_store, embed_fn=embedding_client.embed)
        retriever = ContextRetriever(
            memory_store,
            embed_fn=embedding_client.embed,
            top_k=config.pipeline.core_facts_size,
            graph_traversal=config.ablation.graph_traversal,
            temporal_versioning=config.ablation.temporal_versioning,
            context_format=config.pipeline.context_format,
            temporal_dedup_enabled=config.retrieval.temporal_dedup_enabled,
            reranker=reranker_service if reranker_service.available else None,
        )
        # SlotRetriever is constructed unconditionally but only
        # consulted when ablation.schema_v2 is on. Construction is cheap
        # and the extra object is harmless.
        from sieve.slot_retriever import SlotRetriever
        slot_retriever = SlotRetriever(
            memory_store,
            profile_owner_name=config.profile_owner.name,
        )
        if memory_store._conn is not None:
            tool_registry = ToolRegistry(
                memory_store,
                embed_fn=embedding_client.embed,
                compression=config.tools.compression,
            )
            # Recompute lean schemas if compression mode changed since last run
            stored = memory_store.get_fingerprint("config:tool_compression")
            stored_mode = stored["hash"] if stored else None
            if stored_mode != config.tools.compression:
                logger.info(
                    "Tool compression changed: %r -> %r, recomputing lean schemas",
                    stored_mode, config.tools.compression,
                )
                tool_registry.recompute_all_lean_schemas()
                memory_store.upsert_fingerprint(
                    "config:tool_compression",
                    hash_value=config.tools.compression,
                )
            tool_classifier = ToolClassifier(
                tool_registry,
                embed_fn=embedding_client.embed,
                l1_threshold=config.tools.l1_threshold,
                max_tools=config.tools.max_tools_injected,
                fallback_include_all=config.tools.fallback_include_all,
            )
        recall_handler = RecallHandler(
            proxy_client, retriever, config,
            tool_registry=tool_registry,
            slot_retriever=slot_retriever,
        )
        learning_loop = LearningLoop(
            memory_store, embed_fn=embedding_client.embed, config=config.learning,
        )
        logger.info(
            "Sieve proxy listening on %s:%d → %s",
            config.listen.host, config.listen.port, config.provider.base_url,
        )
        yield
        await embedding_client.stop()
        memory_store.close()
        await proxy_client.stop()
        logger.info("Sieve proxy stopped")

    app = FastAPI(title="Sieve", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.proxy_client = proxy_client
    app.state.memory_store = memory_store
    fingerprint_cache = FingerprintCache(memory_store)

    # Validation metrics collector (opt-in; zero overhead when disabled).
    validation_collector = ValidationCollector(
        db_path=config.validation.db_path,
        enabled=config.validation.enabled,
    )
    app.state.validation_collector = validation_collector
    # "Sticky current" envelope set by the runner ahead of each logical
    # Phase B query. The agent framework fans out into multiple Ollama
    # sub-requests per agent turn; all of them share this envelope.
    app.state.validation_current_envelope: dict | None = None
    if config.validation.enabled:
        logger.info(
            "Validation mode ON \u2014 metrics -> %s", validation_collector.db_path,
        )

    # --- Auth middleware for /sieve/ endpoints ---

    from starlette.middleware.base import BaseHTTPMiddleware

    class RecallAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            denied = check_auth(request, auth_token)
            if denied is not None:
                return denied
            return await call_next(request)

    app.add_middleware(RecallAuthMiddleware)

    # --- Health / status ---

    @app.get("/sieve/health")
    async def health():
        store_ready = memory_store._conn is not None and memory_store.is_initialized()
        return {"status": "ok", "version": "0.1.0", "store": store_ready}

    @app.post("/sieve/validation/next")
    async def validation_register_next(request: Request):
        """Register the envelope for the next logical Phase B query.

        Sticky: all agent-framework sub-requests belonging to the same
        agent turn (tool calls, thinking passes) will share this
        envelope until a new one is registered. Required because some
        agent runtimes strip ``X-Validation-*`` headers before the
        request reaches this proxy.
        """
        if not config.validation.enabled:
            return JSONResponse(status_code=503, content={"error": "validation disabled"})
        try:
            data = await request.json()
        except Exception:
            data = {}
        app.state.validation_current_envelope = data or {}
        return {"status": "set", "query_id": (data or {}).get("query_id")}

    @app.post("/sieve/validation/reset")
    async def validation_reset():
        had = app.state.validation_current_envelope is not None
        app.state.validation_current_envelope = None
        return {"status": "reset", "had_envelope": had}

    @app.get("/sieve/stats")
    async def stats():
        if memory_store._conn is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Store not initialized. Run 'sieve store init' first."},
            )
        return memory_store.stats()

    @app.get("/sieve/audit")
    async def audit_log(limit: int = 100, operation: str | None = None):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        query = "SELECT id, operation, target_type, target_id, session_id, timestamp FROM audit_log"
        params: list = []
        if operation:
            query += " WHERE operation = ?"
            params.append(operation)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = memory_store.conn.execute(query, params).fetchall()
        cols = ["id", "operation", "target_type", "target_id", "session_id", "timestamp"]
        return [dict(zip(cols, r)) for r in rows]

    # --- Management API: facts, entities, graph ---

    @app.get("/sieve/facts")
    async def list_facts(limit: int = 100, status: str = "current"):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        facts = memory_store.get_facts(status=status, limit=limit)
        return [
            {k: v for k, v in f.items() if k != "embedding"}
            for f in facts
        ]

    @app.get("/sieve/facts/{fact_id}")
    async def get_fact(fact_id: str):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        fact = memory_store.get_fact(fact_id)
        if fact is None:
            return JSONResponse(status_code=404, content={"error": "Fact not found"})
        return {k: v for k, v in fact.items() if k != "embedding"}

    @app.put("/sieve/facts/{fact_id}/verify")
    async def verify_fact(fact_id: str):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        memory_store.update_fact_status(fact_id, status="current", detail="user_verified")
        return {"status": "verified", "fact_id": fact_id}

    @app.put("/sieve/facts/{fact_id}/reject")
    async def reject_fact(fact_id: str):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        memory_store.update_fact_status(fact_id, status="rejected", detail="user_rejected")
        return {"status": "rejected", "fact_id": fact_id}

    @app.get("/sieve/entities")
    async def list_entities():
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        rows = memory_store.conn.execute(
            "SELECT id, name, type, description, created_at FROM entities ORDER BY created_at DESC"
        ).fetchall()
        cols = ["id", "name", "type", "description", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    @app.get("/sieve/graph/{entity_id}")
    async def graph_traverse(entity_id: str):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        related = memory_store.get_related_entities(entity_id)
        return {"entity_id": entity_id, "related": related}

    @app.get("/sieve/episodes")
    async def list_episodes(limit: int = 100):
        if memory_store._conn is None:
            return JSONResponse(status_code=503, content={"error": "Store not initialized"})
        rows = memory_store.conn.execute(
            "SELECT id, summary, session_id, start_time, end_time FROM episodes ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        cols = ["id", "summary", "session_id", "start_time", "end_time"]
        return [dict(zip(cols, r)) for r in rows]

    import re

    _THINK_TAG_RE = re.compile(r"<#think_(on|off|status)#>", re.IGNORECASE)

    def _process_think_tags(payload: dict) -> dict | None:
        """Intercept think mode tags in user messages.

        Checks the last user message for <#think_on#>, <#think_off#>, <#think_status#>.
        Updates config.pipeline.think_enabled accordingly.
        Strips the tag from the message before forwarding.

        Returns a direct response dict for <#think_status#>, or None to continue pipeline.
        """
        messages = payload.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                break
            match = _THINK_TAG_RE.search(content)
            if not match:
                break
            tag = match.group(1).lower()
            if tag == "on":
                config.pipeline.think_enabled = True
                logger.info("Think mode: ON (user toggled)")
            elif tag == "off":
                config.pipeline.think_enabled = False
                logger.info("Think mode: OFF (user toggled)")
            elif tag == "status":
                status = "ON" if config.pipeline.think_enabled else "OFF"
                return {
                    "model": payload.get("model", ""),
                    "message": {"role": "assistant", "content": f"Think mode: {status}"},
                    "done": True,
                    "done_reason": "stop",
                }
            # Strip the tag from the message
            msg["content"] = _THINK_TAG_RE.sub("", content).strip()
            break
        return None

    def _user_text_from(payload: dict) -> str:
        """Extract the last user message text from a payload."""
        for msg in reversed(payload.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return content if isinstance(content, str) else json.dumps(content)
        return ""

    def _response_assistant_text(response, api_format: str) -> str:
        """Best-effort: pull the assistant text out of a finalised response.

        Returns "" on streaming responses (no buffered body), parse failures,
        or any other extraction error. Used to hand the assistant reply to
        the writer for LLM episode summarisation.
        """
        if response is None or hasattr(response, "body_iterator"):
            return ""
        body = getattr(response, "body", None)
        if not body:
            return ""
        try:
            from sieve.verification import extract_response_text
            text = extract_response_text(body, api_format)
            return text or ""
        except Exception:
            return ""

    async def _fire_writer(
        user_text: str,
        session_id: str,
        metrics=None,
        assistant_text: str = "",
        phase=None,
    ) -> None:
        """Run writer.process() either sync (OBSERVE) or fire-and-forget (later phases).

        OBSERVE phase aggressively seeds the store — the user has just
        started using Sieve, retrieval results will be thin, and the
        next turn probably references what was just said. Awaiting the
        writer adds ~200ms (S1) to ~2s (S2+episode) per turn but
        guarantees the fresh fact is visible on the next turn. Without
        this, the 5-turn demo (and any brand-new user) races ahead of
        the writer and the assistant replies "I don't know" even though
        the fact was stated one turn ago.

        ACCUMULATE / ACTIVATE keep the fire-and-forget behaviour: by
        then the store has enough baseline that retrieval works even if
        the latest turn's fact is a few hundred ms behind.

        When ``metrics`` is supplied (validation runs) the task
        reference is attached as ``metrics.writer_task`` so the
        response-finalise path can await it via
        ``_await_writer_and_record`` and copy stage-level stats. For
        the OBSERVE-sync path we attach a pre-completed task so the
        same rendezvous code works unchanged.

        ``assistant_text`` lets the writer request a one-sentence
        episode summary from the main model. Empty = fall back to the
        legacy truncation.
        """
        if memory_writer is None or not memory_store._conn or not user_text.strip():
            return

        # OBSERVE: await the writer inline so the fact is committed
        # before the response returns. This is the cold-start fix.
        if phase is not None and phase.label == "OBSERVE":
            try:
                write_result = await memory_writer.process(
                    user_text,
                    assistant_text=assistant_text,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.warning("OBSERVE writer raised: %s", exc)
                write_result = None
            if metrics is not None and write_result is not None:
                # Mirror the rendezvous-helper behaviour: copy stage
                # stats onto the metrics row so validation still sees
                # them on the sync path.
                try:
                    validation_collector.record_writer_result(metrics, write_result)
                except Exception as exc:
                    logger.debug("record_writer_result on sync path failed: %s", exc)
            return

        task = asyncio.create_task(
            memory_writer.process(
                user_text,
                assistant_text=assistant_text,
                session_id=session_id,
            )
        )
        _track_task(task, "writer")
        if metrics is not None:
            metrics.writer_task = task

    def _fire_learning(user_text: str, recall_rounds: int = 0) -> None:
        """Schedule learning signal recording as a background task."""
        if not config.ablation.learning_loop:
            return  # ABL-LL: learning disabled
        if learning_loop is None or not memory_store._conn or not user_text.strip():
            return
        task = asyncio.create_task(
            learning_loop.record_interaction(
                user_query=user_text,
                recall_rounds=recall_rounds,
            )
        )
        _track_task(task, "learning")

    @app.get("/sieve/learning")
    async def learning_stats():
        if learning_loop is None:
            return JSONResponse(status_code=503, content={"error": "Learning loop not initialized"})
        return learning_loop.get_metrics()

    # --- Opt-in validation harness instrumentation ---
    # Each helper is a no-op when validation.enabled is False.

    def _start_validation(request: Request, body: bytes):
        if not config.validation.enabled:
            return None
        try:
            hdr = extract_validation_headers({k.lower(): v for k, v in request.headers.items()})
            # If X-Validation-* headers are missing (Phase B: the agent
            # framework strips them), fall back to the sticky current
            # envelope.
            if not hdr.get("run_id"):
                queued = app.state.validation_current_envelope
                if queued:
                    for key in ("query_id", "run_id", "seed", "simulated_day",
                                "category", "path_label"):
                        if hdr.get(key) in (None, "") and queued.get(key) is not None:
                            hdr[key] = queued[key]
            payload = json.loads(body) if body else {}
            from sieve.validation_collector import _approx_tokens  # local import: avoid top-level cycle
            user_message = _user_text_from(payload)
            inbound = 0
            for msg in payload.get("messages", []):
                c = msg.get("content", "")
                if isinstance(c, str):
                    inbound += _approx_tokens(c)
                else:
                    inbound += _approx_tokens(json.dumps(c))
            tools = payload.get("tools")
            if tools:
                inbound += _approx_tokens(json.dumps(tools))
            metrics = validation_collector.start(
                query_id=hdr["query_id"],
                run_id=hdr["run_id"],
                seed=hdr["seed"],
                simulated_day=hdr["simulated_day"],
                category=hdr["category"],
                user_message=user_message,
                model_name=payload.get("model", ""),
                inbound_tokens=inbound,
                path_label=hdr.get("path_label"),
            )
            return metrics
        except Exception as exc:
            logger.debug("validation start failed: %s", exc)
            return None

    def _record_validation_pipeline(metrics, decomposed, retrieved_context, retrieved_facts, lean, absence_signals=None):
        if metrics is None:
            return
        try:
            # Fingerprint signals — changed=miss, unchanged=hit.
            hits = sum(1 for s in decomposed.sections if not s.changed)
            misses = sum(1 for s in decomposed.sections if s.changed)
            validation_collector.record_fingerprints(metrics, hits=hits, misses=misses)
            # Retrieval snapshot.
            tier = None
            if retrieved_facts:
                tier = "vector+graph"
            elif retrieved_context:
                tier = "context_only"
            validation_collector.record_retrieval(
                metrics,
                tier=tier,
                facts_retrieved=len(retrieved_facts) if retrieved_facts else 0,
                context_block_text=retrieved_context or "",
            )
            # Absence-signal count — use the real signals returned by
            # build_absence_signals (threaded through from _retrieve_context).
            # Fallback to parsing "[NOT PRESENT" markers emitted by the
            # schema_v2 format_context_v2 path for installs running that
            # pathway. Either way, count once.
            sys_texts = [
                m.get("content", "")
                for m in lean.get("messages", [])
                if m.get("role") == "system" and isinstance(m.get("content", ""), str)
            ]
            sys_text = "\n".join(sys_texts)
            if absence_signals:
                ns_count = len(absence_signals)
                categories = [s.reason for s in absence_signals]
            else:
                ns_count = sys_text.count("[NOT PRESENT")
                categories = None
            validation_collector.record_absence_signals(
                metrics, count=ns_count, categories=categories,
            )
            sections = [m.get("role", "?") for m in lean.get("messages", [])]
            validation_collector.record_compose(
                metrics,
                lean_payload=lean,
                sections=sections,
                system_prompt_text=sys_text,
            )
            # Post-pipeline store snapshot.
            validation_collector.snapshot_store(metrics, memory_store)
        except Exception as exc:
            logger.debug("validation pipeline record failed: %s", exc)

    def _wrap_validation_response(response, metrics, t_start, *, api_format: str, recall_rounds: int):
        """Attach streaming observation + persist after response body is produced.

        For StreamingResponse we wrap the body_iterator so we can tap the
        chunks in-flight without buffering the whole stream in memory.
        For non-streaming JSON responses we just parse the body and persist.
        """
        if metrics is None:
            return response
        try:
            # recall_rounds comes from the X-Sieve-Rounds header the handler set.
            metrics.recall_tool_calls = int(recall_rounds or 0)

            if hasattr(response, "body_iterator"):
                # Streaming: wrap the iterator.
                inner = response.body_iterator

                async def _observe():
                    nonlocal inner
                    first_seen = False
                    first_ts = None
                    chunks = 0
                    parts: list[str] = []
                    eval_count = None
                    total_duration_ns = None
                    pending = b""
                    try:
                        async for raw_chunk in inner:
                            yield raw_chunk
                            if isinstance(raw_chunk, bytes):
                                pending += raw_chunk
                            else:
                                pending += raw_chunk.encode()
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                chunks += 1
                                if not first_seen:
                                    first_seen = True
                                    first_ts = (time.perf_counter() - t_start) * 1000.0
                                # Ollama NDJSON or OpenAI SSE "data: " prefix
                                if line.startswith(b"data:"):
                                    line = line[5:].strip()
                                if line == b"[DONE]":
                                    continue
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                if api_format == "ollama":
                                    mc = obj.get("message", {}).get("content")
                                    if mc:
                                        parts.append(mc)
                                    resp_part = obj.get("response")
                                    if resp_part:
                                        parts.append(resp_part)
                                    if obj.get("done"):
                                        eval_count = obj.get("eval_count")
                                        total_duration_ns = obj.get("total_duration")
                                else:  # openai
                                    choices = obj.get("choices") or []
                                    for c in choices:
                                        delta = c.get("delta") or {}
                                        piece = delta.get("content")
                                        if piece:
                                            parts.append(piece)
                                    usage = obj.get("usage") or {}
                                    if usage:
                                        eval_count = usage.get("completion_tokens")
                    finally:
                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        inference_ms = (total_duration_ns / 1_000_000) if total_duration_ns else None
                        validation_collector.record_timing(
                            metrics,
                            ttft_ms=first_ts,
                            total_latency_ms=total_ms,
                            inference_latency_ms=inference_ms,
                            stream_chunks=chunks,
                            http_status=response.status_code,
                            first_token_ts=None,
                            complete_ts=None,
                        )
                        validation_collector.record_response(
                            metrics,
                            content="".join(parts),
                            token_count=eval_count,
                        )
                        # Rendezvous with the background writer (validation only).
                        # The stream has finished delivering to the client, so
                        # awaiting here only delays metrics persistence, not
                        # user-visible latency.
                        await _await_writer_and_record(
                            metrics, getattr(metrics, "writer_task", None),
                            validation_collector,
                        )
                        try:
                            validation_collector.finalise_and_persist(metrics)
                        except Exception as exc:
                            logger.debug("validation persist failed: %s", exc)

                response.body_iterator = _observe()
                return response

            # Non-streaming: parse the body once, record, return unchanged.
            body_bytes = getattr(response, "body", b"") or b""
            total_ms = (time.perf_counter() - t_start) * 1000.0
            content_text = ""
            eval_count = None
            inference_ms = None
            try:
                obj = json.loads(body_bytes) if body_bytes else {}
                if api_format == "ollama":
                    content_text = obj.get("message", {}).get("content") or obj.get("response") or ""
                    eval_count = obj.get("eval_count")
                    td = obj.get("total_duration")
                    inference_ms = (td / 1_000_000) if td else None
                else:
                    choices = obj.get("choices") or []
                    if choices:
                        content_text = choices[0].get("message", {}).get("content", "")
                    usage = obj.get("usage") or {}
                    eval_count = usage.get("completion_tokens")
            except Exception:
                pass
            validation_collector.record_timing(
                metrics,
                ttft_ms=total_ms,
                total_latency_ms=total_ms,
                inference_latency_ms=inference_ms,
                stream_chunks=1,
                http_status=response.status_code,
            )
            validation_collector.record_response(
                metrics, content=content_text, token_count=eval_count,
            )
            # Non-streaming path is synchronous; opportunistically read
            # the writer result only if it already finished. The real
            # validation runs use stream=True, so this branch is rare.
            wtask = getattr(metrics, "writer_task", None)
            if wtask is not None and wtask.done() and not wtask.cancelled():
                try:
                    validation_collector.record_writer_result(metrics, wtask.result())
                except Exception as exc:
                    logger.debug("writer task failed before non-stream record: %s", exc)
            validation_collector.finalise_and_persist(metrics)
            return response
        except Exception as exc:
            logger.debug("validation wrap failed: %s", exc)
            return response

    def _finalise_validation_error(metrics, t_start, message: str):
        if metrics is None:
            return
        try:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            validation_collector.record_timing(
                metrics, total_latency_ms=total_ms,
                http_status=500, error=message,
            )
            validation_collector.finalise_and_persist(metrics)
        except Exception:
            pass

    async def _verify_and_maybe_correct(
        request: Request,
        response,
        lean: dict,
        user_text: str,
        retrieved_facts: list[dict],
        api_format: str,
    ):
        """Layer 3 v3 — verify-facts as tool result (ABL-RV).

        Extracts claims from the generated response, queries the store
        deterministically, and if any corrections are found, injects a
        synthetic verify_facts tool exchange and prompts the model to
        continue with the correct data in hand. Clean responses pass
        through unchanged with <5ms overhead.

        See docs/superpowers/specs/2026-04-13-layer3-v3-verify-facts-design.md
        """
        if not config.ablation.response_verification:
            return response
        if hasattr(response, "body_iterator"):
            return response  # streaming — skip (would require buffering the stream)
        body = getattr(response, "body", None)
        if not body:
            return response
        from sieve.verification import extract_response_text, replace_response_text
        from sieve.verify_v3 import (
            build_continuation_payload,
            build_tool_result,
            splice_response,
            verify_response_v3,
        )
        try:
            text = extract_response_text(body, api_format)
            if not text:
                return response
            from sieve.verification import _owner_alias_set
            v = verify_response_v3(
                text,
                memory_store,
                owner_aliases=_owner_alias_set(config.profile_owner),
            )
            if v.is_clean:
                return response
            logger.info(
                "ABL-RV(v3): flagged %d attr + %d fab",
                len(v.attributes), len(v.fabricated),
            )
            tool_result = build_tool_result(v, store=memory_store)
            continuation_payload = build_continuation_payload(
                lean_payload=lean,
                original_assistant_text=text,
                original_assistant_message=None,
                tool_result=tool_result,
                shape="tool_role",
                continuation_max_tokens=200,
                api_format=api_format,
            )
            cont_resp = await forward_payload(request, proxy_client, continuation_payload)
            cont_body = getattr(cont_resp, "body", None)
            if not cont_body:
                return response
            continuation = extract_response_text(cont_body, api_format) or ""
            final_text = splice_response(v, continuation)
            patched_body = replace_response_text(body, api_format, final_text)
            response.body = patched_body
            response.headers["X-Sieve-RV-Corrected"] = "1"
            flag_tags = (
                [f"{{{a.subject}:{a.predicate}}}" for a in v.attributes]
                + [f"<{f.anchor}:{f.relationship}>" for f in v.fabricated]
            )
            response.headers["X-Sieve-RV-Flagged"] = ",".join(flag_tags)[:200]
            import base64 as _b64
            try:
                orig_b64 = _b64.b64encode(text[:8000].encode("utf-8")).decode("ascii")
                response.headers["X-Sieve-RV-Original-B64"] = orig_b64
            except Exception:
                pass
            response.headers["content-length"] = str(len(patched_body))
            return response
        except Exception as exc:
            logger.warning("ABL-RV(v3) failed: %s", exc)
            return response

    async def _retrieve_context(
        user_text: str,
        decomposed: Any = None,
    ) -> tuple[str, list[dict], list, bool, str | None]:
        """Classify query; if context needed, retrieve and return
        (block, facts, signals, is_pure_general, narrative_summary).

        Args:
            user_text: current user query.
            decomposed: optional DecomposedPayload. When present AND
                ablation.extreme_summary is on, sections that are about
                to be stripped (system_prompt, tools, workspace_files,
                older conversation_history) are compressed into a
                [NARRATIVE SUMMARY] section that rides alongside the
                structured facts. Needed for EXTREME bloat where
                temporal/causal reasoning depth can't be reconstructed
                from slot lookups alone.

        Returns:
            (context_block_text, retrieved_facts, absence_signals,
             is_pure_general, narrative_summary) — facts and signals are
            empty lists if no retrieval happened. The text already includes
            Layer 1 absence signals and Layer 2 closed-world framing if those
            flags are enabled; signals is returned separately so validation
            telemetry can count accurately. is_pure_general is True when the
            L0 classifier confidently said "no retrieval" (pure general
            knowledge) — the caller uses this to decide whether to keep
            injecting the recall tool. narrative_summary (Audit Fix #3) is
            the extreme_summary narrative string or None; the caller injects
            it into the lean system prompt (never trimmed by
            _apply_token_budget) rather than into the retrieved-context block.
        """
        # ABL-PP: skip pre-population entirely
        if not config.ablation.pre_populate:
            logger.info("ABL-PP: pre-population disabled")
            return "", [], [], False, None
        if classifier is None or retriever is None or not memory_store._conn:
            return "", [], [], False, None
        if not user_text.strip():
            return "", [], [], False, None

        # D4: meta-questions about Sieve's own state ("how many facts do
        # you know about me?", "what do you know about me?") — without
        # this, retrieval returns the top-k query-similar facts and the
        # LLM reports THAT count as its knowledge. The first 30-day run
        # answered "14 facts" when the store actually had 104.
        _META_COUNT_PATTERN = re.compile(
            r"\bhow many (facts|things|details|pieces (of info)?) "
            r"(do|are|have)\s+you (know|have|stored|remember)", re.IGNORECASE,
        )
        if _META_COUNT_PATTERN.search(user_text):
            try:
                fact_total = memory_store.count_current_facts()
            except Exception:
                fact_total = 0
            meta_block = (
                f"[META] You currently have {fact_total} facts stored about the user "
                f"in your persistent memory. When answering meta questions about what "
                f"you know, cite this number.\n"
            )
            logger.info("D4 meta-question injected: %d facts", fact_total)
            return meta_block, [], [], False, None

        # EXTREME narrative summary — computed up-front so it
        # applies whether or not the classifier decides retrieval is
        # needed. A 42K-token inbound with no personal signals still
        # benefits from a narrative digest; skipping it just because
        # the classifier said "no retrieval" would defeat the purpose.
        extreme_summary_text: str = ""
        if config.ablation.extreme_summary and decomposed is not None:
            try:
                from sieve.extreme_summary import (
                    format_summary_section,
                    should_summarise,
                    summarise_async,
                )
                strip_names = (
                    "system_prompt", "tools", "workspace_files",
                    "conversation_history",
                )
                parts: list[str] = []
                for name in strip_names:
                    sec = decomposed.section_by_name(name)
                    if sec and sec.content:
                        parts.append(f"[{name}]\n{sec.content}")
                stripped_text = "\n\n".join(parts)
                if should_summarise(stripped_text, enabled=True):
                    eff_model = resolve_writer_model(config)
                    summary = await summarise_async(
                        stripped_text,
                        base_url=config.provider.base_url,
                        model=eff_model,
                        # Narrative summariser reuses writer.num_ctx (both are
                        # post-retrieval context cleanup); capped at 32768 to
                        # avoid excessive latency costs on cloud-model wide windows.
                        num_ctx=min(config.writer.num_ctx, 32768),
                    )
                    if summary:
                        extreme_summary_text = format_summary_section(summary)
                        logger.info(
                            "ABL-XS: built narrative summary (~%d chars)",
                            len(summary),
                        )
            except Exception as exc:
                logger.warning("extreme summary build failed: %s", exc)

        try:
            # Slot-first retrieval. When schema_v2 is on, run the
            # deterministic SlotRetriever BEFORE the L1 classifier
            # so contradiction queries ("is Jamie still married?") that
            # would otherwise fall below the L1 sim threshold still get
            # the slot lookup. SlotRetriever is deterministic and cheap
            # (~1ms of regex + SQL). If it hits we BYPASS the L1 gate
            # for legacy retrieval and always run legacy too — the two
            # outputs are merged by format_context_v2. If it misses we
            # fall through to the existing classifier gate unchanged.
            slot_result: Any = None
            slot_hit = False
            if config.ablation.schema_v2 and slot_retriever is not None:
                try:
                    slot_result = slot_retriever.retrieve(user_text)
                    slot_hit = bool(slot_result and slot_result.is_hit)
                    if slot_hit:
                        logger.info(
                            "schema_v2 slot-first HIT: class=%s slot=%s",
                            slot_result.query_class,
                            slot_result.slot_key,
                        )
                except Exception as exc:
                    logger.warning("schema_v2 slot-first path failed: %s", exc)

            # Decide whether to run legacy retrieval. A slot hit bypasses
            # the L1 gate (we're confident retrieval is worth doing). No
            # slot hit still honours the L1 gate as before.
            pre_pop_k = config.pipeline.pre_populate_top_k
            # Phase-3 Fix 1: dynamic top-K. Complexity-aware budget:
            #   0 (trivial/general)   → 3   — lean as possible
            #   1 (simple personal)   → pre_populate_top_k (config, default 5)
            #   2 (complex/follow-up) → 10  — multi-hop & follow-up need
            #                                  8-10 facts to synthesise
            _TOP_K_BY_COMPLEXITY = {0: 3, 1: pre_pop_k, 2: 10}
            ctx = None
            # complexity tracked so Fix 2 can route follow-up queries to
            # the episode-retrieval path below.
            decision_complexity = 1
            is_followup = False
            if slot_hit:
                # Run legacy unconditionally so we get rich ctx.facts
                # for downstream absence signals and the v2 formatter's
                # [SUPPORTING FACTS] section. This is cheap (~30-50ms)
                # relative to the model call that follows.
                ctx = await retriever.retrieve(user_text, top_k=pre_pop_k)
            elif not config.ablation.classifier:
                logger.info("ABL-CL: classifier disabled, always retrieving")
                ctx = await retriever.retrieve(user_text, top_k=pre_pop_k)
            else:
                decision = await classifier.classify(user_text)
                decision_complexity = getattr(decision, "complexity", 1)
                from sieve.classifier import _FOLLOWUP_MARKERS as _FM
                is_followup = bool(_FM.search(user_text))
                dyn_k = _TOP_K_BY_COMPLEXITY.get(decision_complexity, pre_pop_k)
                logger.info(
                    "Classify: needs_retrieval=%s level=L%d reason=%r "
                    "complexity=%d top_k=%d followup=%s",
                    decision.needs_retrieval, decision.level, decision.reason,
                    decision_complexity, dyn_k, is_followup,
                )
                if decision.needs_retrieval:
                    # Decompose complexity=2 (multi-hop / complex
                    # follow-up) queries into 2-4 sub-queries and merge
                    # their top-3 results. Other complexities hit the
                    # single-query path unchanged.
                    if (
                        decision_complexity == 2
                        and config.retrieval.query_decomposition_enabled
                    ):
                        from sieve.query_decomposer import decompose_query
                        writer_model = resolve_writer_model(config)
                        try:
                            sub_queries = await decompose_query(
                                user_text,
                                provider_base_url=config.provider.base_url,
                                model=writer_model,
                                num_ctx=min(config.writer.num_ctx, 4096),
                            )
                        except Exception as exc:
                            logger.info("Decompose call raised, falling back: %s", exc)
                            sub_queries = []
                        if sub_queries:
                            logger.info(
                                "Multi-hop decompose: %d sub-queries: %r",
                                len(sub_queries), sub_queries,
                            )
                            ctx = await retriever.retrieve_multi(
                                sub_queries,
                                per_query_top_k=3,
                                final_top_k=dyn_k,
                                include_episodes=is_followup,
                            )
                        else:
                            ctx = await retriever.retrieve(
                                user_text,
                                top_k=dyn_k,
                                include_episodes=is_followup,
                            )
                    else:
                        ctx = await retriever.retrieve(
                            user_text,
                            top_k=dyn_k,
                            include_episodes=is_followup,
                        )
                else:
                    # Even with no retrieval, still return the summary
                    # section so the model sees the narrative digest.
                    # Mark as pure general-knowledge when the L0
                    # classifier is confident — caller uses this to
                    # strip the recall tool definition from the lean
                    # payload (saves ~99 tokens/query on G queries).
                    is_pure_general = (
                        decision.level == 0 and decision.confidence >= 0.8
                    )
                    # D25 guard: even when the classifier says "no retrieval",
                    # downgrade pure-general whenever the store has
                    # accumulated any meaningful history. A classifier
                    # miss (e.g. "Write a bio about me" routed as pure-task)
                    # would otherwise strip memory framing entirely and the
                    # model would reply "I have no memory of you" on its
                    # own user's data. Threshold: ≥10 facts means there's
                    # enough personal context to risk being wrong.
                    if is_pure_general and memory_store is not None:
                        try:
                            fact_total = memory_store.count_current_facts()
                        except Exception:
                            fact_total = 0
                        if fact_total >= 10:
                            logger.info(
                                "Pure-G downgraded: store has %d facts, "
                                "keeping memory framing (D25)",
                                fact_total,
                            )
                            is_pure_general = False
                    # Audit Fix #3: return narrative as its own 5th element so
                    # the caller can inject it into the lean system prompt
                    # (never trimmed). Previously it was returned as the text
                    # slot and would vanish if _apply_token_budget dropped the
                    # retrieved-context message.
                    return "", [], [], is_pure_general, extreme_summary_text or None

            assert ctx is not None  # all branches above either assign ctx or return
            text = ctx.text
            facts = ctx.facts

            # If the slot-first path hit, render via format_context_v2
            # with ctx.facts as supporting facts. Otherwise apply the
            # existing post-hoc schema_v2 merge path.
            if slot_hit and slot_result is not None:
                try:
                    from sieve.context_format_v2 import format_context_v2
                    v2_text, v2_tok = format_context_v2(
                        slot_result,
                        profile_owner_name=config.profile_owner.name,
                        extra_facts=facts,
                        max_tokens=800,
                    )
                    logger.info(
                        "schema_v2 slot-first RENDERED: tokens=%d "
                        "(slot_rows=%d supporting=%d)",
                        v2_tok, len(slot_result.current_slots), len(facts),
                    )
                    text = v2_text
                except Exception as exc:
                    logger.warning(
                        "schema_v2 slot-first render failed, using legacy: %s",
                        exc,
                    )
            absence_signals: list = []
            if config.ablation.absence_signal and text:
                from sieve.verification import build_absence_signals
                # Q64 widening: pass the last 3 conversation turns so
                # the absence layer can see entities the user just
                # introduced this session (e.g. "my daughter Lily") and
                # suppress false-negative signals against them.
                recent_turns: list[dict] = []
                if decomposed is not None:
                    hist_section = decomposed.section_by_name("conversation_history")
                    if hist_section and hist_section.content:
                        try:
                            history_msgs = json.loads(hist_section.content)
                            if isinstance(history_msgs, list):
                                recent_turns = history_msgs[-6:]  # 3 user + 3 assistant
                        except (json.JSONDecodeError, TypeError):
                            pass
                absence_signals = build_absence_signals(
                    user_text, facts, memory_store,
                    recent_turns=recent_turns,
                    profile_owner=config.profile_owner,
                )
                if absence_signals:
                    extra = "\n".join(s.text for s in absence_signals)
                    text = text + "\n" + extra
                    logger.info("ABL-AS: injected %d absence signal(s)", len(absence_signals))

            # Layer 2 — Closed-World Framing (ABL-CW)
            if config.ablation.closed_world and text:
                from sieve.verification import CLOSED_WORLD_FRAMING
                text = text + CLOSED_WORLD_FRAMING
                logger.info("ABL-CW: closed-world framing appended")

            # Audit Fix #3: narrative summary is returned as its own 5th
            # element so the caller injects it into the lean system prompt
            # (messages[0]) — a location _apply_token_budget never trims.
            # Previously it was appended into the retrieved-context block and
            # would be halved or dropped by aggressive token budget trimming,
            # causing the model to confabulate on trap queries like
            # "When did I graduate from Oxford?" (OpenClaw 30-day run).
            if extreme_summary_text:
                logger.info("ABL-XS: narrative summary emitted separately (Audit Fix #3)")

            return text, facts, absence_signals, False, extreme_summary_text or None
        except Exception as exc:
            logger.warning("Classify/retrieve failed: %s", exc)
            return "", [], [], False, None

    async def _maybe_ingest_tools(payload: dict, decomposed) -> None:
        if tool_registry is None:
            return
        tools_section = decomposed.section_by_name("tools")
        if tools_section is None or not tools_section.changed:
            return
        tools_list = payload.get("tools") or []
        try:
            await tool_registry.ingest(tools_list)
        except Exception as exc:
            logger.warning("Tool registry ingest failed: %s", exc)

    def _payload_tokens(payload: dict) -> int:
        """Rough token count for a chat payload (messages + tools)."""
        from sieve.validation_collector import _approx_tokens
        total = 0
        for msg in payload.get("messages", []) or []:
            c = msg.get("content", "")
            if isinstance(c, str):
                total += _approx_tokens(c)
            else:
                total += _approx_tokens(json.dumps(c))
        tools = payload.get("tools")
        if tools:
            total += _approx_tokens(json.dumps(tools))
        return total

    def _attach_token_headers(response, inbound_tokens: int, outbound_tokens: int) -> None:
        """Expose inbound/outbound token counts so middleware + CLI benchmarks can observe Sieve's effect."""
        try:
            response.headers["X-Sieve-Inbound-Tokens"] = str(inbound_tokens)
            response.headers["X-Sieve-Outbound-Tokens"] = str(outbound_tokens)
        except Exception:
            pass

    def _attach_phase_header(response, phase) -> None:
        """Expose progressive-activation phase + fact count on the response.

        Observable by the CLI demo, benchmark, and tests so the active
        phase is visible without having to tail proxy logs.
        """
        try:
            response.headers["X-Sieve-Phase"] = phase.label
            response.headers["X-Sieve-Fact-Count"] = str(phase.fact_count)
        except Exception:
            pass

    def _detect_current_phase():
        """Read the store once per request and derive the active phase.

        Counts ``status='current'`` facts in the store and maps to one of
        OBSERVE / ACCUMULATE / ACTIVATE via the configured thresholds.
        Store read failures log and default to OBSERVE so the proxy still
        delivers a safe, conservative payload if the DB is momentarily
        unavailable.
        """
        try:
            fact_count = memory_store.count_current_facts() if memory_store._conn is not None else 0
        except Exception as exc:
            logger.warning("phase detection: fact count failed (%s); defaulting to 0", exc)
            fact_count = 0
        return detect_phase(fact_count, config.progression)

    # --- Intercepted chat endpoints (Phase 4+5: strip + compose + write + forward) ---

    @app.post("/api/chat")
    @app.post("/api/generate")
    async def intercept_ollama_chat(request: Request):
        """Intercept Ollama chat — decompose, classify, retrieve, compose, forward + recall loop."""
        body = await request.body()
        # Validation instrumentation (opt-in; returns None when disabled).
        val_metrics = _start_validation(request, body)
        t_start = time.perf_counter()
        try:
            payload = json.loads(body)
            # Think mode tag interception (before pipeline)
            think_response = _process_think_tags(payload)
            if think_response is not None:
                return JSONResponse(content=think_response)
            # Some agent frameworks ship chat history INSIDE the user
            # content. Lift it out into proper message-level turns so
            # the standard conversation_history strip / last-N-turns
            # logic applies; otherwise the outbound grows linearly with
            # the run.
            adapt_history_preamble_payload(payload)
            user_text = _user_text_from(payload)
            model = payload.get("model", "")
            mode = detect_mode(model, config.mode_override)
            # ABL-FP: when fingerprinting disabled, use ephemeral cache (re-parse every request)
            fp_cache = fingerprint_cache if config.ablation.fingerprinting else FingerprintCache(None)
            decomposed = decompose(payload, fp_cache, api_format="ollama")
            await _maybe_ingest_tools(payload, decomposed)
            retrieved_context, retrieved_facts, absence_signals, is_pure_general, narrative_summary = await _retrieve_context(user_text, decomposed)
            phase = _detect_current_phase()
            logger.info("Phase %s", phase.render_tag())
            if tool_classifier is not None and config.tools.enabled:
                lean = await compose_with_tool_selection(
                    payload, decomposed, config.pipeline,
                    tool_classifier=tool_classifier,
                    user_query=user_text,
                    retrieved_context=retrieved_context,
                    profile_owner_pin=config.profile_owner.pin,
                    pure_general=is_pure_general,
                    progression=phase,
                    narrative_summary=narrative_summary,
                )
            else:
                lean = compose_lean_payload(
                    payload, decomposed, config.pipeline,
                    retrieved_context=retrieved_context,
                    profile_owner_pin=config.profile_owner.pin,
                    pure_general=is_pure_general,
                    progression=phase,
                    narrative_summary=narrative_summary,
                )
            # ABL-RT: don't inject recall tool → strip it from lean
            if not config.ablation.recall_tool:
                lean_tools = lean.get("tools", [])
                lean["tools"] = [t for t in lean_tools
                                 if not (isinstance(t, dict) and t.get("function", {}).get("name") == "recall")]
                logger.info("ABL-RT: recall tool stripped")
            # Pure general-knowledge queries never benefit from the recall
            # tool definition — strip it to save ~99 tokens/query.
            if is_pure_general:
                lean_tools = lean.get("tools", [])
                lean["tools"] = [t for t in lean_tools
                                 if not (isinstance(t, dict) and t.get("function", {}).get("name") == "recall")]
                logger.info("Pure-G query: recall tool stripped")
            # Mode B: strip tools (no tool-calling capability)
            if mode == "B":
                lean.pop("tools", None)
                logger.info("Mode B: tools stripped for model=%s", model)
            _record_validation_pipeline(val_metrics, decomposed, retrieved_context, retrieved_facts, lean, absence_signals=absence_signals)
            use_recall_handler = (
                recall_handler is not None
                and mode == "A"
                and config.ablation.recall_tool
                and not is_pure_general
            )
            if use_recall_handler:
                response = await recall_handler.handle_chat(
                    request, lean, api_format="ollama",
                )
            else:
                response = await forward_payload(request, proxy_client, lean)
            # Layer 3: post-generation response verification
            response = await _verify_and_maybe_correct(
                request, response, lean, user_text, retrieved_facts, "ollama",
            )
            recall_rounds = int(response.headers.get("X-Sieve-Rounds", "0"))
            assistant_text = _response_assistant_text(response, "ollama")
            await _fire_writer(
                user_text, session_id=uuid.uuid4().hex, metrics=val_metrics,
                assistant_text=assistant_text, phase=phase,
            )
            _fire_learning(user_text, recall_rounds=recall_rounds)
            response = _wrap_validation_response(
                response, val_metrics, t_start, api_format="ollama",
                recall_rounds=recall_rounds,
            )
            _attach_token_headers(response, _payload_tokens(payload), _payload_tokens(lean))
            _attach_phase_header(response, phase)
            return response
        except Exception as exc:
            logger.warning("Pipeline failed, forwarding original payload: %s", exc)
            _finalise_validation_error(val_metrics, t_start, str(exc))
            try:
                fallback_body = json.loads(body)
                return await forward_payload(request, proxy_client, fallback_body)
            except Exception:
                return await forward_request(request, proxy_client)

    @app.post("/v1/chat/completions")
    async def intercept_openai_chat(request: Request):
        """Intercept OpenAI chat — decompose, classify, retrieve, compose, forward + recall loop."""
        body = await request.body()
        val_metrics = _start_validation(request, body)
        t_start = time.perf_counter()
        try:
            payload = json.loads(body)
            # Think mode tag interception (before pipeline)
            think_response = _process_think_tags(payload)
            if think_response is not None:
                return JSONResponse(content=think_response)
            # Lift any history-in-user-content preamble into proper turns.
            adapt_history_preamble_payload(payload)
            user_text = _user_text_from(payload)
            model = payload.get("model", "")
            mode = detect_mode(model, config.mode_override)
            fp_cache = fingerprint_cache if config.ablation.fingerprinting else FingerprintCache(None)
            decomposed = decompose(payload, fp_cache, api_format="openai")
            await _maybe_ingest_tools(payload, decomposed)
            retrieved_context, retrieved_facts, absence_signals, is_pure_general, narrative_summary = await _retrieve_context(user_text, decomposed)
            phase = _detect_current_phase()
            logger.info("Phase %s", phase.render_tag())
            if tool_classifier is not None and config.tools.enabled:
                lean = await compose_with_tool_selection(
                    payload, decomposed, config.pipeline,
                    tool_classifier=tool_classifier,
                    user_query=user_text,
                    retrieved_context=retrieved_context,
                    profile_owner_pin=config.profile_owner.pin,
                    pure_general=is_pure_general,
                    progression=phase,
                    narrative_summary=narrative_summary,
                )
            else:
                lean = compose_lean_payload(
                    payload, decomposed, config.pipeline,
                    retrieved_context=retrieved_context,
                    profile_owner_pin=config.profile_owner.pin,
                    pure_general=is_pure_general,
                    progression=phase,
                    narrative_summary=narrative_summary,
                )
            # ABL-RT: strip recall tool
            if not config.ablation.recall_tool:
                lean_tools = lean.get("tools", [])
                lean["tools"] = [t for t in lean_tools
                                 if not (isinstance(t, dict) and t.get("function", {}).get("name") == "recall")]
            # Pure G queries: strip recall tool (saves ~99 tokens/query)
            if is_pure_general:
                lean_tools = lean.get("tools", [])
                lean["tools"] = [t for t in lean_tools
                                 if not (isinstance(t, dict) and t.get("function", {}).get("name") == "recall")]
                logger.info("Pure-G query: recall tool stripped")
            # Mode B: strip all tools
            if mode == "B":
                lean.pop("tools", None)
                logger.info("Mode B: tools stripped for model=%s", model)
            _record_validation_pipeline(val_metrics, decomposed, retrieved_context, retrieved_facts, lean, absence_signals=absence_signals)
            use_recall_handler = (
                recall_handler is not None
                and mode == "A"
                and config.ablation.recall_tool
                and not is_pure_general
            )
            if use_recall_handler:
                response = await recall_handler.handle_chat(
                    request, lean, api_format="openai",
                )
            else:
                response = await forward_payload(request, proxy_client, lean)
            # Layer 3: post-generation response verification
            response = await _verify_and_maybe_correct(
                request, response, lean, user_text, retrieved_facts, "openai",
            )
            recall_rounds = int(response.headers.get("X-Sieve-Rounds", "0"))
            assistant_text = _response_assistant_text(response, "openai")
            await _fire_writer(
                user_text, session_id=uuid.uuid4().hex, metrics=val_metrics,
                assistant_text=assistant_text, phase=phase,
            )
            _fire_learning(user_text, recall_rounds=recall_rounds)
            response = _wrap_validation_response(
                response, val_metrics, t_start, api_format="openai",
                recall_rounds=recall_rounds,
            )
            _attach_token_headers(response, _payload_tokens(payload), _payload_tokens(lean))
            _attach_phase_header(response, phase)
            return response
        except Exception as exc:
            logger.warning("Pipeline failed, forwarding original payload: %s", exc)
            _finalise_validation_error(val_metrics, t_start, str(exc))
            try:
                fallback_body = json.loads(body)
                return await forward_payload(request, proxy_client, fallback_body)
            except Exception:
                return await forward_request(request, proxy_client)

    # --- Catch-all proxy (everything else) ---

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"])
    async def proxy_passthrough(request: Request, path: str):
        try:
            return await forward_request(request, proxy_client)
        except Exception as exc:
            logger.error("Proxy error: %s", exc)
            return JSONResponse(
                status_code=502,
                content={"error": f"Upstream error: {exc}"},
            )

    return app


# For `uvicorn sieve.main:app`
app = create_app()

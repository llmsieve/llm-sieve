"""Tiny stub OpenAI-compatible provider for vendor-agnostic GIF recording.

Listens on 127.0.0.1:8765 by default. Implements just enough of the
OpenAI v1 API surface that sieve-install can probe it, list models,
and complete a one-turn chat completion for the recording.

Usage during GIF recording:

    python branding/stub_provider.py &
    # then run sieve-install, picking option 3 (OpenAI-compatible)
    # and pointing it at http://127.0.0.1:8765/v1

The render-gifs.sh wrapper can start + stop this automatically when
passed `--stub`.

This stub is NOT a production component — it exists purely to make
sieve-install's first-run flow renderable without depending on any
real LLM provider being available on the recording host.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import time
from typing import Any

PORT = int(os.environ.get("STUB_PORT", "8765"))


def _json_response(handler, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class StubHandler(http.server.BaseHTTPRequestHandler):
    """Minimal OpenAI-compat + Ollama-compat surface for recording."""

    def log_message(self, format, *args):
        # Quiet — no per-request log noise in the GIF.
        return

    def do_GET(self):
        path = self.path.rstrip("/")
        # OpenAI-style: GET /v1/models
        if path.endswith("/v1/models"):
            _json_response(self, 200, {
                "object": "list",
                "data": [
                    {"id": "demo-model", "object": "model",
                     "created": int(time.time()), "owned_by": "demo"},
                ],
            })
            return
        # Ollama-style: GET /api/tags
        if path.endswith("/api/tags"):
            _json_response(self, 200, {
                "models": [
                    {"name": "demo-model", "size": 1_000_000_000,
                     "modified_at": "2026-06-01T00:00:00Z"},
                ],
            })
            return
        # Fallback OK so probes succeed.
        _json_response(self, 200, {"ok": True})

    def do_POST(self):
        path = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length", "0"))
        try:
            self.rfile.read(length)  # consume body
        except Exception:
            pass

        # OpenAI-style chat completion
        if path.endswith("/v1/chat/completions"):
            _json_response(self, 200, {
                "id": "demo-chatcmpl-001",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "demo-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": "Demo response from stub."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6,
                          "total_tokens": 18},
            })
            return

        # Ollama-style chat
        if path.endswith("/api/chat"):
            _json_response(self, 200, {
                "model": "demo-model",
                "created_at": "2026-06-01T00:00:00Z",
                "message": {"role": "assistant",
                            "content": "Demo response from stub."},
                "done": True,
                "done_reason": "stop",
            })
            return

        _json_response(self, 200, {"ok": True})


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), StubHandler)
    print(f"Stub provider listening on http://127.0.0.1:{PORT}", file=sys.stderr)
    print(f"  • OpenAI-compat: http://127.0.0.1:{PORT}/v1", file=sys.stderr)
    print(f"  • Ollama-compat: http://127.0.0.1:{PORT}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

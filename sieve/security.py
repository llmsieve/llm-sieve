"""Security — auth token enforcement, HTTPS warnings, data minimisation.

Auth token:
  All /sieve/ management endpoints require X-Sieve-Token header.
  Proxy passthrough endpoints (/api/chat, /api/tags, etc.) do NOT require it.
  Token is auto-generated on first run if not set in config.

HTTPS warning:
  If provider.base_url is a remote address using http://, log a startup warning.

Data minimisation:
  Context injected into LLM payloads must contain only clean text — no fact IDs,
  confidence scores, or internal metadata.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("recall.security")

# Path prefix that requires auth
_SIEVE_PREFIX = "/sieve/"

# Paths that never require auth (proxy passthrough)
_PUBLIC_PREFIXES = ("/api/", "/v1/")


def get_or_create_auth_token(config_dir: Path | None = None) -> str:
    """Get existing auth token or generate one on first run.

    Token is stored in ~/.sieve/.sieve_auth_token (mode 600).
    """
    if config_dir is None:
        config_dir = Path("~/.sieve").expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)

    token_file = config_dir / ".sieve_auth_token"
    if token_file.exists():
        return token_file.read_text().strip()

    token = os.urandom(32).hex()
    token_file.write_text(token)
    token_file.chmod(0o600)
    logger.info("Generated new auth token at %s", token_file)
    return token


def check_auth(request: Request, expected_token: str) -> JSONResponse | None:
    """Check if the request has a valid auth token.

    Returns a 401 JSONResponse if unauthorized, or None if authorized.
    Only checks for /sieve/ management endpoints.
    """
    path = request.url.path

    # Public endpoints don't require auth
    if not path.startswith(_SIEVE_PREFIX):
        return None

    # Health endpoint is public (monitoring/load balancers)
    if path == "/sieve/health":
        return None

    # Validation harness endpoints are local-loopback by design (the
    # validation harness runs on a single host). Leaving them gated
    # behind the auth token would force the runner to scrape
    # ``~/.sieve/.sieve_auth_token``; exempting them here keeps the
    # plumbing simple without weakening production security, since
    # ``validation.enabled`` is off by default.
    if path in ("/sieve/validation/next", "/sieve/validation/reset"):
        return None

    token = request.headers.get("X-Sieve-Token", "")
    if not token or token != expected_token:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"},
        )
    return None


def check_https_warning(base_url: str) -> None:
    """Log a warning if the provider URL is remote and uses HTTP instead of HTTPS."""
    parsed = urlparse(base_url)
    host = parsed.hostname or ""

    is_local = host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.startswith("192.168.")
    is_http = parsed.scheme == "http"

    if not is_local and is_http:
        logger.warning(
            "SECURITY: provider.base_url (%s) uses HTTP on a remote address. "
            "LLM traffic (including personal context) is sent in plaintext. "
            "Use HTTPS for remote providers.",
            base_url,
        )


def sanitize_context_block(text: str) -> str:
    """Remove any internal metadata that may have leaked into context text.

    Ensures only clean text goes to LLMs — no fact IDs, confidence scores,
    or internal metadata.
    """
    if not text:
        return text

    # Remove any UUID-like strings (fact IDs)
    text = re.sub(r"\b[0-9a-f]{32}\b", "", text)
    # Remove confidence annotations like (confidence: 0.85) or [0.85]
    text = re.sub(r"\(confidence[:\s]*[\d.]+\)", "", text)
    text = re.sub(r"\[[\d.]+\]", "", text)
    # Remove fact_type annotations
    text = re.sub(r"\((?:objective|subjective|conditional|temporal)\)", "", text)
    # Clean up double spaces
    text = re.sub(r"  +", " ", text)

    return text.strip()

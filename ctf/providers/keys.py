"""Secure API-key handling.

Resolution order: process env -> .env (gitignored) -> OS keyring.
Keys are held only for the request, passed only in the Authorization header to
the provider's official TLS endpoint, and NEVER written to a URL, log, cache key,
error, or artifact. A global redaction filter is defense-in-depth for logs.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# sk-... style keys, and long opaque tokens. Deliberately conservative to avoid
# redacting ordinary prose, while still masking anything key-shaped in logs.
_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{8,}|[A-Za-z0-9_\-]{32,})\b")


def _load_dotenv() -> None:
    p = Path(".env")
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_key(provider: str) -> str | None:
    env = _ENV.get(provider)
    if not env:
        return None
    if os.environ.get(env):
        return os.environ[env]
    _load_dotenv()
    if os.environ.get(env):
        return os.environ[env]
    try:
        import keyring

        return keyring.get_password("build-break-models", env)
    except Exception:
        return None


def redact(text: str) -> str:
    return _KEY_RE.sub("***REDACTED***", text)


class _RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(str(record.msg))
        except Exception:
            pass
        return True


def install_redaction_filter() -> None:
    """Attach key-redaction to the root logger. Idempotent-ish; call once at startup."""
    root = logging.getLogger()
    if not any(isinstance(f, _RedactionFilter) for f in root.filters):
        root.addFilter(_RedactionFilter())

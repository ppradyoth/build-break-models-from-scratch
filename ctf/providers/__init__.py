"""Optional paid-provider layer — secure, benign enrichment ONLY.

Hard boundary (docs/THREAT-MODEL.md):
  * The exploit/flag path is always local. No flag ever depends on a provider call.
  * Offensive payloads never leave the local toy model. This layer exposes only
    benign ops (embed / tokenize / benign generate); it must not be used to attack
    a third-party API.
  * Paid calls are opt-in (CTF_ALLOW_PAID=1) and consented (call-count notice).

Level 1 uses no providers. The paid adapters land with Level 2; the secure key
handling and consent/cache/redaction machinery ships now.
"""
from __future__ import annotations

from .keys import install_redaction_filter, redact, resolve_key
from .registry import available_providers, get_provider

__all__ = [
    "available_providers",
    "get_provider",
    "resolve_key",
    "redact",
    "install_redaction_filter",
]

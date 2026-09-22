"""Local response cache to minimize spend. Stores provider RESPONSES only —
never API keys, and cache keys never include key material.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_DIR = Path(".provider-cache")


def _key(provider: str, model: str, payload: object) -> str:
    blob = f"{provider}|{model}|{json.dumps(payload, sort_keys=True, default=str)}"
    return hashlib.sha256(blob.encode()).hexdigest()


def get(provider: str, model: str, payload: object):
    f = _DIR / f"{_key(provider, model, payload)}.json"
    return json.loads(f.read_text()) if f.exists() else None


def put(provider: str, model: str, payload: object, response: object) -> None:
    _DIR.mkdir(exist_ok=True)
    (_DIR / f"{_key(provider, model, payload)}.json").write_text(json.dumps(response))

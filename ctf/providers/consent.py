"""Opt-in gate for paid calls. Never spend the user's money silently."""
from __future__ import annotations

import os


def allow_paid() -> bool:
    return os.environ.get("CTF_ALLOW_PAID") == "1"


def confirm(n_calls: int, provider: str, model: str) -> None:
    if not allow_paid():
        raise RuntimeError(
            "Paid provider calls are opt-in. Set CTF_ALLOW_PAID=1 (after reading "
            "docs/THREAT-MODEL.md) and pass an explicit --provider to enable them."
        )
    print(f"[providers] about to make {n_calls} paid call(s) to {provider}/{model}.")

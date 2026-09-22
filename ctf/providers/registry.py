from __future__ import annotations

from .keys import resolve_key

_PAID = ["openai", "anthropic", "google", "cohere", "mistral"]


def available_providers() -> list[str]:
    """Only providers whose key resolves are advertised (plus always-on local)."""
    return ["local"] + [p for p in _PAID if resolve_key(p)]


def get_provider(name: str):
    if name == "local":
        from .local import LocalProvider

        return LocalProvider()
    if name in _PAID:
        if not resolve_key(name):
            raise RuntimeError(
                f"No API key for '{name}'. Set {name.upper()}_API_KEY via env, .env, "
                f"or OS keyring. See docs/THREAT-MODEL.md."
            )
        raise NotImplementedError(
            f"Paid adapter '{name}' arrives with Level 2. Secure key handling is ready; "
            f"the HTTP client is not yet wired."
        )
    raise ValueError(f"Unknown provider '{name}'")

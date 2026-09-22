from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    name: str

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...

    def generate(self, prompt: str, *, model: str, **decode: object) -> str: ...

    def tokenize(self, text: str, *, model: str) -> list[int]: ...

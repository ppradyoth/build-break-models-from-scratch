from __future__ import annotations


class LocalProvider:
    """No-key, local provider. Will wrap Ollama / a local HF model.

    Ships as a stub so the core install stays dependency-free; implemented with
    the levels that need it (embeddings in L2, generation in L3/L4).
    """

    name = "local"

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        raise NotImplementedError("Local embeddings arrive with Level 2.")

    def generate(self, prompt: str, *, model: str, **decode: object) -> str:
        raise NotImplementedError("Local generation arrives with Level 3/4.")

    def tokenize(self, text: str, *, model: str) -> list[int]:
        raise NotImplementedError("Use Level 1's BPE tokenizer for tokenization.")

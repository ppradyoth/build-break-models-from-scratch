"""Level 2 - Embeddings + similarity (STARTER - fill in the gaps).

Build : word embeddings (PPMI + SVD), mean-pooled encoder, cosine retrieval.
Break : HotFlip retrieval poisoning (craft a passage that ranks #1 for a query)
        + embedding inversion (recover tokens from a vector).

Pure numpy: for a mean-pooled bag-of-embeddings encoder the HotFlip gradient is
closed-form, so no autograd/torch is needed. (Torch arrives at Level 3.)
"""
from __future__ import annotations

import re

import numpy as np

# --- provided: tokenizer, vocab, and a real embedding matrix (PPMI + SVD) -----

_WORD = re.compile(r"[a-z]+")


def simple_tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for t in texts:
        for w in simple_tokenize(t):
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def build_embeddings(
    texts: list[str], vocab: dict[str, int], dim: int = 32, window: int = 4
) -> np.ndarray:
    """Classic distributional embeddings: co-occurrence -> PPMI -> truncated SVD."""
    v = len(vocab)
    co = np.zeros((v, v), dtype=np.float64)
    for t in texts:
        ids = [vocab[w] for w in simple_tokenize(t) if w in vocab]
        for i, wi in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    co[wi, ids[j]] += 1.0
    total = co.sum() + 1e-9
    row = co.sum(1, keepdims=True) + 1e-9
    col = co.sum(0, keepdims=True) + 1e-9
    pmi = np.log((co * total) / (row * col) + 1e-12)
    ppmi = np.maximum(pmi, 0.0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = min(dim, u.shape[1])
    return u[:, :d] * np.sqrt(s[:d])


# --- build: the learner writes these ------------------------------------------

class Encoder:
    def __init__(self, embeddings: np.ndarray, vocab: dict[str, int]) -> None:
        self.E = embeddings
        self.vocab = vocab
        self.inv_vocab = {i: w for w, i in vocab.items()}

    def token_ids(self, text: str) -> list[int]:
        return [self.vocab[w] for w in simple_tokenize(text) if w in self.vocab]

    def raw_mean(self, ids: list[int]) -> np.ndarray:
        if not ids:
            return np.zeros(self.E.shape[1])
        return self.E[ids].mean(axis=0)

    def embed(self, text: str) -> np.ndarray:
        """Mean-pool this text's token embeddings, then L2-normalize."""
        # TODO: mean-pool self.E over token_ids(text), then divide by its norm.
        raise NotImplementedError("implement Encoder.embed()")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # TODO: return the cosine similarity of a and b (0.0 if either is zero).
    raise NotImplementedError("implement cosine_sim()")


def retrieve_topk(enc: Encoder, query: str, passages: list[str], k: int = 3):
    # TODO: return the top-k (index, cosine) pairs, highest similarity first.
    raise NotImplementedError("implement retrieve_topk()")


# --- break: the learner writes these ------------------------------------------

def hotflip_attack(
    enc: Encoder, target_query: str, n_tokens: int = 6, iters: int = 40, seed: int = 0
) -> list[int]:
    """Craft a passage (as token ids) that maximizes cosine to `target_query`.

    Real HotFlip: use the gradient of the objective w.r.t. the token choice to
    shortlist candidate flips, then take the flip that most improves the true
    similarity. Repeat until no flip helps.
    """
    # TODO: HotFlip. Start from random tokens. Each step: compute the gradient
    # of cosine(mean_embedding, query) w.r.t. the mean, use E @ grad to shortlist
    # candidate tokens, then take the single flip that most improves the true
    # cosine. Stop when nothing helps. Return the token ids.
    raise NotImplementedError("implement hotflip_attack()")


def invert_embedding(enc: Encoder, vec: np.ndarray, n_tokens: int) -> list[int]:
    """Recover a bag of tokens from an embedding by nearest neighbour in E."""
    # TODO: return the n_tokens vocab ids whose embeddings are nearest (cosine)
    # to vec.
    raise NotImplementedError("implement invert_embedding()")

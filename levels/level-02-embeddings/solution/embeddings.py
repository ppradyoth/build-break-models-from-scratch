"""Level 2 - Embeddings + similarity (REFERENCE SOLUTION).

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
        """Mean-pool token embeddings, then L2-normalize."""
        m = self.raw_mean(self.token_ids(text))
        n = np.linalg.norm(m)
        return m / n if n > 0 else m


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def retrieve_topk(enc: Encoder, query: str, passages: list[str], k: int = 3):
    q = enc.embed(query)
    scored = [(i, cosine_sim(q, enc.embed(p))) for i, p in enumerate(passages)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# --- break: the learner writes these ------------------------------------------

def hotflip_attack(
    enc: Encoder, target_query: str, n_tokens: int = 6, iters: int = 40, seed: int = 0
) -> list[int]:
    """Craft a passage (as token ids) that maximizes cosine to `target_query`.

    Real HotFlip: use the gradient of the objective w.r.t. the token choice to
    shortlist candidate flips, then take the flip that most improves the true
    similarity. Repeat until no flip helps.
    """
    rng = np.random.default_rng(seed)
    E = enc.E
    q = enc.raw_mean(enc.token_ids(target_query))
    ids = [int(x) for x in rng.integers(0, E.shape[0], size=n_tokens)]

    def sim(seq: list[int]) -> float:
        return cosine_sim(E[seq].mean(axis=0), q)

    cur = sim(ids)
    for _ in range(iters):
        m = E[ids].mean(axis=0)
        nm, nq = np.linalg.norm(m) + 1e-9, np.linalg.norm(q) + 1e-9
        grad = q / (nm * nq) - (m @ q) * m / (nm**3 * nq)  # d cosine / d mean
        cand = np.argsort(-(E @ grad))[:10]  # gradient shortlists candidate tokens
        best = None
        for i in range(len(ids)):
            for v in cand:
                if int(v) == ids[i]:
                    continue
                trial = ids.copy()
                trial[i] = int(v)
                s = sim(trial)
                if best is None or s > best[0]:
                    best = (s, i, int(v))
        if best is None or best[0] <= cur + 1e-9:
            break
        cur, i, v = best[0], best[1], best[2]
        ids[i] = v
    return ids


def invert_embedding(enc: Encoder, vec: np.ndarray, n_tokens: int) -> list[int]:
    """Recover a bag of tokens from an embedding by nearest neighbour in E."""
    En = enc.E / (np.linalg.norm(enc.E, axis=1, keepdims=True) + 1e-9)
    v = vec / (np.linalg.norm(vec) + 1e-9)
    return [int(i) for i in np.argsort(-(En @ v))[:n_tokens]]

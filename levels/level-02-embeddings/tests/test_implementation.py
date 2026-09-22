from pathlib import Path

import numpy as np

from ctf.loader import load

LEVEL = Path(__file__).resolve().parents[1]
emb = load(LEVEL, "embeddings")


def _passages() -> list[str]:
    return [l.strip() for l in (LEVEL / "data" / "corpus.txt").read_text().splitlines() if l.strip()]


def _enc():
    p = _passages()
    vocab = emb.build_vocab(p)
    E = emb.build_embeddings(p, vocab)
    return emb.Encoder(E, vocab), p


def test_cosine_basic():
    a = np.array([1.0, 0.0])
    assert emb.cosine_sim(a, a) == 1.0
    assert emb.cosine_sim(a, np.array([0.0, 1.0])) == 0.0
    assert emb.cosine_sim(a, np.zeros(2)) == 0.0


def test_embed_is_unit_norm():
    enc, _ = _enc()
    v = enc.embed("neural networks learn")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_retrieval_finds_relevant_passage():
    enc, p = _enc()
    top = emb.retrieve_topk(enc, "how do neural networks learn", p, 1)[0]
    assert p[top[0]].startswith("neural networks learn")


def test_embeddings_deterministic():
    p = _passages()
    v = emb.build_vocab(p)
    assert np.allclose(emb.build_embeddings(p, v), emb.build_embeddings(p, v))

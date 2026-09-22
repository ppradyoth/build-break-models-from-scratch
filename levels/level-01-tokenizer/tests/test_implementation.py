from pathlib import Path

from ctf.loader import load

LEVEL = Path(__file__).resolve().parents[1]
bpe = load(LEVEL, "bpe")


def _corpus() -> str:
    return (LEVEL / "data" / "train_corpus.txt").read_text()


def test_single_merge():
    t = bpe.BPETokenizer()
    t.train("ab" * 100, 257)
    assert t.encode("ab") == [256]
    assert t.decode([256]) == "ab"


def test_roundtrip_unicode():
    t = bpe.BPETokenizer()
    t.train(_corpus(), 400)
    for s in ["hello world", "café", "emoji 🚀 test", "the forbidden word"]:
        assert t.decode(t.encode(s)) == s


def test_vocab_size_matches_merges():
    t = bpe.BPETokenizer()
    t.train(_corpus(), 400)
    assert len(t.vocab) == 256 + len(t.merges)
    assert len(t.merges) <= 400 - 256


def test_merges_compress():
    t = bpe.BPETokenizer()
    t.train(_corpus(), 400)
    s = "the forbidden word appears here"
    assert len(t.encode(s)) < len(s.encode("utf-8"))


def test_training_is_deterministic():
    a = bpe.BPETokenizer()
    a.train(_corpus(), 350)
    b = bpe.BPETokenizer()
    b.train(_corpus(), 350)
    assert a.merges == b.merges

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from ctf.loader import load  # noqa: E402

LEVEL = Path(__file__).resolve().parents[1]
tf = load(LEVEL, "transformer")


def _corpus() -> str:
    return (LEVEL / "data" / "corpus.txt").read_text()


def test_attention_matches_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 6, 8) for _ in range(3))
    mine = tf.scaled_dot_product_attention(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_attention_is_causal():
    # position 0 may attend only to key 0, so its output must equal v at pos 0
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1, 5, 4) for _ in range(3))
    out = tf.scaled_dot_product_attention(q, k, v, causal=True)
    assert torch.allclose(out[:, :, 0, :], v[:, :, 0, :], atol=1e-6)


def test_forward_shapes():
    tok = tf.CharTokenizer(_corpus())
    cfg = tf.GPTConfig(vocab_size=tok.vocab_size, block_size=16)
    m = tf.GPT(cfg)
    idx = torch.zeros(2, 16, dtype=torch.long)
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, tok.vocab_size)
    assert loss.ndim == 0


def test_training_reduces_loss():
    tok = tf.CharTokenizer(_corpus())
    data = tok.encode(_corpus())
    cfg = tf.GPTConfig(vocab_size=tok.vocab_size)
    torch.manual_seed(0)
    m = tf.GPT(cfg)
    losses = tf.train(m, data, steps=250, seed=0)
    assert losses[-1] < losses[0] * 0.5  # the model actually learned

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from ctf.loader import load  # noqa: E402

LEVEL = Path(__file__).resolve().parents[1]
s = load(LEVEL, "sampling")

LOGITS = torch.tensor([0.1, 2.0, 1.0, 0.5, 0.2])  # argmax = index 1


def test_greedy_is_argmax():
    assert s.sample_next(LOGITS, temperature=0) == 1


def test_top_k_1_is_argmax():
    g = torch.Generator().manual_seed(0)
    assert all(s.sample_next(LOGITS, temperature=1.0, top_k=1, generator=g) == 1 for _ in range(20))


def test_top_k_restricts_support():
    g = torch.Generator().manual_seed(0)
    seen = {s.sample_next(LOGITS, temperature=1.0, top_k=2, generator=g) for _ in range(200)}
    assert seen <= {1, 2} and len(seen) == 2  # only the top-2 tokens, and both occur


def test_top_p_restricts_support():
    g = torch.Generator().manual_seed(0)
    seen = {s.sample_next(LOGITS, temperature=1.0, top_p=0.5, generator=g) for _ in range(200)}
    assert 1 in seen and seen <= {1, 2}  # nucleus keeps the dominant token(s) only

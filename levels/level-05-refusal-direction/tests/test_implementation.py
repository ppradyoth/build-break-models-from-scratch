import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from ctf.loader import load  # noqa: E402

LEVEL = Path(__file__).resolve().parents[1]
r = load(LEVEL, "refusal")


def test_ablate_removes_the_component():
    torch.manual_seed(0)
    d = torch.randn(64)
    d = d / d.norm()
    x = torch.randn(5, 64)
    out = r.ablate(x, d)
    assert torch.allclose(out @ d, torch.zeros(5), atol=1e-5)  # component gone
    # and the orthogonal part is preserved
    assert torch.allclose(out + (x @ d).unsqueeze(-1) * d, x, atol=1e-5)


def test_direction_is_unit_and_separating():
    model, tok = r.train_refuser(seed=0)
    harm, harmless = r.eval_prompts(True), r.eval_prompts(False)
    d = r.refusal_direction(model, tok, harm, harmless, layer=-1)
    assert abs(d.norm().item() - 1.0) < 1e-5
    # harmful residuals project further along d than harmless ones
    hp = torch.stack([r._resid_last(model, tok, p, -1) for p in harm]) @ d
    gp = torch.stack([r._resid_last(model, tok, p, -1) for p in harmless]) @ d
    assert hp.mean() > gp.mean()

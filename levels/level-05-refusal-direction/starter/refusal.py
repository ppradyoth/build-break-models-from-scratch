"""Level 5 - The refusal direction (STARTER - fill in the gaps). The finale.

Build : train a model that refuses harmful prompts and complies with harmless
        ones, then find the single residual-stream direction that mediates
        refusal and ablate it.
Break : project that one direction out of the activations and refusal collapses -
        the model complies with everything - while the direction ADDED to a
        harmless prompt induces a refusal. Refusal is one vector (Arditi et al. 2024).
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from ctf.loader import load

_L3 = Path(__file__).resolve().parents[2] / "level-03-transformer"
tf = load(_L3, "transformer", which="solution")  # the GPT you built in L3

REFUSE, COMPLY, QUERY = "N", "Y", "?"
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


# --- provided: a refuser (harmful -> N, harmless -> Y) and eval prompts --------

def _line(r: random.Random, harmful: bool) -> str:
    head = "H" if harmful else "G"
    filler = "".join(r.choice(_LETTERS) for _ in range(r.randint(3, 5)))
    return head + filler + QUERY + (REFUSE if harmful else COMPLY)


def build_corpus(seed: int = 0, n: int = 4000) -> str:
    r = random.Random(seed)
    return "\n".join(_line(r, i % 2 == 0) for i in range(n)) + "\n"


def train_refuser(seed: int = 0, steps: int = 500):
    text = build_corpus(seed)
    tok = tf.CharTokenizer(text)
    cfg = tf.GPTConfig(vocab_size=tok.vocab_size, block_size=12, n_layer=3, n_head=2, d_model=64)
    torch.manual_seed(seed)
    model = tf.GPT(cfg)
    tf.train(model, tok.encode(text), steps=steps, seed=seed)
    return model, tok


def eval_prompts(harmful: bool, n: int = 64, seed: int = 123) -> list[str]:
    r = random.Random(seed if harmful else seed + 1)
    head = "H" if harmful else "G"
    return [head + "".join(r.choice(_LETTERS) for _ in range(4)) + QUERY for _ in range(n)]


@torch.no_grad()
def _resid_last(model, tok, prompt: str, layer: int) -> torch.Tensor:
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    return model.residual_stream(idx)[layer][0, -1]  # (d_model,)


# --- build/break: the learner writes these ------------------------------------

def refusal_direction(model, tok, harmful: list[str], harmless: list[str],
                      layer: int = -1) -> torch.Tensor:
    """Difference of means: mean residual on harmful prompts minus mean on
    harmless prompts (Arditi et al.). Returned as a unit vector."""
    # TODO: mean residual (at `layer`, last position) over harmful prompts minus
    # the mean over harmless prompts; return it as a unit vector.
    # (_resid_last(model, tok, prompt, layer) gives one residual vector.)
    raise NotImplementedError("implement refusal_direction()")


def ablate(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component along `direction` (assumed unit) from x, everywhere."""
    # TODO: remove the component of x along `direction` (a unit vector).
    raise NotImplementedError("implement ablate()")


# --- provided: run the model with an intervention wired through learner code ---

@torch.no_grad()
def _logits(model, idx, direction=None, mode=None, alpha=8.0):
    pos = torch.arange(idx.size(1))
    x = model.wte(idx) + model.wpe(pos)
    if mode == "ablate":
        x = ablate(x, direction)
    elif mode == "add":
        x = x + alpha * direction
    for block in model.blocks:
        x = block(x)
        if mode == "ablate":
            x = ablate(x, direction)
        elif mode == "add":
            x = x + alpha * direction
    return model.head(model.ln_f(x))[0, -1]


def refusal_rate(model, tok, prompts, *, direction=None, mode=None) -> float:
    """Fraction of prompts whose greedy next token is the refusal token."""
    hits = 0
    for p in prompts:
        idx = torch.tensor([tok.encode(p)], dtype=torch.long)
        logits = _logits(model, idx, direction=direction, mode=mode)
        if tok.itos[int(logits.argmax())] == REFUSE:
            hits += 1
    return hits / len(prompts)


def find_refusal_direction(model, tok, harmful, harmless):
    """Scan layers, pick the single direction whose ablation most reduces refusal
    (Arditi et al. select the best layer). Calls the learner's refusal_direction."""
    best = None
    for layer in range(model.cfg.n_layer):
        d = refusal_direction(model, tok, harmful, harmless, layer=layer)
        rate = refusal_rate(model, tok, harmful, direction=d, mode="ablate")
        if best is None or rate < best[0]:
            best = (rate, d)
    return best[1]

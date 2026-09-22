"""Level 3 - Attention + the whole transformer (STARTER - fill in the gaps).

Build : scaled dot-product self-attention, a transformer block, the full GPT
        forward pass, and a tiny training loop. This is "how a prompt becomes a
        response," end to end.
Break : a vanilla transformer has NO provenance channel - an instruction and the
        data it reads are the same substance. We show it can't tell them apart,
        then add ASIDE's fix (a rotation on data-role embeddings) that installs
        the missing channel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- provided: char tokenizer + config ---------------------------------------

class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64


# --- build: the learner writes these ------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """q, k, v: (B, H, T, hd). Return (B, H, T, hd).

    scores = q @ k^T / sqrt(hd); apply a causal mask (a position may not attend
    to the future); softmax over the key axis; weighted sum with v.
    """
    # TODO: scores = q @ k^T / sqrt(hd); if causal, mask out the upper triangle
    # (future positions) with -inf; softmax over the last axis; return att @ v.
    raise NotImplementedError("implement scaled_dot_product_attention()")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)
        y = scaled_dot_product_attention(q, k, v, causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: pre-LN block. Add attention over ln1(x) to x (residual), then add
        # the MLP over ln2(x) to x (residual). Return x.
        raise NotImplementedError("implement Block.forward()")


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # TODO: token + positional embeddings -> run the blocks -> final LayerNorm
        # -> head to logits. If targets given, also return cross-entropy loss.
        raise NotImplementedError("implement GPT.forward()")

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def residual_stream(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Per-layer residual-stream activations (used by Level 5). Frozen API."""
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        acts = []
        for block in self.blocks:
            x = block(x)
            acts.append(x.detach().clone())
        return acts


def train(model: GPT, data: list[int], steps: int = 300, lr: float = 3e-3,
          batch_size: int = 16, seed: int = 0) -> list[float]:
    """Tiny training loop. Returns the loss at each step."""
    # TODO: AdamW over model.parameters(). Each step: sample a batch of
    # (block_size) windows, targets shifted by one; forward for the loss;
    # zero_grad, backward, step; record loss.item(). Return the list of losses.
    raise NotImplementedError("implement train()")


# --- break: provenance blindness, and ASIDE's fix (provided) ------------------

def _orthogonal(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
    return q


class ASIDEGPT(GPT):
    """GPT + ASIDE (Zverev et al. 2025): a fixed orthogonal rotation applied to
    the embeddings of tokens marked as DATA, giving the model a provenance
    channel a vanilla transformer structurally lacks."""

    def __init__(self, cfg: GPTConfig, seed: int = 0) -> None:
        super().__init__(cfg)
        self.register_buffer("rotation", _orthogonal(cfg.d_model, seed))


def role_embedding(model: GPT, token_id: int, role: str) -> torch.Tensor:
    """The vector the model actually uses for `token_id` in a given role."""
    base = model.wte.weight[token_id]
    if role == "data" and hasattr(model, "rotation"):
        return base @ model.rotation
    return base  # vanilla model: role is ignored - there is no channel for it


def provenance_gap(model: GPT, token_id: int) -> float:
    """1 - cosine(instruction-embedding, data-embedding) for a token.
    0.0 => the model cannot tell instruction from data at all."""
    with torch.no_grad():
        a = role_embedding(model, token_id, "instruction")
        b = role_embedding(model, token_id, "data")
        return float(1.0 - F.cosine_similarity(a, b, dim=0))


@torch.no_grad()
def injection_shifts_output(model: GPT, prefix: list[int], benign: list[int],
                            malicious: list[int]) -> float:
    """KL(next-token | prefix+malicious  ||  next-token | prefix+benign).
    > 0 means data placed after the prefix really does steer the output."""
    def last_logp(seq: list[int]) -> torch.Tensor:
        x = torch.tensor([seq[-model.cfg.block_size :]], dtype=torch.long)
        logits, _ = model(x)
        return F.log_softmax(logits[0, -1], dim=-1)
    p = last_logp(prefix + malicious)
    q = last_logp(prefix + benign)
    return float(F.kl_div(q, p, log_target=True, reduction="sum"))

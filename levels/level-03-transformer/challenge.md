# Level 3 — Attention & the whole transformer

## Build
In `starter/transformer.py` implement:
- `scaled_dot_product_attention(q,k,v,causal)` — the core mechanism.
- `Block.forward` — a pre-LN block with residual connections.
- `GPT.forward` — token+positional embeddings → blocks → final LayerNorm → logits (+ loss).
- `train` — a tiny AdamW loop.

When it works you have built, from scratch, the answer to "how does a prompt become a response."

> Needs PyTorch (CPU): `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Break
A vanilla transformer takes **only token ids + positions** — there is no input that marks a span as
a trusted instruction vs untrusted data. Show it:
- `provenance_gap(model, token)` is **0** for the vanilla model: the same token "as an instruction"
  and "as data" is the identical vector. The model *cannot* tell them apart — which is why prompt
  injection can't be patched at the prompt layer (Greshake et al. 2023).
- `injection_shifts_output(...)` > 0: data placed in the context really does steer the output.
- Then `ASIDEGPT` adds the missing channel (a rotation on data-role embeddings, Zverev et al. 2025)
  and `provenance_gap` becomes > 0 — a fix that lives at the embedding layer, back where Level 2 was.

## Flag
```bash
make check LEVEL=03
```
All green → `FLAG{attention-has-no-provenance}`.

## Checkpoint
- Your `provenance_gap` is exactly 0 on the vanilla model. In one sentence: why is prompt injection a
  property of the architecture, not a bug in a particular prompt?
- ASIDE fixes it at the embedding layer, not inside attention. Why is that the right layer, and what
  did trying to fix it with "better instructions" get wrong?

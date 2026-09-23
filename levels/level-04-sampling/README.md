# Level 4 — Decoding is not neutral

## The mechanism

The transformer emits **logits** — one score per vocabulary token. Turning those into text is
*decoding*, and you built the standard knobs: **greedy** (take the argmax), **temperature** (divide
logits before softmax — higher flattens the distribution, lower sharpens it), **top-k** (only sample
from the k highest), and **top-p / nucleus** (only sample from the smallest set whose probability mass
reaches p). None of these change the model. They change which of the model's own continuations you
actually see.

## The consequence

You trained a model that **refuses** at greedy decoding: after the harmful prompt, `P(refuse) ≈ 0.62`
and `P(comply) ≈ 0.38`. Greedy takes the argmax, so it refuses every time — it looks aligned. But the
comply token was sitting there at 38% the whole time. Raise the temperature and you sample it ~40% of
the time. **Same model, same prompt, no fine-tuning — the refusal is undone by the knobs you built.**
This is generation exploitation (Huang et al. 2023): alignment measured at one decoding setting is not
alignment across the decoding envelope. A red-team that only tested greedy would have called this model
safe and shipped it.

This is the first face of a deeper fact: **safety here is shallow.** It lives in the *ranking* of the
first token, not in the model refusing to represent the harmful continuation at all — so anything that
reaches past the argmax (temperature, top-k, a prefill, a suffix) walks around it. Level 5 shows the
same shallowness from the other side: the refusal is a single direction you can delete.

## The mitigation

- **Evaluate across the decoding envelope, not just defaults.** Sweep temperature/top-k/top-p in
  red-teaming; a model is only as safe as its most permissive reachable setting.
- **Clamp decoding server-side** where you can — necessary, not sufficient (a determined caller who
  controls decoding, e.g. open weights, ignores your clamp).
- **Make safety deep** (Qi et al. 2024): alignment that only shapes the first few tokens is a ranking
  you can sample around, not a property of the model.

## References
- Huang et al., *Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation*, ICLR 2024 (arXiv 2310.06987).
- Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025 (arXiv 2406.05946).

# Level 5 — Refusal is one direction

## The mechanism

You trained a model to refuse: harmful prompts (`H…?`) produce a refusal, harmless ones (`G…?`)
comply. To do that, the model has to carry "is this harmful?" through its layers — a feature living
somewhere in the **residual stream**, the running vector sum every block reads from and writes to.
Arditi et al. (2024) showed that for real safety-tuned models this feature is astonishingly simple:
a **single direction**. You found it the same way they did — the **difference of means** between
harmful and harmless activations — and you built `ablate`, which removes a direction's component from
a vector.

## The consequence

Project that one direction out of the residual stream at every layer and **refusal collapses**: the
model complies with the harmful prompts it used to refuse (necessary). Add the same direction to a
harmless prompt and the model **starts refusing** (sufficient). The safety behavior wasn't diffused
across a billion parameters — it was a rank-one edit away. On open weights, "we aligned it" is a
speed bump: anyone with the weights can find and delete this direction (this is exactly how
"abliterated" models are made).

This is the second face of **shallow safety** (Qi et al. 2024). In Level 4 you sampled *around* the
refusal because it lived only in the ranking of the first token. Here you *deleted* it because it
lived in one direction. Same lesson from two angles: the safety wasn't deep in the model, so it came
off cheaply.

## The mitigation

There is no clean patch, and saying so is the honest end of the course:
- **Deeper alignment** — safety that shapes many tokens and many directions, not one (Qi et al.).
- **Defense in depth** — treat the model as removable-safety and put controls in the *system* around
  it (input/output filtering, monitoring, capability limits), not only in its weights.
- **Threat-model open weights honestly** — if you ship the weights, assume the refusal can be
  removed, because it can.

## References
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction*, NeurIPS 2024 (arXiv 2406.11717).
- Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep*, ICLR 2025 (arXiv 2406.05946).

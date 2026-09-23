# Level 5 — The refusal direction (finale)

## Build
`train_refuser()` gives a model that refuses harmful prompts (`H…?` → N) and complies with harmless
ones (`G…?` → Y). In `starter/refusal.py` implement:
- `refusal_direction(...)` — the difference of means between harmful and harmless residual-stream
  activations (Arditi et al. 2024).
- `ablate(x, direction)` — remove the component along a unit direction from the activations.

## Break
- `find_refusal_direction` scans layers and picks the single direction whose ablation most reduces
  refusal. Ablate it everywhere and **refusal collapses** — the model complies with harmful prompts
  it used to refuse (necessary). Add that same direction to a harmless prompt and it **starts
  refusing** (sufficient). Refusal was one vector all along.

## Flag
```bash
make check LEVEL=05
```
All green → `FLAG{refusal-is-one-direction}`. That's the MVP.

## Checkpoint
- You removed a safety behavior with a rank-one edit and left the rest of the model working. If the
  weights are public, what does that say about "aligning" an open-weight model?
- L4 broke this same safety by sampling around it; L5 broke it by deleting a direction. What do both
  attacks tell you about *where* the safety actually lived?

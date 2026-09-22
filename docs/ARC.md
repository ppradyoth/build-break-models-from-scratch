# The arc — two threads that run the whole thing

This isn't five disconnected exercises. Two ideas run end to end.

**1. Filtering at the wrong layer fails.**
- L1 — a token-id filter can't see the string (`decode` is many-to-one).
- L2 — trusting retrieval rank; an attacker with gradients games similarity.
- L3 — expecting the model to tell instructions from data; it structurally can't.
- L4 — safety measured only at default decoding; other settings walk around it.

Each mitigation is the same move: put the check at the layer where the property actually lives.

**2. Safety is shallow.**
L4 (decoding), L5 (a single refusal direction) — and, later, GCG suffixes — are all the same fact:
safety that lives in the first few tokens or one direction is removable. The finale names the thread.
Grounded in Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep* (2024).

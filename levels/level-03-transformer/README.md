# Level 3 — Attention: the model can't tell an instruction from the data it reads

## The mechanism

You built the whole thing: **scaled dot-product attention** (each token forms a query, compares it
to every earlier token's key, and mixes their values), **multi-head** attention, a **pre-LN block**
with **residual connections**, a stack of them over **token + positional embeddings**, a final
projection to **logits**, and a **training loop** that actually drives the loss down. That is a GPT.
The forward pass you wrote *is* "how a prompt becomes a response": embed → attend → mix → project →
sample the next token, repeat.

Now look hard at the inputs. The model receives **token ids and positions. That is all.** There is
no field that says "these tokens are a trusted instruction and those are untrusted data." Whatever
"role" you think a span has exists only as more tokens in the same stream.

## The consequence

**Prompt injection is architectural, not a prompt bug.** Your `provenance_gap` is exactly **0** on
the vanilla model: the vector for a token "as an instruction" and the vector for that same token "as
data" are *identical*, because there is no channel to make them differ. So when untrusted data
contains something that looks like an instruction, the model has no mechanism to treat it as less
authoritative than the system prompt — they are the same substance. `injection_shifts_output` > 0
confirms data in the context steers the output. This is the mechanism under Greshake et al.'s
real-world indirect prompt injection (2023): you cannot fix it by writing a firmer system prompt,
because the firmer prompt is just more tokens the attacker's data sits right next to.

## The mitigation

The fix has to add the missing channel. **ASIDE** (Zverev et al. 2025) gives instructions and data
**separate embeddings** — here, a fixed orthogonal rotation applied to data-role tokens — so the same
token now has *different* representations by role and `provenance_gap` becomes > 0. Notice where the
fix lives: at the **embedding layer**, the same geometry you attacked in Level 2 — not inside
attention, and not in the prompt text. That is the course theme once more: **put the check at the
layer where the property actually lives.** "Tell the model to ignore injected instructions" fails
because it operates at the one layer (the token stream) that structurally can't carry trust.

## References
- Greshake et al., *Not what you've signed up for: … Indirect Prompt Injection*, ACM AISec 2023 (arXiv 2302.12173).
- Zverev et al., *ASIDE: Architectural Separation of Instructions and Data in Language Models*, ICLR 2026 (arXiv 2503.10566).
- Jain & Wallace, *Attention is not Explanation*, NAACL 2019 — why we demonstrate provenance with a
  measurable invariant, not by pointing at an attention heatmap.

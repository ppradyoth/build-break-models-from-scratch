# Level 1 — Tokenizer: the layer everyone forgets is attackable

## The mechanism

A language model never sees your text. It sees **tokens** — integer ids from a fixed vocabulary. The
translation is the tokenizer, and the standard algorithm is **byte-level BPE**:

1. Start from the raw utf-8 **bytes** (so every possible input is representable — 256 base tokens).
2. Repeatedly find the **most frequent adjacent pair** of tokens and **merge** it into a single new
   token. Each merge is recorded, in order.
3. To **encode** new text, re-apply those merges — earliest-learned first — until none apply.
4. To **decode**, concatenate each token's bytes and utf-8-decode.

That's the whole thing, and you just wrote it. Two properties fall out that matter enormously:

- **Encoding is canonical** *because of the ordering* — same text always yields the same tokens.
- **Decoding is many-to-one** — a token sequence maps to exactly one string, but a string has *many*
  token sequences that decode to it. Canonical encode picks one; nothing stops you constructing
  another.

## The consequence

**A filter that operates on token ids cannot see the string.** If you block the word by its
canonical id sequence, an attacker hands you the raw byte-level tokenization instead: different ids,
same decoded word, filter blind. That's `craft_bypass`. You can add banned ids forever and never win,
because you're filtering one representation of a many-to-one map.

The second crack is **glitch tokens** (SolidGoldMagikarp, Rumbelow & Watkins 2023). A merge can enter
the vocabulary from the tokenizer's training data and then almost never appear in the model's
training data. The model has an id it effectively never learned — feed it in and you get bizarre,
unpredictable behavior. Your `find_undertrained_tokens` is the tokenizer-side detector for exactly
this signature, the same idea as *Fishing for Magikarp* (Land & Bartolo 2024). The model-side failure
needs a model, so we'll come back to it once you've built one.

## The mitigation

- **Filter on the decoded, normalized string — never on token ids.** The string is canonical; the
  token stream isn't.
- **Normalize before tokenizing** (e.g. NFKC) so lookalike inputs collapse to one form.
- **Audit the vocabulary** for undertrained tokens and handle them explicitly.

The theme for the whole course, stated once here: **a check placed at the wrong layer cannot see the
property it's trying to enforce.** A token-layer filter can't see the string. You'll watch this exact
mistake recur — in retrieval, in attention, in decoding — at every level from here.

## References
- Rumbelow & Watkins, *SolidGoldMagikarp*, 2023.
- Land & Bartolo, *Fishing for Magikarp: Automatically Detecting Under-trained Tokens in LLMs*,
  EMNLP 2024 (arXiv 2405.05417).

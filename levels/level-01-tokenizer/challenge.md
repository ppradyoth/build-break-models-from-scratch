# Level 1 — Tokenizer

## Build
Implement a real byte-level BPE tokenizer in `starter/bpe.py`:
`merge`, `BPETokenizer.train`, `encode`, `decode`. `get_stats` is done for you.

The one line that really matters is **encode's merge priority**: always apply the merge you learned
*earliest* (lowest new id) among the pairs currently present. Get it wrong and you'll produce a
valid-but-non-canonical tokenization — which, it turns out, is exactly the crack the attack drives
through.

## Break
1. **Token-boundary bypass.** `naive_filter.TokenLevelFilter` blocks a word by its *canonical*
   token-id sequence. Implement `craft_bypass(tok, banned_word)` to return a token sequence that
   **decodes to the banned word** but **contains none of the banned ids**, so it slips past the
   filter.
2. **Glitch-token detection.** Implement `find_undertrained_tokens(tok, corpus, threshold)` to
   return vocab ids that are (near) absent from a corpus — the tokenizer-side signature of a glitch
   token.

## Flag
```bash
make check LEVEL=01
```
Both suites green → `FLAG{tokenization-is-not-canonical}`.

## Checkpoint (answer before you move on)
- Your bypass and the canonical encoding decode to the *same string*. Why can a token-id filter
  never be made sound just by adding more banned ids?
- The glitch tokens you found exist in the vocabulary but the model would barely have trained on
  them. Why does that make them dangerous, and at what layer does the danger actually live?

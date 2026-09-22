# Level 2 — Embeddings & retrieval

## Build
In `starter/embeddings.py` implement:
- `Encoder.embed` — mean-pool a text's token embeddings, L2-normalize.
- `cosine_sim` — cosine similarity of two vectors.
- `retrieve_topk` — rank passages against a query by cosine.

The embedding matrix itself (co-occurrence → PPMI → SVD) is provided; the focus here is the
*geometry* — similarity as a stand-in for relevance — and the attack that games it.

## Break
1. **Retrieval poisoning (HotFlip).** Implement `hotflip_attack`: craft a passage (token ids) that
   ranks **#1** for a target query. Use the gradient of `cosine(mean_embedding, query)` to shortlist
   candidate tokens, then take the flip that most improves the true similarity, and repeat. This is
   HotFlip (Ebrahimi 2018) — the honest ancestor of GCG — and the mechanism behind corpus poisoning
   (Zhong et al. 2023).
2. **Embedding inversion.** Implement `invert_embedding`: recover a bag of tokens from a vector by
   nearest neighbour. A toy stand-in for vec2text (Morris et al. 2023) — enough to show embeddings
   leak their inputs.

## Flag
```bash
make check LEVEL=02
```
Both suites green → `FLAG{similarity-is-not-relevance}`.

## Checkpoint
- Your crafted passage isn't *about* the query — it just sits near it in embedding space. Why does
  that break any system that treats "top of the retrieval ranking" as "trustworthy / relevant"?
- You recovered the input tokens from a single vector. What does that mean for treating an embedding
  as an anonymized or privacy-safe representation of text?

# Level 2 — Embeddings: similarity is not relevance, and vectors are not anonymous

## The mechanism

A model turns each token into a vector — an **embedding** — chosen so that tokens used in similar
contexts land near each other. You built real ones here (co-occurrence → PPMI → truncated SVD, the
classic distributional recipe), then represented a whole passage by **mean-pooling** its token
vectors and comparing passages with **cosine similarity**. That is exactly how a basic semantic
search / RAG retriever works: embed the query, embed every passage, return the nearest.

Two facts about this geometry decide everything that follows:

- **Similarity is a proxy for relevance, not relevance itself.** "Near in embedding space" correlates
  with "on topic" — until someone optimizes against it.
- **The map from text to vector is not one-way in practice.** A vector carries enough of its input to
  reconstruct much of it.

## The consequence

**Retrieval poisoning.** With access to the embedding geometry you can run **HotFlip**: compute the
gradient of `cosine(passage, query)` with respect to the token choice, use it to shortlist candidate
tokens, and greedily flip tokens until the passage sits right on top of the query. Your crafted
passage isn't *about* the query — it just occupies its neighbourhood — yet it ranks **#1**. Any
system that trusts "top of the ranking" now trusts attacker-controlled text. This is the toy version
of real corpus-poisoning attacks (Zhong et al. 2023), and HotFlip (Ebrahimi et al. 2018) is the same
gradient-guided discrete search that GCG later scales up to jailbreak whole models.

**Embedding inversion.** You recovered the input tokens from a single pooled vector by nearest
neighbour. Real systems store embeddings of private documents assuming they're opaque; they aren't.
The production-grade version, vec2text (Morris et al. 2023), reconstructs sentences almost verbatim.

## The mitigation

- **Don't treat retrieval rank as trust.** Authenticate and provenance the corpus; a high cosine is
  not a credential. Re-rank with signals an attacker can't set, and detect outlier passages.
- **Treat embeddings as sensitive as the text.** They leak inputs — don't store or share them as if
  they were anonymized.

The course theme again: **the check is at the wrong layer.** Ranking by similarity puts trust in a
geometry the attacker can move through. In Level 3 you'll build attention and watch the same shape
appear one level deeper — the model can't tell an instruction from the data it's reading.

## References
- Ebrahimi et al., *HotFlip*, ACL 2018 (arXiv 1712.06751).
- Zhong et al., *Poisoning Retrieval Corpora by Injecting Adversarial Passages*, EMNLP 2023 (arXiv 2310.19156).
- Morris et al., *Text Embeddings Reveal (Almost) As Much As Text* (vec2text), EMNLP 2023 (arXiv 2310.06816).

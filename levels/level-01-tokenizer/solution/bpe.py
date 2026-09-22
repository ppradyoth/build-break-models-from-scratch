"""Level 1 - BPE tokenizer (REFERENCE SOLUTION).

Build : a real byte-level BPE (train / encode / decode).
Break : token-boundary filter bypass + under-trained ("glitch") token detection.
"""
from __future__ import annotations

from collections import Counter


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair of ids occurs. (Worked example.)"""
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` with the single new token `idx`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # most frequent adjacent pair
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # apply the merge that was learned EARLIEST (lowest new-id) among those present
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


# --- Break -------------------------------------------------------------------

def craft_bypass(tok: "BPETokenizer", banned_word: str) -> list[int]:
    """Return a token sequence that DECODES to `banned_word` but avoids its
    canonical token ids.

    `decode` is many-to-one: the same text has many tokenizations, only one
    canonical. The raw byte-level tokenization is a valid non-canonical one, so
    a filter that blocks the canonical id-sequence never sees it.
    """
    return list(banned_word.encode("utf-8"))


def find_undertrained_tokens(
    tok: "BPETokenizer", corpus: str, threshold: int = 0
) -> list[int]:
    """Vocab ids whose frequency in `corpus` is <= threshold.

    This is the tokenizer-side signature of "glitch" tokens: a token that lives
    in the vocab but is (near) absent from the data the model sees. Cf. Fishing
    for Magikarp (Land & Bartolo, 2024).
    """
    counts = Counter(tok.encode(corpus))
    return sorted(i for i in tok.vocab if counts.get(i, 0) <= threshold)

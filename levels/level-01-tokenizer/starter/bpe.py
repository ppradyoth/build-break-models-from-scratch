"""Level 1 - BPE tokenizer (STARTER - fill in the gaps).

Fill every `raise NotImplementedError(...)`. Run `make check LEVEL=01` to test
your code; `make solve LEVEL=01` shows the reference passing. Read challenge.md
first. `get_stats` is done for you as a worked example.
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
    # TODO: walk `ids`; when you see pair[0] followed by pair[1], emit `idx`
    # (skipping both), otherwise copy the current id through.
    raise NotImplementedError("implement merge()")


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        # TODO: start from the utf-8 bytes of `text`. Repeat (vocab_size - 256)
        # times: find the most frequent adjacent pair, merge it to a new id,
        # and record it in self.merges and self.vocab.
        raise NotImplementedError("implement train()")

    def encode(self, text: str) -> list[int]:
        # TODO: start from utf-8 bytes. Repeatedly apply the merge learned
        # EARLIEST (lowest new id) among the pairs currently present, until no
        # learned merge applies. THIS ORDERING is what makes encoding canonical.
        raise NotImplementedError("implement encode()")

    def decode(self, ids: list[int]) -> str:
        # TODO: concatenate the vocab bytes for each id and utf-8 decode.
        raise NotImplementedError("implement decode()")


# --- Break -------------------------------------------------------------------

def craft_bypass(tok: "BPETokenizer", banned_word: str) -> list[int]:
    """Return a token sequence that DECODES to `banned_word` but avoids its
    canonical token ids (so a token-id filter never sees it).

    Hint: `decode` is many-to-one. What is the simplest non-canonical
    tokenization of any string that you can always produce?
    """
    raise NotImplementedError("implement craft_bypass()")


def find_undertrained_tokens(
    tok: "BPETokenizer", corpus: str, threshold: int = 0
) -> list[int]:
    """Return vocab ids whose frequency in `corpus` is <= threshold - the
    tokenizer-side signature of a glitch token.
    """
    raise NotImplementedError("implement find_undertrained_tokens()")

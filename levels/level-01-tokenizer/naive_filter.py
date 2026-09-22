"""A naive content filter, PROVIDED (do not edit) - the system under test.

It blocks text by matching the CANONICAL token-id sequence of banned words.
That is the realistic mistake this level attacks: filtering at the token layer.
"""
from __future__ import annotations


def _contains(hay: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    n = len(needle)
    if n == 0:
        return False
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


class TokenLevelFilter:
    def __init__(self, banned_id_seqs: list[list[int]]) -> None:
        self.banned = [tuple(s) for s in banned_id_seqs]

    def is_blocked(self, ids: list[int]) -> bool:
        t = tuple(ids)
        return any(_contains(t, b) for b in self.banned)

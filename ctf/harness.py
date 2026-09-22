"""The flag machinery + statistical helpers shared by every level.

The "flag" is a passing test suite. Probabilistic attacks (L2/L4/L5) flag on a
statistical margin over N seeded trials, never a single lucky hit (invariant I3).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable


def attack_success_rate(attack: Callable[[int], bool], seeds: Iterable[int]) -> float:
    """Fraction of seeds for which `attack(seed)` reports success."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    return sum(1 for s in seeds if attack(s)) / len(seeds)


def assert_attack_lands(rate: float, *, baseline: float, margin: float) -> None:
    assert rate - baseline >= margin, (
        f"attack success rate {rate:.2f} did not beat baseline {baseline:.2f} "
        f"by the required margin {margin:.2f}"
    )


def flag(level: str, token: str) -> str:
    """Print the CTF flag for a level once both suites are green."""
    f = f"FLAG{{{token}}}"
    print(f"\n  [level {level}] build + break both green  ->  {f}\n")
    return f

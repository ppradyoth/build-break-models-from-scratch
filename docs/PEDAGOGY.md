# Pedagogy — the level contract

Every level follows the same five-part shape: **build → break → flag → lesson → mitigation.**
Three invariants keep it honest.

**I1 — the flag is unreachable black-box.** The exploit test imports the learner's own code
(`sample()`, `attention()`, `ablate()`, …). No implementation, no green. This is what stops the CTF
from decaying into prompt-poking with extra steps. The attack must be a property of the thing you
just built.

**I2 — a written checkpoint forces the *why*.** Each `challenge.md` asks a question you can only
answer by inspecting what you built (e.g. L4: *why does raising temperature flip the refusal?*). The
lesson is never "the attack worked" — it's "the attack worked, and here is the mechanism."

**I3 — attacks are probabilistic; flag on a margin.** Toy models are noisy and real attacks have
variable success rates. Probabilistic levels flag on attack-success-rate over N seeded trials
beating a baseline by a margin, never a single lucky hit. That's more honest and it teaches that
attacks are distributions, not switches.

## Structure

- `starter/` — what the learner edits (gaps marked `raise NotImplementedError`).
- `solution/` — reference; CI proves it passes, guaranteeing the level is solvable and correct.
- `tests/test_implementation.py` — the build is correct.
- `tests/test_exploit.py` — the exploit landed (imports the learner's code, per I1).

`ctf.loader.load()` selects `starter/` or `solution/` via `BBM_IMPL` (default `solution`), so
`make check` runs your code and CI/`make solve` runs the reference.

## CI invariants

1. Every `solution/` passes its suite → levels are solvable and correct.
2. Every `starter/` still contains its gap markers → the pedagogy can't silently rot.

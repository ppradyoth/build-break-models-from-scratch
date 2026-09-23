# Build & Break Models From Scratch

[![CI](https://github.com/ppradyoth/build-break-models-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/ppradyoth/build-break-models-from-scratch/actions/workflows/ci.yml) &nbsp;![python](https://img.shields.io/badge/python-3.11%2B-blue) &nbsp;![license](https://img.shields.io/badge/license-MIT-green)

*A speedrun through how AI models actually work — by building each piece yourself, then breaking it.*

Most people fall into one of two camps. You can attack models but can't explain what's happening
inside them, or you can implement attention from memory but have never thought about how your system
gets attacked. This repo is the bridge. Every level, you **build one component of the model stack
from scratch, then break it with a real, published attack.** The test suite going green is the flag.

I built this because I couldn't answer basic mechanism questions in a research interview — how a
prompt becomes a response, what attention is really doing — even though I break these systems for a
living. That gap is the standard shape of the AI-security practitioner, and the mirror image (ML
engineers who've never been attacked) is just as common. The fastest way I know to close it is to
build the thing and then break it, because **you understand a mechanism when you can attack it.**

The other honest reason: I can't stick to a dry curriculum, and I doubt you can either. So this is
gamified on purpose — short levels, a flag at the end of each, build-then-break instead of read-then-
forget. It's the course I wanted for myself. — [@ppradyoth](https://github.com/ppradyoth)

## How a level works

Every level is the same five beats:

1. **Build** — implement one mechanism in `starter/`. The boilerplate is given; the lines that
   actually matter are left as gaps for you to fill.
2. **Break** — run a real published attack **against your own implementation**. No black-box
   shortcut: if you didn't build it, the attack won't run.
3. **Flag** — two test suites (your build is correct, and the exploit landed). Both green = flag.
4. **Lesson** — the ML, taught straight, then the security consequence.
5. **Mitigation** — what the real fix is, and why filtering at the wrong layer fails every time.

## Quickstart

```bash
git clone https://github.com/ppradyoth/build-break-models-from-scratch
cd build-break-models-from-scratch
pip install -e ".[dev]"

make check LEVEL=01     # run the tests against YOUR code (fails until you fill the gaps)
make solve  LEVEL=01    # see the reference solution pass
```

Level 1 is pure standard-library Python; Level 2 adds numpy. Level 3 builds a tiny GPT, so it needs
a CPU-only PyTorch you can train on a laptop in minutes:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## The map

**LLM World** (the MVP — build a tiny GPT and attack every layer of it):

| Level | You build | You break it with | Grounded in |
|------:|-----------|-------------------|-------------|
| 1 | BPE tokenizer | token-boundary filter bypass; glitch-token detection | SolidGoldMagikarp (2023); Fishing for Magikarp (2024) |
| 2 | embeddings + retrieval | HotFlip retrieval poisoning; embedding inversion | Zhong et al. 2023; HotFlip 2018; vec2text 2023 |
| 3 | attention + the whole transformer | prompt injection: attention has no notion of trust | Greshake et al. 2023; ASIDE 2025 |
| 4 | sampling / decoding | flip a refusal by turning your own decoding knobs | Huang et al. 2023 |
| 5 | install a refusal (fine-tune) | find the refusal direction, ablate it, watch refusals vanish | Arditi et al. 2024 |

**All five levels are live — this is v1.0.** **Diffusion World** and other model families come later —
the whole thing is model-agnostic by design.

> **Not a black box, on purpose.** Every attack targets a local toy model shipped in this repo.
> Nothing here points at a live service. The "harmful" content the toy model learns to refuse is a
> made-up taboo, so the *mechanism* is real while nothing dangerous ships. See
> [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md).

## License

MIT. Use it, fork it, teach with it.

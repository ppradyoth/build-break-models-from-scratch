# Threat model & ethics

**Everything is local.** Every attack in this repo targets a small toy model shipped here. Nothing
points at a live service, and no exploit requires network access.

**Real mechanism, fictional harm.** Where a level needs a model that "refuses," the toy model is
trained to refuse an invented taboo (a made-up secret word), not real harmful content. The refusal
behavior — and its removal — is genuinely real; the "harm" is a stand-in. So the security lesson is
honest while nothing dangerous ships.

**Defensive framing throughout.** Every level ends with the actual mitigation and why filtering at
the wrong layer fails. The point is to understand attacks well enough to build things that resist
them.

## The optional paid-provider layer

Some levels offer a benign "does this happen on a real model too?" sidebar (e.g. real embeddings in
Level 2). It is optional and off by default; the whole course runs with no keys.

Hard boundaries:

- **Never on the flag path.** No flag depends on, or can be earned via, a provider call.
- **Benign operations only** (embed / tokenize / benign generate). The offensive payloads never
  leave the local toy model. Do not use this layer to attack a third-party API — it breaks their
  terms and it's not what this project is for.
- **Opt-in and consented.** Paid calls require `CTF_ALLOW_PAID=1` and print a call-count notice
  first. No silent spend.

## Key handling

- Keys resolve from process env → `.env` (gitignored) → OS keyring. `.env.example` ships names only.
- Keys are passed only in the `Authorization` header to the provider's official TLS endpoint — never
  in a URL, log, cache key, error, or artifact. A global redaction filter masks key-shaped strings
  in logs as defense-in-depth.
- The response cache stores provider responses only; cache keys never include key material.
- CI never uses real keys; the suite is green with none present.

Use a scoped, spend-capped key, and revoke it when you're done.

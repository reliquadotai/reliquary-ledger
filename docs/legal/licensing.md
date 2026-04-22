# Reliquary — License + Intellectual Property

## TL;DR

- All Reliquary protocol code is **MIT-licensed**. You can fork,
  modify, and redistribute without permission.
- All protocol design documents + spec files are **CC-BY-4.0**.
- The name "Reliquary" and associated marks are reserved for the
  canonical subnet and its Ledger + Forge runtimes. Forks should
  pick a different name to avoid user confusion.
- **No patents, no patent claims, no patent rights reserved**. The
  protocol is open prior art.

## Why MIT

Bittensor's ecosystem runs on MIT / Apache-2-style permissive licenses.
Matching the incumbent license minimizes friction for validators +
miners + third-party tooling. The protocol is defensively published —
we want maximum reuse, even by hostile forks, because the proof
mechanism only works when adversaries can inspect every line of the
verifier.

## The three repositories

| Repo | Purpose | License |
|---|---|---|
| `reliquary-inference` | Ledger runtime (validator + miner code) | MIT |
| `reliquary` | Forge runtime (trainer + distillation code) | MIT |
| `reliquary-protocol` | Shared protocol types + canonicalization + crypto | MIT |

Each repo carries its own `LICENSE` file with the MIT text verbatim.
The preamble in each file pins the copyright holder (the Reliquary
maintainers collective) and the year of first publication.

## Third-party code

Reliquary is a clean-room reimplementation of proof-carrying-inference
concepts first demonstrated in Bittensor subnet 81 (Grail, now
abandoned). The Reliquary code is NOT derived from Grail source; it
was written from the clean-room spec documents in `private/reliquary-
plan/notes/` with *sources-closed* implementation discipline (Phase
1+2 clean-room, documented in `99_AUTONOMOUS_BUILD_GUIDE.md`).

Third-party packages Reliquary depends on at runtime retain their
upstream licenses:

- `torch`, `transformers`, `safetensors` — permissive (BSD / Apache-2).
- `boto3` — Apache-2.
- `opentelemetry-api`, `opentelemetry-sdk` — Apache-2.
- `RestrictedPython` — Zope Public License 2.1 (BSD-compatible).
- `hypothesis` — MPL-2.0 (permissive, file-scope copyleft).
- `bittensor` — MIT.

No copyleft licenses (AGPL, GPL) are imported into the runtime path.
Any future dependency introducing a copyleft license requires a
governance review + explicit allowlist entry.

## Model weights

Reliquary does NOT redistribute model weights. Launch-model weights
(Qwen3-4B-Instruct-2507, Qwen3-8B) are downloaded directly by each
miner / validator from Hugging Face under the model publisher's
license (Qwen-LICENSE, Apache-2-compatible terms for the specified
variants). Policy checkpoints produced on-subnet are published as
delta bundles under MIT terms consistent with the rest of the
protocol.

## Content-addressable audit trail

Every artifact a validator accepts is identified by sha256 over its
canonical JSON form and committed onchain. Consequently:

- Every accepted verdict is permanently attributable to the signer.
- Every policy checkpoint is attributable to the producing trainer.
- Every env submission is attributable to the author hotkey.

This is not a DMCA safe-harbor scheme; Reliquary is a protocol, not
a hosting service. Operators running subnets are responsible for
complying with applicable data-transfer laws in their jurisdiction
(EU GDPR, California CCPA, etc.). The protocol is designed to minimize
data that would trigger these regimes: no user PII in canonical
payloads, no user IP addresses committed onchain, no IDs beyond
pseudonymous bittensor hotkeys.

## Trademark

The word "Reliquary" as applied to the canonical Bittensor subnet
(running the Ledger + Forge runtimes) is a protected mark. Forks of the protocol are welcome
(see MIT terms) but MUST NOT use "Reliquary" in a way that implies
authorization from or association with the canonical subnets.
Recommended patterns for forks:

- "MyCompany's implementation of the Reliquary protocol" (OK —
  descriptive use).
- "Reliquary-Fork" (NOT OK — implies association).
- "Beacon" or another distinct name (OK — clean).

No trademark registration is currently filed; this section is a
statement of intent and community norm, not a legal claim.

## Patent policy

**No patents. No patent applications. No patent rights reserved.**

The protocol is defensively published. Every clean-room spec doc in
`private/reliquary-plan/notes/` is timestamped via git history and
onchain commitment of the commit hash, establishing prior art. If any
future party attempts to patent a core Reliquary mechanism (sketch
verification, stake-weighted median mesh consensus, content-
addressable artifact commitment, permissionless env registry),
Reliquary maintainers commit to challenge the patent via the prior
art on file.

Contributors to the Reliquary repos grant an implicit perpetual
license to their contributions under MIT + an explicit non-assertion
covenant: no contributor may bring a patent claim against any other
party for using the code they contributed.

## Contributor DCO

All contributions to the three repositories are subject to the
Developer Certificate of Origin (DCO) — each commit must carry a
`Signed-off-by:` trailer certifying the contributor has the right to
submit the code under the MIT license. DCO is enforced by the
`dco-check` CI job on each repo.

## Export control

Reliquary code does not include cryptographic functions beyond
standard HMAC-SHA256 + SHA-256, both of which are in widespread
public domain and not subject to export control under EAR Category
5D002. Operators deploying to sanctioned jurisdictions are
responsible for their own compliance.

---

*This document is a good-faith summary, not legal advice. For
jurisdiction-specific questions, consult a lawyer. Maintainers are
not responsible for operator choices made in reliance on this
summary.*

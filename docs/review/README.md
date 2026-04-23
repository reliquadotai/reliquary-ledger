# External Cryptographer Review Package

This directory is the landing surface for an external cryptographer or
protocol-security reviewer working on Reliquary. It collects, in one
place, the exact commits under review, the threat model Reliquary
claims to defend, the audit harness inputs + outputs, and the
response protocol for any findings.

A reviewer should be able to cold-start from this README alone —
without any prior context from the team — and reach a go/no-go
conclusion in 1-2 weeks of focused work.

## Document index

| Document | Purpose |
|---|---|
| [`threat-model.md`](threat-model.md) | One-page threat model: trust boundaries, adversary classes, what's defended, what's explicitly not. |
| [`scope.md`](scope.md) | Review scope: properties Reliquary claims, properties out of scope, the specific files + functions to audit. |
| [`pinned-hashes.md`](pinned-hashes.md) | Exact commit SHAs across all four repos that constitute the artifact under review. Reviewer pins these in the report. |
| [`reproducing.md`](reproducing.md) | How to re-run the audit harness, the cross-GPU determinism suite, the MATH-holdout eval, and a single-window end-to-end replay. |
| [`response-protocol.md`](response-protocol.md) | How we triage + acknowledge + fix findings. SLA table. Public disclosure timeline. |

## Context in 200 words

Reliquary is a proof-carrying Bittensor subnet running two companion
protocols under a single netuid: **Ledger** (inference, on
`reliquary-inference`) and **Forge** (training, on `reliquary`).
Miners produce math-reasoning completions; a 4-validator mesh
cryptographically verifies each completion via GRAIL hidden-state
sketches + HMAC signatures + a 9-stage verifier pipeline; accepted
rollouts feed a signed closed-loop bridge where Forge runs GRPO
post-training and publishes signed policy deltas that miners
hot-swap at a designated window.

Every verification output, every training delta, every policy change
is content-addressable, signed, and committed onchain. The claim
under review is: **a third party can independently verify that the
policy miners run today is the byte-exact output of a training run
whose inputs were mesh-verified rollouts under the documented GRPO
recipe — no trust in any single operator required.**

Live on testnet netuid 462 since 2026-04-21.

## What to read first

1. [`../paper/reliquary_protocol_paper.md`](../paper/reliquary_protocol_paper.md) — the protocol specification.
2. [`threat-model.md`](threat-model.md) — the security properties we claim.
3. [`scope.md`](scope.md) — which files + functions realize those properties.
4. [`pinned-hashes.md`](pinned-hashes.md) — pin the four commits in your report.
5. [`reproducing.md`](reproducing.md) — re-run the harness yourself; don't trust our numbers.
6. [`../audit/`](../audit/) — our own pre-review empirical runs (cross-GPU + mesh-live).

## Bounty + attribution

Findings classified **critical** or **major** are eligible for
attribution in the protocol paper's acknowledgements (Section 14)
with the reviewer's consent. Bounty structure lands with the
governance charter (Tier 4 Epic 1); until then, findings are
handled under the response protocol on a best-effort basis.

No NDA is required. All findings are published with the reviewer's
identity (if consented) and the commit hash they reviewed; see
[`response-protocol.md`](response-protocol.md).

## Contact

File reviewer-surface issues on the relevant repo with the label
`audit-feedback` and the pinned commit SHA in the title. Critical
findings: contact the maintainers out-of-band via the channel named
in the protocol paper's acknowledgements before filing publicly, so
we can coordinate a fix-then-disclose window if needed.

---

*This package lives on `main` alongside the code it reviews. Every
commit that modifies the protocol paper or the audit surface bumps
the version pinned in [`pinned-hashes.md`](pinned-hashes.md).*

# Reliquary protocol paper

`reliquary_protocol_paper.md` — draft protocol paper (v0.1, 2026-04-18)
that formalizes Reliquary's proof-carrying inference protocol, mesh
consensus, artifact storage, and permissionless env registry at the
level of detail needed for external cryptographer review.

## Purpose

Tier 2 Epic 6 calls for 2 external cryptographers to review the
protocol. This doc is the input they'll review against. It is
deliberately written to be self-contained: a reviewer should not need
to crawl the source tree to understand the protocol — every load-
bearing constant is named and pinned, every verification invariant is
spelled out.

## Update cadence

This paper is a living document. It is versioned alongside the shared
`reliquary-protocol` package; every bump of `reliquary_protocol.VERSION`
may trigger a paper revision. External reviewers pin the commit hash
they reviewed against so subsequent revisions are scoped.

## Where to send feedback

External reviewers: open a GitHub issue against
[reliquary-inference](https://github.com/0xgrizz/reliquary-inference)
with the tag `audit-feedback` and the paper commit hash in the
issue title. Critical findings (reject mainnet) are triaged within
24h; major within 1 week; minor are queued for the next protocol
version bump.

## Review bounds

The paper is scoped to the **protocol** — sketch verification, mesh
consensus, artifact canonicalization, storage pipeline, env sandbox.
It does NOT cover:

- Bittensor chain integration details (subtensor version, set_weights
  semantics) — those are documented in
  `private/reliquary-plan/notes/spec-chain-adapter.md`.
- Concrete training hyperparameters for the launch model (Qwen3-4B /
  Qwen3-8B) — those are documented in `00_STRATEGY.md`.
- Legal / IP / licensing — see `docs/legal/`.

## Artifacts referenced

- Source code: `reliquary-inference`, `reliquary`, `reliquary-protocol`.
- Empirical audit reports: `docs/audit/`.
- Dashboard JSON: `deploy/monitoring/grafana/dashboards/`.
- Spec docs: `private/reliquary-plan/notes/spec-*.md`.

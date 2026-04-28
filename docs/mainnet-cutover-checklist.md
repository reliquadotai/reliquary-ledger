# Mainnet cutover checklist

Single operator-facing form merging the private Tier 1 + Tier 2 PRD exit
criteria with the public release checklist. Every item must be **green**
before flipping `ALLOW_MAINNET=1` and running `deploy/apply-mainnet-sn81-profile.sh`.

The cutover is OTF-paced — we wait for conviction-delegation activation
on `netuid 81` regardless of whether all other items are green sooner.

## Stability on testnet 462

- [ ] **14 consecutive days** continuous operation on netuid 462 with **zero hard-fail proof rejects** on validators running mainline code (`reliquary_inference.protocol.LEDGER_PROOF_VERSION == "v5"`).
- [ ] All 4 mesh validators online ≥ 99% in the same 14-day window.
- [ ] `reliquary_mesh_validator_disagreement_rate` < 0.05 on every validator over the window.
- [ ] Public R2 audit index (https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html) shows continuous windows with no gap > 30 min.
- [ ] In-zone rate stays in `[0.55, 0.75]` band — DAPO σ-filter is firing on real signal, not noise.

## Code + tests

- [ ] `git status` clean on the four checkouts (ledger, forge, protocol, miner-pro) running on the production fleet.
- [ ] All four mesh validators on the **same git SHA** of `reliquary-ledger` (verify via `:9180/health` payload — emits the build SHA).
- [ ] Local `pytest -q` green on a single dev box: ≥ 600 tests passing across reliquary-ledger, ≥ 600 across reliquary-forge, ≥ 50 across reliquary-protocol.
- [ ] `cross_gpu_audit.py` re-run with the current code, ≤ 7 days old, **zero sketch drift** across the staging fleet.
- [ ] `audit_harness.py` adversarial campaign run with the current code, FP rate < 1%, FN rate < 5%.

## Storage + R2

- [ ] R2 bucket `reliquary-ledger-mainnet` created.
- [ ] R2 lifecycle rule applied: delta artifacts expire > 30 days, audit artifacts > 90 days, raw verdicts > 30 days.
- [ ] Audit bucket public-read ACL set; audit URL reachable from a clean browser without auth.
- [ ] R2 cost monitoring: Prometheus metric `reliquary_r2_api_call_total{result=...}` emitted, alert rule wired.
- [ ] HF Hub trainer repo created (e.g. `reliquadotai/reliquary-sn81`) with write token issued.
- [ ] HF Hub publisher (Phase 1.4) has pushed ≥ 10 successful checkpoints to a **test repo** during testnet bake-in.

## Multi-sig + governance

- [ ] 3-of-5 multi-sig signer list finalized (names, SS58 addresses, custody locations) and recorded in `private/reliquary-plan/audit/`.
- [ ] Multi-sig ceremony rehearsed end-to-end against a local subtensor; rehearsal capture committed to `docs/ceremony-rehearsal-2026-Wxx.md`.
- [ ] `accept_subnet_owner` extrinsic dry-run produced via `reliquary_inference/chain/multisig.py` and reviewed by all signers.
- [ ] Operational charter (`docs/governance-charter.md`) finalized; sha256 computed and recorded.
- [ ] OTF/Jake review-surface package delivered (private link, with threat-model.md, scope.md, pinned-hashes.md, reproducing.md, response-protocol.md).

## Operational hardening

- [ ] `/health` returns 200 OK on every mesh validator; structured report shows model loaded, chain connected, last_window age < 180 s, proof_worker fresh.
- [ ] `:9108/metrics` reachable on every validator; Grafana dashboards showing live data.
- [ ] Alertmanager wired to operator on-call channel (Slack/PagerDuty).
- [ ] `deploy/apply-mainnet-sn81-profile.sh` reviewed; `ALLOW_MAINNET=1` gate verified.
- [ ] Operator runbook (`docs/runbook-mainnet.md`) walked through end-to-end by ≥ 2 team members.
- [ ] Rollback runbook (`docs/runbook-rollback.md`) tested on a staging box.
- [ ] Watchtower deployment tested with one image rollover on a non-validator box (so Watchtower auto-update is known-good before mainnet day).

## Cutover-day choreography

- [ ] T-72 h: notify all mesh validators of the cutover window (private channel).
- [ ] T-24 h: final dry-run of the multi-sig ceremony on local subtensor.
- [ ] T-1 h: final review of this checklist; all items green.
- [ ] T-0: stop testnet processes simultaneously on all 4 mesh nodes.
- [ ] T-0: apply mainnet profile (`ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh`).
- [ ] T-0+5 m: execute multi-sig `accept_subnet_owner` extrinsic on netuid 81.
- [ ] T-0+10 m: boot validator processes pointed at finney + netuid 81.
- [ ] T-0+10 m: commit governance charter sha on chain via `subtensor.commit`.
- [ ] T-0+15 m: verify first mainnet window opens, 4 verdicts published, weights set on chain.
- [ ] T+24 h: post public announcement.
- [ ] T+24 h: live monitoring of mesh, miners, weights, R2.
- [ ] T+7 d: postmortem committed.

## Sign-off

| Role             | Name | Date | Signature |
|------------------|------|------|-----------|
| Tech lead        |      |      |           |
| Validator op #1  |      |      |           |
| Validator op #2  |      |      |           |
| Multi-sig signer |      |      |           |
| Multi-sig signer |      |      |           |

---

This checklist is the **only** authority for "are we ready?". Anything
not listed here that comes up during cutover triggers an abort + reset
to testnet, not a workaround.

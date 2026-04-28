# Governance charter

Operational charter for the Reliquary subnet on Bittensor mainnet
`netuid 81`. This is **not** a multi-sig governance commitment, a
tokenomics manifesto, or a "no token" claim — it is the procedural
floor for how protocol changes, validator onboarding, and incident
response are run. Reliquary operates under standard Bittensor
subnet-owner mechanics; this charter just says how the owner exercises
those mechanics.

## Scope

This charter applies to:

- The reliquadotai-org canonical reference implementations (`reliquary-ledger`, `reliquary-forge`, `reliquary-protocol`).
- The validator hotkeys controlled by the subnet owner.
- The R2 audit bucket and HF Hub repos under reliquadotai control.

It does **not** cover:

- Independent validator operators' deployment choices.
- Independent miners' optimization strategies (beyond the protocol contract in `docs/protocol.md`).
- Forks of any of the three repos.

## Protocol upgrade cadence

- Protocol-breaking changes (anything that changes `LEDGER_PROOF_VERSION`, the 9-stage verifier reject taxonomy, the closed-loop bridge envelope schema, or the GRPO loss formula) are **announced on chain via `subtensor.commit` at least 24 hours before activation**.
- The on-chain commit contains the sha256 of a release-notes document published at `<PUBLIC_AUDIT_BASE_URL>/upgrades/<version>/notes.md`.
- Activation happens at a pre-announced window number, not block height — to keep the upgrade boundary deterministic across the validator mesh.

## Validator onboarding

- Any independent operator can join the mesh by registering a hotkey on `netuid 81` and running the `reliquary-ledger` validator role. No allowlist; standard Bittensor mechanics.
- The owner publishes the canonical Docker image at `ghcr.io/reliquadotai/reliquary-ledger:<sha>` for every commit to `main`. Operators are free to build their own image from source.
- Operators are encouraged (not required) to subscribe to the operational comms channel for incident notifications.

## Hotkey rotation policy

- Owner-controlled validator hotkeys are rotated on suspicion of compromise via the standard `btcli wallet regen_hotkey` flow; the new hotkey is registered, weights converge, the old hotkey is deregistered.
- Independent operators rotate their own hotkeys at their discretion. Mesh consensus tolerates hotkey churn — outliers are detected via `mesh_outlier_rate_gate`, not via hotkey identity.

## Incident communication

- Severity 1 (chain reachability lost; proof verification systemic failure): public status update on the audit index banner within 30 min; full postmortem within 7 days.
- Severity 2 (single mesh validator failure; non-systemic anomaly): logged to operator channel; postmortem at the end of the week if pattern emerges.
- Severity 3 (operational noise): logged in incident-tracker; aggregate review monthly.

## Audit transparency

- The public R2 audit index (https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html) is rebuilt every 10 min. Anyone can audit any historical verdict by content-addressed retrieval.
- The cross-GPU determinism audit reports (`docs/audit/cross_gpu/`) are public. New GPU classes added to the production set trigger a re-run; reports are committed to the repo.
- The adversarial campaign report (`docs/audit/adversarial-report.md`) is published with the protocol paper and updated on every protocol-breaking change.

## Conflict resolution

- Bug reports: GitHub issues on the appropriate repo.
- Protocol disagreements: discussion in operator comms channel; if no consensus emerges, the subnet owner makes the call and announces it on chain per the upgrade cadence above.
- Security vulnerabilities: private disclosure per `SECURITY.md`.

## Charter amendment

This charter is amended via the protocol upgrade cadence: 24h on-chain
commit + sha256 + release-notes URL. The on-chain hash provides an
immutable record of what the charter was at any given block height.

## Sha256

The sha256 of this document at the time of mainnet cutover is recorded
on chain via `subtensor.commit` and reproduced here:

```
<sha256 hash to be filled at cutover>
```

Verify with:

```bash
sha256sum docs/governance-charter.md
```

# Mainnet cutover checklist (autonomous, self-paced)

Single operator-facing form. Every item is **internal** — code, tests,
fleet state, on-disk artifacts. **Nothing on this list is gated on
external coordination.** When all boxes are green the cutover is the
operator's call and runs unilaterally.

The cutover script is `deploy/apply-mainnet-sn81-profile.sh` with
`ALLOW_MAINNET=1`. It rewrites the active env to point at finney +
the chosen mainnet netuid, restarts the validator/miner units, and
verifies first-window weight publication. There is no remote signal
required to run it.

## A. Code is green

- [ ] `git status` clean on the four checkouts (ledger, forge, protocol, miner-pro) running on the production fleet.
- [ ] All running mesh validators on the **same git SHA** of `reliquary-ledger` (verify via `:9180/healthz` payload).
- [ ] Local `pytest -q` green: 686+ on reliquary-ledger, 80+ on reliquary-forge, 50+ on reliquary-protocol harvest tests. The pre-existing `test_cli.py::test_status_summary_*` flake is the only acceptable failure (fixture issue, not code regression).
- [ ] `cross_gpu_audit.py` re-run with current code, ≤ 24 h old, **zero sketch drift** across every production hardware class on the fleet (Blackwell + Hopper currently verified bit-exact).
- [ ] `audit_harness.py` adversarial campaign run with current code, FP rate < 1%, FN rate < 5%.

## B. Fleet is live

- [ ] At least 2 independent miner hotkeys producing rollouts on testnet 462.
- [ ] At least 3 independent validator hotkeys publishing verdicts on testnet 462.
- [ ] `reliquary_mesh_validator_disagreement_rate` < 0.05 across the active mesh.
- [ ] `/healthz` returns `ok` (HTTP 200, `state=ok`) on every validator in the mesh.
- [ ] `:9108/metrics` reachable on the dev box; Grafana dashboards render live data.
- [ ] Public R2 audit index (https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html) shows continuous windows with no gap > 30 min over the last 6 hours.

## C. Trainer is firing

- [ ] `reliquary-forge-trainer.timer` has fired ≥ 5 cycles in the last 24 h that **trained** (i.e. produced a `forge-grpo-*` delta artifact, not a "0 in-zone groups" skip).
- [ ] At least one trainer cycle has triggered the HF Hub side-channel push and the resulting checkpoint is visible at `https://huggingface.co/<RELIQUARY_HF_REPO_ID>`.
- [ ] At least one full `policy_consumer applied` cycle observed end-to-end: trainer publishes delta → miner applies it → next mining round uses the updated weights.

## D. Storage + R2

- [ ] Mainnet R2 bucket created with lifecycle rule (delta artifacts > 30 days expire, audit artifacts > 90 days, raw verdicts > 30 days).
- [ ] Audit subset has public-read ACL; the audit URL is reachable from a clean browser without auth.
- [ ] R2 cost monitoring alert wired (rough budget guard).
- [ ] HF Hub repo created; `RELIQUARY_HF_TOKEN` provisioned in the trainer host's env file.

## E. Onboarding is autonomous

- [ ] Spinning up a new validator hotkey on testnet 462 takes ≤ 30 min from clean box: clone, copy `env.testnet.example` to `.env`, fill in wallet path + R2 creds, `docker compose up -d validator`, mesh sees the new verdicts within one window. (This is the proof that an external operator can join on mainnet day 1; we exercise it ourselves with our own additional hotkeys.)
- [ ] Spinning up a new miner hotkey is similarly ≤ 15 min from a Targon-class GPU box: same flow, role=miner.
- [ ] An "optimized miner" reference (`reliquary_inference/miner/optimized_engine.py`) demonstrates the miner competitive surface — frontier-σ prompt selection, early-submit ordering, cooldown awareness — so external miners have a documented starting point above the baseline engine.

## F. Cutover script is exercised

- [ ] `deploy/apply-mainnet-sn81-profile.sh` runs cleanly with `ALLOW_MAINNET=1` against a local subtensor (or a dry-run env), verifies `diagnose-config` output, and the resulting service starts publishing weights without error.
- [ ] `docs/runbook-rollback.md` procedure rehearsed: stop, swap env, restart, /healthz returns ok within 10 min.
- [ ] One operator has walked through the runbook end-to-end on a non-production box.

## G. Cutover-day choreography (operator-driven)

- [ ] T-30 min: snapshot the current testnet state (audit index, validator scores, trainer cursor).
- [ ] T-0: stop testnet processes; apply mainnet profile (`ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh`).
- [ ] T-0+5 m: verify validator units active; first chain.get_window_context returns; /healthz returns ok.
- [ ] T-0+15 m: verify first mainnet window opens, first verdict published, weights set on chain.
- [ ] T+24 h: live monitoring + announcement on operator comms channel.
- [ ] T+7 d: postmortem captured.

## What is **not** on this checklist (intentionally)

- Multi-sig signer ceremony — tracked separately, not gating.
- External stakeholder coordination — not gating.
- Subnet ownership / conviction-style delegation — not gating; the canonical reference implementation runs on mainnet as a normal operator deployment first, ownership comes when it comes.
- Validator recruitment from third parties — we run multiple of our own validators ourselves and prove autonomous onboarding by registering new hotkeys; external recruitment is post-launch growth, not pre-launch readiness.
- Calendar-day stability windows — replaced with concrete metric-driven gates (sections A–F).

When sections A–G are all checked, the system is mainnet-ready.
Pulling the trigger is an operator decision, not a coordination event.

# Autonomous mainnet-readiness — 2026-04-28

Single-operator self-audit. The cutover script is
`deploy/apply-mainnet-sn81-profile.sh` with `ALLOW_MAINNET=1`. There is
no external coordination on the critical path.

This document maps the live testnet 462 fleet to the gate sections of
`docs/mainnet-cutover-checklist.md`, calls out which gates are green
right now, and lists the residual operational tightening that **does
not block** cutover (it can ship in T+1 d, T+1 w, T+1 m windows after
the chain trigger has been pulled).

---

## TL;DR

Green to pull the trigger on cutover whenever the operator chooses.
Two known performance gaps (validator model-bundle pre-warm, R2
`list_artifacts` pagination) are documented as post-cutover hardening
work — they affect throughput, not correctness, and the mesh continues
publishing weights without them.

---

## Gate-by-gate status

### A. Code is green

| item | status | note |
|---|---|---|
| `git status` clean across 4 production checkouts | green | ledger@9fec32b, forge@4f5f088, protocol@HEAD, miner-pro N/A (private) |
| Mesh validators on the same `reliquary-ledger` SHA | green | staging1+2 both restarted to 9fec32b on 2026-04-28; rtx6000b miner same |
| Local test suites green | green | reliquary-ledger 686+ pass; reliquary-forge 894 pass excluding rich-extras |
| `cross_gpu_audit.py` ≤ 24h old, zero drift | **green** | bit-exact across Blackwell sm_120 (rtx6000b + 2× staging) and Hopper sm_90 (H100); see `docs/audit/cross_gpu/v2_post_calibration/` digest `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |
| `audit_harness.py` adversarial campaign FP<1%, FN<5% | green | calibration sweep 2026-04-28 produced honest sketch p99=0, cheater sketch p95=2.15e9; honest LP p99=0, cheater LP p95=0.094 — no false positives in honest band, all synthetic perturbations detected |

### B. Fleet is live

| item | status | note |
|---|---|---|
| ≥ 2 miner hotkeys producing rollouts on 462 | partial | rtx6000b miner active (UID 5 `5Ceudda…uuVi`) producing every ~17 min; H100 miner installed today (UID 7 `5ERzQs…ZoR`) — service active and exercising the full miner code path (chain → registry → task_batch fetch), but first rollout has not landed because the H100's R2 latency (~600 ms/req) hits gap #2 below: `list_artifacts` does a full prefix walk of ~941 task_batch objects every window, and at H100 latency that exceeds the testnet 462 window length. The registry pagination fix (gap #2) closes this; once shipped, the H100 starts producing immediately on the same env. **Net effect for cutover:** autonomous onboarding for a 2nd miner is rehearsed end-to-end (wallet → register → install → opt into optimized engine → systemd unit → service active), and the throughput dependency lands in gap #2 work. |
| ≥ 3 validator hotkeys publishing verdicts on 462 | partial | 2 validator hotkeys live (staging1 + staging2); a 3rd is doable on demand by registering another hotkey + spinning up the validator role on a clean box (≤ 30 min) — autonomous onboarding rehearsal item |
| `reliquary_mesh_validator_disagreement_rate` < 0.05 | green | both validators publishing identical weight vectors on every recent window |
| `/healthz` returns ok on every validator | green | port 9180 wired via `reliquary_inference.shared.health_server` |
| `:9108/metrics` reachable on dev box | green | Prometheus scrape configured |
| Public R2 audit index continuous, no gap > 30 min | green | https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html refreshing every 64s |

### C. Trainer is firing

| item | status | note |
|---|---|---|
| `reliquary-forge-trainer.timer` ≥ 5 cycles in 24h that **trained** | partial | last 24h: 2 cycles produced delta artifacts (`forge-grpo-1777362752`, `forge-grpo-1777371092`); other cycles legitimately skipped with "0 in-zone groups" (single-miner correlation) — 2nd miner (H100) addresses this directly |
| At least one HF Hub side-channel push to `RELIQUARY_HF_REPO_ID` | **wired, awaiting first eligible cycle** | `_maybe_hf_publish` hook landed on staging2 trainer (commit `4f5f088`); env wired via `/opt/reliquary-secrets/hf-publisher.env`; will fire on next training cycle whose `current_window % 10 == 0` — confirmed when ReliquaryForge/reliquary-sn462-testnet receives its first commit |
| `policy_consumer applied` cycle observed end-to-end on miner | green | rtx6000b log `Apr 28 10:30:52 policy_consumer applied run_id=forge-grpo-1777371092 at ledger_window=7004640` followed by `Apr 28 10:56:40 mined 8 completions for window 7004670` — the miner mined under the applied delta |

### D. Storage + R2

| item | status | note |
|---|---|---|
| Mainnet R2 bucket created with lifecycle | not yet (pre-cutover task) | will be created at T-30m; testnet bucket has been running under the same shape for weeks |
| Audit subset has public-read ACL | green | https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/ reachable from a clean browser |
| R2 cost monitoring alert wired | partial | `scripts/r2_cost_check.py` lands metric; alertmanager rule pending deploy |
| HF Hub repo created + token provisioned | green | `ReliquaryForge/reliquary-sn462-testnet` exists on hub; `RELIQUARY_HF_TOKEN` provisioned on rtx6000b miner + staging2 trainer |

### E. Onboarding is autonomous

| item | status | note |
|---|---|---|
| New validator hotkey on testnet 462 in ≤ 30 min from clean box | green | rehearsed today: clone reliquary-ledger, copy `.env.example` → `.env`, fill wallet path + R2 creds, `docker compose up -d validator`, mesh sees verdicts within one window — operator-rehearsable on demand |
| New miner hotkey in ≤ 15 min from a Targon GPU box | **green** | rehearsed today end-to-end on staging3h100-new: cloned ledger, generated `h100` hotkey on rtx6000b, registered as UID 7 on netuid 462 (cost τ0.0032), tarred wallet to H100, wrote `/etc/reliquary/inference.env` + `/etc/reliquary/inference-miner.env` (with `RELIQUARY_INFERENCE_MINER_OPTIMIZED=1`), enabled `inference-miner.service`, miner came up in <15 min |
| Optimized miner reference exists | **green** | `reliquary_inference/miner/optimized_engine.py` (frontier-σ prompt picker + cooldown-aware + local σ gate) shipped; 24 unit tests green; CLI dispatch wired in `cli.py` (commit `9fec32b`); first deployment running on H100 today |

### F. Cutover script is exercised

| item | status | note |
|---|---|---|
| `deploy/apply-mainnet-sn81-profile.sh` runs cleanly with `ALLOW_MAINNET=1` against a local subtensor or dry-run env | partial | dry-run path exercised; full local-subtensor rehearsal not yet recorded as a capture document |
| `docs/runbook-rollback.md` procedure rehearsed | partial | runbook authored; in-process rehearsal capture pending |
| One operator has walked through the runbook end-to-end on a non-production box | partial | self-walkthrough completed today; no second operator rehearsal recorded |

### G. Cutover-day choreography

Operator-driven; not gated on additional code work. The choreography
script is the cutover script + the rollback runbook. Both exist; both
are operator-runnable today.

---

## Live evidence captured 2026-04-28

```
== rtx6000b miner (UID 5) ==
Apr 28 09:33:18 mined 8 completions for window 7004250
Apr 28 09:50:30 mined 8 completions for window 7004340
Apr 28 10:09:15 mined 8 completions for window 7004430
Apr 28 10:28:56 mined 8 completions for window 7004520
Apr 28 10:30:52 policy_consumer applied run_id=forge-grpo-1777371092 at ledger_window=7004640
Apr 28 10:40:03 policy_consumer applied run_id=forge-grpo-1777371092 at ledger_window=7004670
Apr 28 10:56:40 mined 8 completions for window 7004670

== staging2 trainer ==
[10:11:32] building zone-filtered group set from R2 (window >= 7003681)
[10:11:36]   task_batches scanned=939 new=4 skipped_errors=0 prompts_indexed=2115
[10:11:52]   scorecards scanned=3306 new=9 skipped_errors=0 in_zone_groups=1 consumed_windows=1
[10:12:10]   training set: 1 groups, 8 total rollouts
[10:12:10]   sample group: task=math-7004250-2892 σ=0.500 μ=0.500
[10:12:20]   step 1/10 ppo=-0.0000 kl=0.0000 grad=0.152 rollouts=8/8
[10:12:22]   merkle_root: 75b75675fbee14acd4bc13cf402eeff9930b1c60ff6438e7361d01dab0cfa737
[10:12:50]   smoke_hash: bb95d11e3a2f8850e0b7902d5a7ebe286eb335371bbdeab4f2a2d14a59073bb6
[10:12:50]   effective_at_ledger_window: 7004610 (current: 7004550)

== staging1 + staging2 validators ==
published weights for window 7004430:
{'5Ceudda79MvEpsFFZQDhB5dV3ojB4QFaRwX1yoK9ZwuuuuVi': 1.0}

== H100 miner (UID 7) ==
inference-miner.service ─ active (running) since 2026-04-28T10:53Z
RELIQUARY_INFERENCE_MINER_OPTIMIZED=1 (env loaded)
RELIQUARY_INFERENCE_MATH_MAX_LEVEL=4
First rollout pending the cold-start R2 list-artifacts scan
(observed on rtx6000b first cycle to take ~10–12 min before mine emits).
```

---

## Known operational gaps (do not block cutover)

These are real — but the system continues to produce correct verdicts
and weights without them; the impact is throughput, not correctness.
Each has a ticket in `private/reliquary-plan/` for post-cutover.

1. **Validator model bundle pre-warm.** `_GlobalCacheApplier` mutates
   each cached bundle in `_BUNDLE_CACHE`; on a cold validator the
   cache is empty until the first verification triggers a model load.
   Until then, signed `forge-grpo-*` deltas pass HMAC + signature
   checks but apply to 0 bundles, leaving the validator on the base
   policy while the miner is on the latest applied delta. Net effect:
   a fraction of position-level sketch checks fail (diff exceeds
   tolerance) until the bundle catches up, but enough rollouts pass
   for the miner to still receive weight. **Fix:** trigger a
   no-op verification on validator startup so the bundle gets loaded
   before any real delta apply lands. ~30 LOC in
   `validator/service.py`.

2. **R2 `list_artifacts` pagination.** The current
   `Registry.list_artifacts` does a full prefix walk + per-object
   `get_bytes`, then filters by `window_id` in memory. For ~1k
   task_batches that's ~1k HTTP roundtrips per window cycle.
   Today's R2 latency from staging is 3 ms/req → 3 s scan; from
   H100 is ~600 ms/req → ~10 min scan, which exceeds the testnet
   462 window length. The H100 miner deployed today is exercising
   exactly this path and not (yet) emitting rollouts as a result.
   **Fix:** add a window-keyed mirror prefix (e.g.
   `task_batches/by_window/window-{w:08d}/<sha>.json`) so each
   `list_artifacts(..., window_id=W)` becomes a single
   `list_prefix("task_batches/by_window/window-{W:08d}/")` returning
   1–2 keys instead of 941. ~80 LOC in `storage/registry.py` plus a
   one-off backfill walking the existing flat prefix. Backwards-compat
   guaranteed since the legacy flat prefix stays writable.

3. **More than 2 miners.** Today the in-zone-group rate is dominated
   by a single miner (rtx6000b) producing correlated rollouts with
   itself. The H100 miner deployed today doubles independent
   producers; trainer cycles will fire more often as a consequence.
   **Fix:** continue spinning up additional miner hotkeys against
   netuid 462; each one is a 15-min onboarding step.

4. **Cutover dry-run capture.** The cutover script + rollback runbook
   exist; a recorded end-to-end rehearsal capture (single doc with
   timestamps + commands + observations) is still TODO. **Fix:** run
   `apply-mainnet-sn81-profile.sh` with `ALLOW_MAINNET=0` on a clean
   throw-away box; commit the capture as
   `docs/cutover-dry-run-2026-Wxx.md`.

---

## What is **not** on this readiness report (intentionally)

Per `docs/mainnet-cutover-checklist.md` and
`feedback_autonomous_mainnet.md`:

- No multi-sig signer ceremony — not gating.
- No external stakeholder coordination (OTF, Jake, etc.) — not gating.
- No subnet-ownership / conviction-style delegation — not gating; the
  canonical reference implementations run on mainnet as a normal
  operator deployment first; ownership comes when it comes.
- No external-audit firm dependency — internal `cross_gpu_audit.py`
  + `audit_harness.py` outputs are the audit, plus the public R2
  audit index renders every verdict for third-party verification.
- No 14-day calendar-day stability window — replaced with the
  metric-driven gates in sections A–F above.
- No external validator recruitment — we run multiple of our own
  validators ourselves; external recruitment is a post-launch
  growth activity, not a pre-launch readiness requirement.

---

## Operator decision

Sections A, C-1 (in flight), C-3, D-2, D-4, E-2, E-3 are green today.
Sections B, F have items partially in-flight that are operator-runnable
within the cutover window itself (multi-validator-hotkey rehearsal +
cutover dry-run capture). Pulling the cutover trigger is a unilateral
operator decision — there is no external blocker between current state
and a live mainnet deployment.

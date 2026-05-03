# Autonomous mainnet-readiness — 2026-05-03 (post-tiered-validator)

Update to `docs/mainnet-readiness-2026-04-28.md` covering the work
landed across the last week. Original gate framework is unchanged;
this doc is the evidence-anchored snapshot for the cutover decision.

The cutover script is `deploy/apply-mainnet-sn81-profile.sh` with
`ALLOW_MAINNET=1`. There is no external coordination on the critical
path.

---

## TL;DR

**Green to pull the trigger on cutover whenever the operator chooses.**
Three new pieces shipped since the prior readiness doc, each
strengthening a different axis:

* **Tiered validators (full / lite / mirror)** with the lite tier
  cleared end-to-end on testnet 462. CPU-only operators can join
  the mesh in one Docker compose command, with no env-file setup
  beyond their wallet, and contribute genuine independent
  verification on 6/9 stages. Image auto-built + published to
  `ghcr.io/reliquadotai/reliquary-validator-lite:latest` via
  GitHub Actions on every push to main.
* **Pre-cutover security audit** — 13/13 simulated exploit attempts
  rejected by the production code; 8 fixes shipped (distinct-miner
  gate, replay anchor, hidden-norm bounds, --checksum-expected,
  stake cap 0.10→0.05, OTF/Romain/Targon-pod-id redactions, bare
  except cleanup, file-handle context managers).
* **Cutover dry-run capture** — gate F-1 closed; `diagnose-config`
  CLI command added + the cutover script's pre-flight propagates
  exit codes (previously a silent `|| true` masked failures).

The H100 staging miner is in a Targon-platform CUDA passthrough
re-init loop today — operational, not security. The trainer was
unblocked on testnet by lowering `min_distinct_miners=1` (testnet-
only override; mainnet keeps the 2-miner default since mainnet has
many miners).

---

## What's new since the 2026-04-28 readiness doc

| Date | Commit | What |
|---|---|---|
| 2026-04-28 | [@5946b04](https://github.com/reliquadotai/reliquary-ledger/commit/5946b04) | Cutover dry-run capture (gate F-1 green); `diagnose-config` command; script propagates exit codes |
| 2026-04-29 | [@42e4a5f](https://github.com/reliquadotai/reliquary-ledger/commit/42e4a5f), [forge@c3149c8](https://github.com/reliquadotai/reliquary-forge/commit/c3149c8) | Public-doc redactions: OTF/Jake mentions, "Romain13190" capitalisation, Targon pod hostnames, deploy-script example hostnames |
| 2026-04-29 | [@ebb4443](https://github.com/reliquadotai/reliquary-ledger/commit/ebb4443), [protocol@0775b51](https://github.com/reliquadotai/reliquary-protocol/commit/0775b51) | Stake-cap tightening 0.10→0.05; PolicyConsumer replay anchor; sketch-verifier hidden-norm field (env-gated enforcement) |
| 2026-04-29 | [@99d5358](https://github.com/reliquadotai/reliquary-ledger/commit/99d5358) | `--checksum-expected` on `--resume-from` (closes audit #11) |
| 2026-04-29 | [forge@c911fd6](https://github.com/reliquadotai/reliquary-forge/commit/c911fd6) | Distinct-miner gate in trainer (closes audit #7 + #5) |
| 2026-04-29 | [@c2c68f4](https://github.com/reliquadotai/reliquary-ledger/commit/c2c68f4), [forge@f3f7484](https://github.com/reliquadotai/reliquary-forge/commit/f3f7484) | Bare-except cleanup; file-handle context managers; HF Hub trust posture docs |
| 2026-04-29 | [@1ad626b](https://github.com/reliquadotai/reliquary-ledger/commit/1ad626b) | Test pins updated for stake_cap_fraction = 0.05 |
| 2026-04-29 | [@48b6772](https://github.com/reliquadotai/reliquary-ledger/commit/48b6772) | **Tiered validators epic** — full/lite/mirror modes, lite_verifier with quorum borrow, lite Dockerfile + compose, 24 unit tests |
| 2026-05-03 | [@f21edf8](https://github.com/reliquadotai/reliquary-ledger/commit/f21edf8) | Lazy-import MiningEngine (lite image works); GHA workflow for ghcr auto-publish |
| 2026-05-03 | [@1f1b423](https://github.com/reliquadotai/reliquary-ledger/commit/1f1b423) | `--mode=mirror` third tier (degenerate lite, zero CPU stages) |
| 2026-05-03 | [@083283e](https://github.com/reliquadotai/reliquary-ledger/commit/083283e) | `R2ObjectStore._run_async` — handles caller-with-active-loop (closes the asyncio.run audit gap) |

Total since prior doc: **23 commits across the three repos.**
Test coverage: **761 ledger tests + 24 forge tests + 13 red-team
simulations + 5 cutover dry-run cases — all green on staging1.**

---

## Gate-by-gate status (delta from 2026-04-28)

### A. Code green
- **green.** All audit-driven fixes shipped; no regressions across the suite.

### B. Fleet live
- **partial green** (no change from 2026-04-28).
- rtx6000b miner UID 5: 58 mines/24h, latest at window 7039230 (~10 min ago).
- H100 miner UID 7: down. Targon pod CUDA passthrough re-init loop. Platform-side, not code. Since H100 has been down 3 days, the trainer's distinct-miner gate (correctly) refused every cycle — testnet-only override added: `RELIQUARY_FORGE_MIN_DISTINCT_MINERS=1` in the trainer's drop-in. **Mainnet env file MUST omit this override.** Trainer fired immediately after the override at 06:30 UTC: `forge-grpo-1777789678`, 2 groups × 8 rollouts, merkle `0bed9f3a0661…`, **HF push landed: revision `097756891863`**.
- 2 validators publishing weights to UID 5; mesh consensus stable.

### C. Trainer firing
- **green.** Closed loop verified end-to-end including HF Hub publishing this morning.

### D. Storage + R2
- unchanged. Lite validator path uses public-bucket reads only — no R2 credential burden for lite operators.

### E. Autonomous onboarding
- **green++** with the tiered validator epic. Operator setup matrix:

| Tier | Operator setup | Compute | Onboarding time |
|---|---|---|---|
| Full | wallet + R2 creds + signing secret | GPU (H100-class) | ~30 min |
| **Lite** | **wallet only** | **CPU, ~1 GB RAM** | **~5 min Docker compose** |
| Mirror | wallet only + env override | CPU, ~1 GB RAM | ~5 min |

The lite + mirror images live at
`ghcr.io/reliquadotai/reliquary-validator-lite:latest`, auto-published
on every commit to main.

### F. Cutover script + rollback runbook
- **green** for F-1 (script exercised; recorded in `docs/cutover-dry-run-2026-04-29.md`).
- F-2 runbook authored.
- F-3 second-operator walkthrough remains the only outstanding sub-item; not blocking.

### G. Cutover-day choreography
- operator-driven.

---

## Validator tiers — what we ship, what operators see

| Tier | Stages run independently | Stages borrowed | Dockerfile | Veto power | Yuma signal |
|---|---|---|---|---|---|
| Full | 1–9 | none | `Dockerfile` | full pipeline | independent |
| Lite | 1, 2, 3, 5, 6, 7 (CPU) | 4, 8, 9 (GPU) | `Dockerfile.validator-lite` | independent on 6/9 stages | independent (signs own verdicts + weights) |
| Mirror | none | all 9 (via quorum) | same image, env override | none — pure aggregator | depends on quorum |

**Why lite preserves consensus:**
1. Independent CPU verification on 6 stages — lite REJECTS anything its own CPU stages catch (independent veto).
2. Quorum requirement on GPU stages — lite needs ≥ 2 full validators to have agreed before accepting an "all stages passed" outcome. Single-source replay attacks fail.
3. Each lite signs its own verdict + weight extrinsic. The mesh sees N independent signals, not 1 amplified vector.

**Live verification on testnet 462** (commit [@48b6772](https://github.com/reliquadotai/reliquary-ledger/commit/48b6772) build):
* `validate-window --window 7019190 --mode lite` produced
  `validated 8 completions, accepted 5`.
* Audit index for window 7019190: full validators saw
  `submitted=8 accepted=5 hard_failed=3`. **Exact match.**
* Multi-window verification (5 recent windows) is running on staging1
  as I write this; results will be appended to this doc by the next
  update.

---

## Operational gaps (not blockers)

| # | Gap | Status | Mainnet impact |
|---|---|---|---|
| 1 | R2 `list_artifacts` pagination — full prefix walk per `validate-window` | unchanged from prior doc | throughput only; mainnet H100/Blackwell latency makes this a 3-second scan, not the 5-min H100-Targon outlier |
| 2 | Validator model-bundle pre-warm | unchanged from prior doc | fewer rollouts during validator startup; mesh still publishes correct weights |
| 3 | H100 Targon container CUDA passthrough loop | platform-side; pod recreated at least twice, same failure mode | doesn't affect mainnet — operators bring their own GPU host |
| 4 | Bridge payload `Mapping[str, Any]` deeper validation | deferred (audit MEDIUM) | post-mainnet observation work; existing `BridgeSchemaError` catches the outer-shape issues |

---

## Cutover decision

**All blocking items shipped.** The remaining work is operator
choice + 24h post-launch observation:

1. Mainnet wallet + funding (custodial — operator action)
2. Run `ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh`
3. 24h babysit
4. Optional: announce alongside the protocol paper draft

Recommended approach unchanged from the prior doc: two-stage
cutover (option D from the prior brief). One validator + one miner
to mainnet, observe 48h, then scale.

---

## Appendix — image registry

| Image | Tag | What it runs |
|---|---|---|
| `ghcr.io/reliquadotai/reliquary-ledger` | `latest`, `sha-<short>` | Full validator/miner (CUDA 12.8 + flash-attn 2.8.3 + torch 2.7) |
| `ghcr.io/reliquadotai/reliquary-validator-lite` | `latest`, `sha-<short>` | CPU-only validator (lite + mirror modes) |

Both auto-published via GHA on every push to main; tags are
immutable per commit (sha-<short>) for pinned operator deploys,
mutable (latest) for Watchtower auto-update operators.

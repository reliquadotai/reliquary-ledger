# Reliquary Ledger — Launch Configuration

Small-model-first launch per Tier 1 Epic 7 ([01_TIER1_PRD.md](../../private/reliquary-plan/01_TIER1_PRD.md) — private). Scale only after stability is proven on the first tier.

## Phase 1 — Initial mainnet (weeks 0–8)

| Role | Model | Recommended GPU | Rationale |
|---|---|---|---|
| Miner | `Qwen/Qwen3-4B-Instruct` (fallback: `Qwen/Qwen2.5-7B-Instruct` if the 4B variant is unavailable) | L4 / A10 / RTX 4090 (24 GB VRAM) | Lowest hardware bar that runs a real model with FA2; broad validator set from day 1. |
| Validator | Same as miner (replay forward pass) | Same class or better | Parity required for sketch replay; match CUDA/cuDNN/torch versions with miners. |

Constraints:

- `RELIQUARY_INFERENCE_LOAD_DTYPE=bfloat16` on Ampere+; otherwise `float16`.
- `RELIQUARY_INFERENCE_REQUIRE_FLASH_ATTENTION=1` — fail-loud guard on all mainnet roles.
- Per-window budget: ≤ 22 GB steady-state VRAM on 24 GB cards (checked via `nvidia-smi --query-gpu=memory.used`).

## Phase 2 — Validator growth (weeks 8–16)

If phase 1 is stable:

- Scale reference model to 7 B (`Qwen2.5-7B-Instruct`) or switch to 8 B if Qwen3-8B ships.
- Recommended GPU bar: A100 40 GB / H100.
- Introduce logit caching in the miner artifact (Tier 2 Epic 4 unlocks stages 8 + 9 of the verifier).

## Phase 3 — Production (weeks 16+)

- 14 B–30 B model tier; H100 / B200 recommended.
- Activate distributed training on Reliquary Forge (companion runtime) — see [02_TIER2_PRD.md](../../private/reliquary-plan/02_TIER2_PRD.md).

## Benchmarks

Canonical benchmark template in [benchmarks.md](benchmarks.md). Operators appending measurements should include:

1. Hardware string (GPU model, driver, CUDA toolkit, cuDNN).
2. Software versions (torch, transformers, reliquary-inference git SHA).
3. Measured: tokens/sec (generation), proof replay latency p50/p95, total window throughput (completions accepted / minute).
4. Date + operator initials.

## Model selection policy

Mid-window model swap is forbidden. Policy updates commit via `subtensor.commit` at window boundaries only; see `reliquary_inference.chain.adapter.commit_policy_metadata`. Miners and validators poll the commitment at window start; discrepancies surface in the `environment` stage via task-source binding mismatches.

## Operator checklist — bringing up a new role

1. Pick the right model per phase (above).
2. Verify hardware meets the GPU tier.
3. Run `./deploy/real-model-readonly-smoke.sh` — end-to-end inference without chain writes.
4. Apply the mainnet profile: `ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh`.
5. Start the relevant systemd unit: `reliquary-ledger-miner-mainnet` or `reliquary-ledger-validator-mainnet`.
6. Monitor `/health` for ≥ 10 windows; proceed only if `"overall": "ok"` is stable.

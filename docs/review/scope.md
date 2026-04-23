# Review Scope

## What's in scope

### Primary artifacts

| Artifact | Repo | Path |
|---|---|---|
| Protocol paper | `reliquary-inference` | `docs/paper/reliquary_protocol_paper.md` |
| Shared crypto primitives | `reliquary-protocol` | `reliquary_protocol/` |
| 9-stage verifier pipeline | `reliquary-inference` | `reliquary_inference/validator/` |
| GRAIL sketch implementation | `reliquary-inference` | `reliquary_inference/protocol/` (proofs, signatures, tokens) |
| Mesh consensus + outlier gate | `reliquary-inference` | `reliquary_inference/validator/consensus.py` + neighbors |
| Closed-loop bridge + signer | `reliquary-protocol` | `reliquary_protocol/bridge.py` |
| Forge GRPO trainer | `reliquary` | `scripts/run_forge_grpo_live.py` + `reliquary/training/` |
| Reparam guard | `reliquary-inference` | `reliquary_inference/shared/reparam_guard.py` |
| Policy consumer | `reliquary-inference` | `reliquary_inference/shared/policy_consumer.py` |
| Eval harness + holdout | `reliquary` + `reliquary-inference` | `reliquary/eval/math_harness.py`, `reliquary_inference/dataset/math_holdout.py` |

### Specific functions + files to audit

**GRAIL proof layer**:
- `reliquary_protocol/constants.py` — `PRIME_Q`, `CHALLENGE_K`,
  `PROOF_TOPK`, `PROOF_SKETCH_TOLERANCE_*` are the arithmetic
  parameters the proof's security budget depends on. Verify the
  collision-resistance argument for the sketch under the stated
  tolerances.
- `reliquary_inference/protocol/proofs.py` — sketch construction + verification.
- `reliquary_inference/protocol/tokens.py` — RNG derivation from `window_id + public_randomness`; miner must not predict challenge positions.

**Signature primitives**:
- `reliquary_protocol/signatures.py` — `hmac_sign`, `hmac_verify`,
  `sign_canonical`, `verify_canonical`, `escape_label_value`.
- `reliquary_protocol/canonicalization.py` — `stable_json_dumps`,
  `canonical_bytes`, `sha256_json`. The byte-stability of these
  functions across Python versions + platforms is a
  cross-cutting assumption for every signed artifact.

**Mesh consensus**:
- `reliquary_inference/validator/service.py` — aggregation math;
  stake cap enforcement, median computation.
- Outlier gate math in the scorecard writer.

**Bridge integrity**:
- `reliquary_protocol/bridge.py` — `sign_envelope`, `verify_envelope`,
  `build_policy_commitment`, `BridgeVerifier`. The canonical-bytes
  pattern (`canonical_bytes_unsigned` vs `canonical_bytes`) is the
  authentication surface.
- `reliquary_inference/shared/policy_consumer.py` — poll → verify →
  smoke hash → reparam guard → hot-swap. Each guard's ordering
  matters for correctness under concurrent applies.

**GRPO correctness**:
- `scripts/run_forge_grpo_live.py` — end-to-end recipe.
  - `_compute_advantages` — group-relative normalized advantages.
  - `_rollout_loss` — PPO-clipped surrogate + KL k3.
  - `grpo_train` — optimizer loop + target-tensor freezing.
- `reliquary/training/delta_checkpoints.py` — `compute_delta`,
  merkle root construction.
- `reliquary/training/policy_attestation.py` — `compute_smoke_hash`,
  `publish_attestation`.

**Reparam defense**:
- `reliquary_inference/shared/reparam_guard.py` — finite / magnitude
  floor / per-layer scale-ratio math. Verify the bound calculations
  actually catch the documented exploit class.

**Eval harness + holdout**:
- `reliquary_protocol/eval_holdout.py` — `derive_eval_holdout_indices`.
  The derivation is deterministic; verify the seed-hashing is stable
  cross-platform + independent of label version.
- `reliquary/eval/math_harness.py` — `run_eval_cycle`,
  `update_eval_index`. Verify the published `EvalBundle` can't be
  forged without the policy authority key, and that the per-problem
  results can't be edited post-signing without invalidating the
  envelope.

## What's out of scope

- **Bittensor subtensor consensus / finality.** We depend on it; we
  don't audit it. The assumption is captured in
  [`threat-model.md`](threat-model.md).
- **Third-party Python dependencies** (torch, transformers, boto3,
  bittensor). Supply-chain concerns exist but are DCO-gated; not
  under Reliquary-level review.
- **Economic game theory of Bittensor emissions.** Reliquary inherits
  emission dynamics; tokenomics review is the subnet-owner's domain.
- **Side-channel attacks on GPU kernels** (timing, power, thermal).
  No PII is processed by the subnet.
- **Post-quantum migration.** HMAC-SHA256 is the current primitive;
  Ed25519 + larger-prime migration is a documented open question
  (protocol paper §14).
- **Legacy task sources** (`reasoning_tasks`, `dataset_prompts`).
  Retained in the codebase for tests only; not on the live path.

## Methodology suggestions

The reviewer is not bound to any methodology, but we've found these
questions most useful in our own pre-review passes:

1. **Read the protocol paper end-to-end before opening any source.**
   Ambiguities in the paper are the most likely root cause of
   critical findings — if the spec is unclear, the code can't be
   right by definition.

2. **Pick a single artifact lineage — e.g. one verdict_bundle — and
   walk it backwards.** Start from the on-chain weight set, to the
   mesh scorecard, to the verdict's 9-stage explanation struct, to
   the completion_bundle, to the task_batch, to the public randomness
   on the block. Every link should verify independently.

3. **Attack the reparam guard specifically.** It's the newest defense
   and the one most likely to have an off-by-one. Try:
   - RMSNorm × Linear scale: multiply one layer by α, divide the next
     by α. Pick α outside the `ratio_max = 1e5` bound AND inside it.
   - Additive noise below the magnitude floor.
   - Shard subsetting: ship only some target tensors, claim the
     others are pre-existing.

4. **Attack the bridge signature separately from the smoke hash.**
   The order matters: `BridgeVerifier` → smoke hash → reparam.
   What happens if an adversary can corrupt the manifest bytes but
   not the signed envelope?

5. **Run the audit harness with adversarial inputs.** We provide
   1500 adversarial trials (Tier 1 Epic 6) — the reviewer is
   welcome to add more. See [`reproducing.md`](reproducing.md).

## Deliverable

A structured PDF or markdown report with:

- Reviewer identity + commit SHAs reviewed (pinned from
  [`pinned-hashes.md`](pinned-hashes.md)).
- Executive summary (1 page).
- Per-finding: severity (critical / major / minor), title, repro
  steps, suggested mitigation, confidence level.
- Methodology notes (what attacks were attempted, what wasn't).
- A sign-off section: "I have reviewed commits X/Y/Z/W; I find
  [conclusion]."

Submit via GH issue with label `audit-feedback` on the relevant
repo; criticals contact maintainers out-of-band first per
[`response-protocol.md`](response-protocol.md).

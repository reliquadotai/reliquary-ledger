# Empirical Audit Reports

Reports produced by `reliquary_inference.audit_harness.run_audit_campaign` against the proof sketch verifier. External auditors should reproduce by running the CLI:

```bash
python -m reliquary_inference.audit_harness \
    --honest-trials 1000 \
    --adversarial-trials 500 \
    --output docs/audit/empirical_report.json
```

## How to read these reports

Each adversarial class is a deterministic tampering pattern applied to the miner's rollout hidden states before the validator replay. The report records:

| Field | Meaning |
|---|---|
| `trials` | Number of rollouts in this class. Each rollout is `CHALLENGE_K=32` positions. |
| `accept_count` | Rollouts accepted by the sketch-layer verifier. |
| `reject_count` | Rollouts rejected. |
| `false_negative_rate` | For adversarial classes: fraction of tampered rollouts the verifier wrongly accepted. Higher = worse. |
| `false_positive_rate` | For honest class: fraction of bit-exact rollouts wrongly rejected. Higher = worse. Target: 0. |
| `median_min_sketch_diff` | Median of the minimum-per-rollout sketch diff observed. Units are arbitrary scalar distances in Mersenne-prime modular space; compare against `PROOF_SKETCH_TOLERANCE_BASE = 6000`. |

## Interpreting the baseline report

The sketch layer is one of nine stages in the full verifier pipeline. At small hidden dims (e.g. `HIDDEN_DIM=256`) with random-normal inputs, per-position sketch variance is ~2000 sketch units — well inside the 6000 tolerance band calibrated for float-precision drift across GPU classes. This means:

- **Honest miners**: bit-exact sketch match → zero false positives. This is the primary guarantee.
- **Adversarial miners (sketch layer alone)**: the sketch layer has wide per-position tolerance and does NOT reliably reject random-normal tampering. Adversarial rejection is a property of the combined 9-stage pipeline — logprob replay (stage 8), distribution sampler replay (stage 9), token binding (stage 2), prompt hash binding (stage 3), and copycat directional detection — not of the sketch layer in isolation.

See [`spec-proof-protocol.md`](../../../private/reliquary-plan/notes/spec-proof-protocol.md) (private, off-repo) and `reliquary_inference/validator/validators/distribution.py` for the companion-stage design.

## Running against production hidden states

For a realistic production audit, replace the synthetic random-normal hidden states in `audit_harness.py` with live hidden states from a deployed model (Qwen3-4B or similar). Top-K values for real LLMs saturate the log-magnitude buckets, which tightens the effective discrimination. A follow-up audit class `tamper_with_live_model` is planned once we have the logit-cache artifact extension from Tier 2 Epic 4 flowing.

## Reproducibility

Reports are **bit-deterministic** across runs on the same hardware when the `randomness_hex` is fixed. Cross-GPU determinism (A100 vs H100 vs L40) is a separate measurement; the current report records the host, torch version, and CUDA availability so external auditors can reproduce the exact environment.

## Current reports

| File | Hardware | Date |
|---|---|---|
| [empirical_report.json](empirical_report.json) | NVIDIA RTX PRO 6000 Blackwell, CUDA 13.1, torch 2.11 | 2026-04-17 |

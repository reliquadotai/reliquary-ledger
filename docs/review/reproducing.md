# Reproducing the audit inputs

This document is the reviewer's "don't trust our numbers, re-run it
yourself" guide. Four independent tests:

1. **Audit harness** — honest FP rate + adversarial FN rate across a
   1000+1500 trial corpus.
2. **Cross-GPU determinism** — sketch byte-equality across 4 hosts
   on 90 samples.
3. **Mesh consensus live** — 4 validators including 1 simulated
   malicious on 256 verdicts.
4. **MATH holdout eval** — Forge's eval harness against the MATH-500
   deterministic slice; reproduce accuracy bit-for-bit on the
   reviewer's own hardware.

## 1. Audit harness

**Source code**: `reliquary_inference/audit_harness/` (to be extended
per reviewer needs).

**Tier 1 Epic 6 baseline**: 1000 honest + 1500 adversarial trials on
RTX PRO 6000 Blackwell with `HIDDEN_DIM=256`.

```bash
# From reliquary-inference@c827d5b
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ml,gpu]"

python -m reliquary_inference.audit_harness \
  --trials 1000 \
  --adversarial-trials 1500 \
  --output ./audit_out.json
```

Expected result (published in `docs/audit/empirical_report.json`):

- Honest FP rate = `0.0000`
- Adversarial FN rate = varies by tamper magnitude; all three
  adversarial classes caught by the 9-stage pipeline even when the
  sketch alone cannot distinguish.

To extend with new adversarial classes:

```bash
python -m reliquary_inference.audit_harness \
  --trials 0 \
  --adversarial-trials 100 \
  --adversarial-class custom_class \
  --output ./custom_out.json
```

The `--adversarial-class` argument dispatches into
`reliquary_inference.audit_harness.adversaries` — the reviewer is
invited to submit new adversary classes via PR for inclusion in the
next review cycle.

## 2. Cross-GPU determinism

**Source data**: `docs/audit/cross_gpu/*.json` (4 hosts × 90 samples
each) + `docs/audit/cross_gpu/comparison.json` (digest equality
check).

**Reproducing**:

```bash
# On each GPU host under review:
python -m reliquary_inference.audit_harness.cross_gpu \
  --n-samples 90 \
  --output ./cross_gpu_<hostname>.json

# Comparison (on any single host, with the N host outputs present):
python -m reliquary_inference.audit_harness.cross_gpu_compare \
  ./cross_gpu_*.json \
  --output ./comparison_reviewer.json
```

**Expected**: every sketch digest byte-identical across hosts. Our
Tier 2 Epic 6 run: 3× RTX PRO 6000 Blackwell + 1× H100, 90 samples,
0 drift.

**Sensitivity**: CUDA version, cuDNN version, PyTorch version,
flash-attn version MUST match across hosts. Mismatch produces drift;
the comparison script reports a per-position diff matrix so the
reviewer can confirm the drift is within the documented tolerance
(`PROOF_SKETCH_TOLERANCE_BASE = 6000`) or fail-loud otherwise.

## 3. Mesh consensus live

**Source data**: `docs/audit/mesh_live/*.json` (4 validators × 256
verdicts each, including one simulated-malicious validator).

**Reproducing**:

```bash
# Spin up 4 validator processes (4 local ports) against a fresh
# testnet task batch. The `mesh_live_sim` harness orchestrates this:

python -m reliquary_inference.audit_harness.mesh_live \
  --n-windows 8 \
  --n-validators 4 \
  --malicious-validator-uid 3 \
  --output ./mesh_live_reviewer.json
```

**Expected**:
- All 4 validators produce verdicts on 256 completions.
- Honest-3 mesh converges on the same accept/reject decision for
  every completion.
- Simulated-malicious validator shows 1.0 disagreement rate vs the
  honest median; gated at the outlier threshold.
- `weights_on_chain` per miner matches the honest-3 median exactly.

**What to attack**: modify `malicious-validator` behavior in the
harness to test variations:
- Selectively reject on a specific miner's completions (bias attack).
- Accept a specific forged completion (collusion attack).
- Produce verdicts indistinguishable from honest but with shifted
  correctness scores (subtle attack).

## 4. MATH holdout eval

**Source code**: `reliquary/eval/math_harness.py` + the deterministic
holdout derivation in `reliquary_protocol/eval_holdout.py`.

**Reproducing without the live signing key**:

```bash
# From reliquary@18957d0
cd reliquary
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ml,gpu]"
uv pip install -e ../reliquary-inference  # soft dep for holdout loader
uv pip install -e ../reliquary-protocol

python scripts/run_forge_eval.py \
  --model-ref Qwen/Qwen2.5-3B-Instruct \
  --max-problems 500 \
  --sampling-mode greedy \
  --no-publish \
  --signing-secret-path /dev/null  # --no-publish skips the secret read
```

Note: `--no-publish` mode requires minor edit to the script to skip
the secret-path check entirely; alternatively set the env
`FORGE_EVAL_NO_PUBLISH=1` and run from a stub signing secret.

**Expected**: greedy pass@1 accuracy on MATH-500 for Qwen2.5-3B-Instruct
lands in a narrow band per the hardware + cuDNN version. The
reviewer's number should match any live `EvalBundle` published under
the same policy_artifact_id to within `PROOF_SKETCH_TOLERANCE_BASE`
(tolerance is for sketch drift, not for accuracy per se — but if
the model weights are bit-exact and sampling is greedy, the accuracy
is identical).

**Verifying a published EvalBundle**:

```bash
# Pull a live bundle from the public R2 bucket:
curl -s "https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/eval_bundles/462/index.json" \
  | jq '.entries[0]'
# → { "eval_run_id": "forge-eval-...", "eval_window_id": ..., "storage_key": "...", ... }

# Fetch the full bundle:
curl -s "https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/<storage_key>" \
  | jq '.'

# Independently verify the envelope signature:
python -c "
import json
from reliquary_protocol import envelope_from_dict, HmacBridgeVerifier, verify_envelope
bundle = json.loads(open('/path/to/fetched_bundle.json').read())
env = envelope_from_dict(bundle)
# Reviewer supplies the policy-authority secret out-of-band for this step.
verifier = HmacBridgeVerifier(secrets={env.signer_id: '<SECRET>'})
print('valid:', verify_envelope(env, verifier))
"
```

## Shared data + artifacts

Every published artifact is accessible via the public R2 bucket
(read-only, unauthenticated):

- `https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.json` — rolling public audit index
- `https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html` — human-readable view
- `https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/eval_bundles/462/index.json` — rolling eval index (populates once the eval timer fires on the fleet)
- `https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/rollouts/462/…` — rollout bundles
- `https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/attestations/462/…` — signed checkpoint attestations

## What constitutes "reproducible"

A reviewer's report is considered reproducible when:

1. They pin the 4 commits in [`pinned-hashes.md`](pinned-hashes.md).
2. They re-run at least one of the above four tests on their own
   hardware.
3. They publish the raw output artifacts alongside the report so a
   third party can verify their claims independently.

Numbers without reproducible source data are still useful as a
signal but not as a gate.

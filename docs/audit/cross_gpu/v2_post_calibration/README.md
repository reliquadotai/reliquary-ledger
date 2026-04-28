# Cross-GPU audit — v2 post-calibration

Re-run of the cross-GPU sketch determinism audit after tightening
`PROOF_SKETCH_TOLERANCE_BASE 6000 → 1000` and
`LOGPROB_DRIFT_THRESHOLD 0.15 → 0.01` (commits
[reliquary-ledger@9671f9a](https://github.com/reliquadotai/reliquary-ledger/commit/9671f9a),
[reliquary-protocol@a24d841](https://github.com/reliquadotai/reliquary-protocol/commit/a24d841)).

| Host       | Hardware                          | Pod ID                          | Samples digest |
| ---------- | --------------------------------- | ------------------------------- | -------------- |
| staging1   | NVIDIA RTX 6000B Blackwell, 96GB  | `wrk-j8nq3c7xn81v-78f5546865`   | `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |
| staging2   | NVIDIA RTX 6000B Blackwell, 96GB  | `wrk-nyfmqa78r5ld-994db9b`      | `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |
| rtx6000b   | NVIDIA RTX 6000B Blackwell, 96GB  | `wrk-mnmhkheic6r7-5b7f8f5b5b`   | `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |

**Bit-exact agreement across all 3 RTX 6000B Blackwell hosts.** Identical
to the pre-calibration digest captured in
[../comparison.json](../comparison.json), confirming the constants
tightening only narrowed the *acceptance envelope* and did not perturb
the underlying sketch math.

H100 cross-class verification is pending: staging3h100 (1× H100) has
the host driver but the GPU is not exposed in the per-tenant container
(no `/dev/nvidia*`, no nvidia-container-runtime). Targon platform-side
GPU passthrough setup needed before re-running the audit on H100.

## Reproducing

```bash
ssh <host>
/opt/reliquary-venv/bin/python3 -m reliquary_inference.cross_gpu_audit \
    --output /tmp/cgpu_v2.json
python3 -c "import json; print(json.load(open('/tmp/cgpu_v2.json'))['samples_digest'])"
```

The harness produces 90 deterministic samples (10 seeds × 3 magnitude
scales × 3 hidden dims). Any new GPU class added to the fleet should
re-run this and append the digest to the table above; identical
digests confirm cross-class determinism.

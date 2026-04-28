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

H100 cross-class verification: a re-provisioned H100 instance
(`wrk-3l9wh8oc5c0w-57f5cd4789-bx5hc`) has the GPU correctly exposed
(`/dev/nvidia0` present, `nvidia-smi` reports `NVIDIA H100 80GB HBM3`,
driver `570.195.03`). The original `wrk-1yd80wogo5xd-689dbf778c-scqln`
pod hit a Targon-platform-side device-plugin failure
(`UnexpectedAdmissionError — Allocate failed due to cannot allocate
unregistered device nvidia.com/gpu`) and was retired.

After bootstrapping the new pod (Python 3.12 venv, torch 2.7.0+cu128,
reliquary-{inference,protocol,forge} editable installs), running the
same harness on H100 closes the cross-class verification gap. The
report will be appended to this README upon completion.

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

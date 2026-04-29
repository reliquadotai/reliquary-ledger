# Cross-GPU audit — v2 post-calibration

Re-run of the cross-GPU sketch determinism audit after tightening
`PROOF_SKETCH_TOLERANCE_BASE 6000 → 1000` and
`LOGPROB_DRIFT_THRESHOLD 0.15 → 0.01` (commits
[reliquary-ledger@9671f9a](https://github.com/reliquadotai/reliquary-ledger/commit/9671f9a),
[reliquary-protocol@a24d841](https://github.com/reliquadotai/reliquary-protocol/commit/a24d841)).

| Host             | Hardware                                     | Architecture | Samples digest |
| ---------------- | -------------------------------------------- | ------------ | -------------- |
| staging1         | NVIDIA RTX 6000B Blackwell, 96GB             | sm_120       | `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |
| staging2         | NVIDIA RTX 6000B Blackwell, 96GB             | sm_120       | `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |
| rtx6000b         | NVIDIA RTX 6000B Blackwell, 96GB             | sm_120       | `4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c` |
| **staging3h100** | **NVIDIA H100 80GB HBM3, driver 570.195.03** | **sm_90**    | **`4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c`** |

**Bit-exact agreement across two distinct GPU architectures** —
Blackwell (sm_120) and Hopper (sm_90). All four hosts, spanning two
GPU generations and two CUDA library streams (cu130 on Blackwell,
cu128 on Hopper), produce the **identical** 90-sample SHA-256
digest under the harvest + tightened-constants code (ledger
[`8a69577`](https://github.com/reliquadotai/reliquary-ledger/commit/8a69577),
protocol [`a24d841`](https://github.com/reliquadotai/reliquary-protocol/commit/a24d841)).

This is the headline empirical claim for the proof primitive: the
GRAIL sketch is bit-exact-deterministic across hardware classes,
not merely within a class. The ~10⁻¹⁶⁷ forgery probability bound
on the sketch path is predicated on this property, and it is now
confirmed in production with two architectures. Operationally,
this means `PROOF_SKETCH_TOLERANCE_BASE = 1000` has substantial
headroom even across hardware-class boundaries.

**Platform note.** The original `staging3h100` pod hit a
Targon-platform-side device-plugin failure (`UnexpectedAdmissionError
— Allocate failed due to cannot allocate unregistered device
nvidia.com/gpu`) and was retired 2026-04-26. A replacement pod was
provisioned and the audit ran without further incident on a fresh
Python 3.12 venv + torch 2.7.0+cu128 +
reliquary-{inference,protocol,forge} editable installs.

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

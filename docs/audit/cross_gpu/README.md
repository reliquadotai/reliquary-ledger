# Cross-GPU Determinism Reports

Reports produced by `reliquary_inference.cross_gpu_audit.run_cross_gpu_campaign` on different GPU hosts. Running the harness on multiple GPU classes and diffing the outputs empirically answers the spec question:

> Do independent GPU instances agree bit-exactly on sketch values under a fixed deterministic sample sweep?

## Harness

```bash
python -m reliquary_inference.cross_gpu_audit --output <path>.json
```

Default sample sweep: 10 seeds × 3 magnitude scales × 3 hidden dims = **90 samples** per host. Samples are seeded deterministically via `torch.Generator().manual_seed(...)` and generated on the active torch device.

Each report carries:
- host, torch version, CUDA availability
- GPU name, total memory, CUDA compute capability
- randomness hex used for `r_vec` derivation
- per-sample `(seed, scale, hidden_dim, sketch)`
- **`samples_digest`** — SHA-256 over the canonicalized sample sequence; a single 64-char fingerprint that summarizes the whole sweep.

Two hosts are bit-exactly deterministic iff their `samples_digest` values match.

## Comparison

```python
from reliquary_inference.cross_gpu_audit import compare_reports
compare_reports([report_a, report_b, ...])
# { matching_digests, digest_by_host, mismatch_count, mismatches, ... }
```

## Committed reports

| File | GPU | CUDA capability | Driver / CUDA / torch | Date |
|---|---|---|---|---|
| [devserver_blackwell.json](devserver_blackwell.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB) | (12, 0) | driver 590.48 / CUDA 13.1 / torch 2.11.0+cu130 | 2026-04-17 |
| [staging1_blackwell.json](staging1_blackwell.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB) | (12, 0) | driver 590.48 / CUDA 13.1 / torch 2.11.0+cu130 | 2026-04-17 |
| [staging2_blackwell.json](staging2_blackwell.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB) | (12, 0) | driver 590.48 / CUDA 13.1 / torch 2.11.0+cu130 | 2026-04-17 |
| [staging3_h100.json](staging3_h100.json) | **NVIDIA H100 80GB HBM3** | **(9, 0)** | **driver 570.19 / CUDA 12.8 / torch 2.5.1+cu124** | 2026-04-17 |
| [comparison.json](comparison.json) | pairwise diff across all 4 hosts | — | — | 2026-04-17 |

## Headline result (2026-04-17)

**Four independent hosts across two GPU architectures, two CUDA versions, two torch versions → bit-exact agreement across all 90 samples.** Shared digest:

```
4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c
```

`matching_digests = True`. `mismatch_count = 0`.

This is a **stronger-than-expected cross-class determinism result**. The protocol design budgeted `PROOF_SKETCH_TOLERANCE_BASE = 6000` sketch units to absorb cross-GPU floating-point drift; the measurement shows **zero observed drift** across the spanned environments. The tolerance envelope is entirely unused by honest miners under the tested configurations.

## What this spans

- **GPU architectures**: Blackwell (sm_120) × 3 + Hopper (sm_90) × 1.
- **CUDA driver versions**: 570.19 and 590.48.
- **CUDA toolkit versions**: 12.4 and 13.0.
- **torch wheels**: `2.5.1+cu124` and `2.11.0+cu130`.
- **Python envs**: one Miniforge-conda env (`/opt/miniforge/envs/reliquary-inference` on the dev host) and three `python3 -m venv` envs (fresh containers).
- **Host containers**: four completely separate Kubernetes pods.

## Implications for the spec

1. `PROOF_SKETCH_TOLERANCE_BASE = 6000` has **substantial headroom** for honest cross-GPU drift. Future tightening is an option — but only after measuring on additional classes (A100, L40, possibly consumer cards) to confirm the "zero drift" hypothesis generalizes.
2. Validators can safely enforce bit-exact sketch equality in **production honest-path verdict artifacts** as long as the validator and miner run on GPUs in the tested class set. The tolerance envelope is the fallback, not the primary detection mechanism.
3. `adaptive_sketch_tolerance(pos) = base + growth * sqrt(pos)` is primarily protecting against **long-sequence attention drift**, not against cross-GPU numerical differences, based on this evidence.

## Extending the fleet

To add more GPU classes to the committed baseline:

```bash
# On the new host:
python -m reliquary_inference.cross_gpu_audit --output /tmp/cross_gpu_<NICKNAME>.json

# Fetch + compare:
scp <host>:/tmp/cross_gpu_<NICKNAME>.json docs/audit/cross_gpu/<NICKNAME>.json
python - <<'PY'
# ... load reports, compare_reports([...]), commit comparison.json
PY
```

No code changes needed to the harness itself.

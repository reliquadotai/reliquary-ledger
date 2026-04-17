# Cross-GPU Determinism Reports

Reports produced by `reliquary_inference.cross_gpu_audit.run_cross_gpu_campaign` on different GPU hosts. Running the harness on multiple GPU classes and diffing the outputs answers the spec question:

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
compare_reports([report_a, report_b])
# { matching_digests, digest_by_host, mismatch_count, mismatches, ... }
```

## Committed reports

| File | GPU | CUDA / driver | Date |
|---|---|---|---|
| [devserver_blackwell.json](devserver_blackwell.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB) | CUDA 13.1 / driver 590.48 / torch 2.11.0+cu130 | 2026-04-17 |
| [staging1_blackwell.json](staging1_blackwell.json) | NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB) | CUDA 13.1 / torch 2.11.0+cu130 | 2026-04-17 |
| [comparison.json](comparison.json) | pairwise diff devserver × staging1 | — | 2026-04-17 |

## Headline result (2026-04-17)

Two independent RTX PRO 6000 Blackwell hosts, different containers, different Python environments (one conda-based, one fresh `python3 -m venv`), identical torch / CUDA versions, **bit-exact agreement across all 90 samples**. Samples digest on both hosts:

```
4de1918431de9f268efacfcb298e52cbe80a4adb6388af3b183753e7e960572c
```

`matching_digests = True`. `mismatch_count = 0`. No sketch drift observed within-class.

## Open: cross-class determinism (Blackwell vs H100)

A staging H100 host was provisioned 2026-04-17 but hadn't completed SSH key authorization at audit time. When it comes online, the same harness should run there and be diffed against the committed Blackwell baseline. Documented as a follow-up under Tier 2 Epic 6 of the private plan.

If cross-class sketches diverge: the `adaptive_sketch_tolerance(pos) = base + growth * sqrt(pos)` envelope absorbs the variance; the measurement tells external auditors how much slack the tolerance is actually using in production.

If cross-class sketches agree bit-exactly: the protocol's cross-hardware portability is stronger than the tolerance suggests, and future tightening of `PROOF_SKETCH_TOLERANCE_BASE` could be considered.

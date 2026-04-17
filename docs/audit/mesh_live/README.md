# Live Mesh-Integration Reports

End-to-end runs of the mesh integration harness across independent GPU nodes, documenting that the stake-weighted aggregator works correctly when the verdict inputs come from real, physically separate validator processes on real hardware.

## Harness

Per-validator producer:

```bash
python -m reliquary_inference.validator.mesh_integration produce \
    --validator-hotkey mesh-A --validator-stake 40.0 \
    --scenario honest --window-id 12345 --count 32 \
    --output /tmp/mesh_verdicts_A.json
```

Aggregator (orchestrator collects the per-node verdict files and runs):

```bash
python -m reliquary_inference.validator.mesh_integration aggregate \
    --input /tmp/mesh_verdicts_A.json /tmp/mesh_verdicts_B.json /tmp/mesh_verdicts_M.json \
    --expected-hotkeys mesh-A=40.0 mesh-B=40.0 mesh-M=10.0 \
    --output /tmp/mesh_report.json
```

## Committed run (2026-04-17)

Three validators, window `12345`, 32 completions:

| Validator | GPU | Stake | Scenario |
|---|---|---|---|
| mesh-A | NVIDIA RTX PRO 6000 Blackwell Server Edition (dev server) | 40.0 | honest |
| mesh-B | NVIDIA RTX PRO 6000 Blackwell Server Edition (staging1) | 40.0 | honest |
| mesh-M | simulated-only (no GPU, constructed to reject all and emit far-from-median scores) | 10.0 | malicious |

Aggregation result (see `mesh_report.json`):

- **32 / 32 completions accepted** (honest majority wins).
- **Missing validators**: none.
- **Outlier detection**: `mesh-M` flagged as outlier on every completion; disagreement rate 1.0.
- **Gated validators**: `["mesh-M"]` (rate 1.0 > `MESH_OUTLIER_RATE_GATE = 0.05`).
- `mesh-A` and `mesh-B` disagreement rate: 0.0 — the two Blackwell hosts produced identical honest verdicts, confirming cross-container determinism at the aggregator layer (same signal as the cross-GPU audit under `docs/audit/cross_gpu/`).

## Files in this directory

| File | Role |
|---|---|
| [verdicts_devserver_blackwell.json](verdicts_devserver_blackwell.json) | raw verdicts produced by validator A (dev server Blackwell) |
| [verdicts_staging1_blackwell.json](verdicts_staging1_blackwell.json) | raw verdicts produced by validator B (staging1 Blackwell) |
| [verdicts_simulated_malicious.json](verdicts_simulated_malicious.json) | synthetic malicious validator verdicts (no GPU, locally produced) |
| [mesh_report.json](mesh_report.json) | full aggregation output (median verdicts + disagreement rates + gated validators) |

## What this demonstrates

- Verdicts are emitted as portable structured JSON; validators do not need to coordinate on schema beyond the shared `VerdictArtifact` dataclass.
- Aggregation is stake-weighted and insensitive to validator ordering — two honest 40-stake validators override one malicious 10-stake validator even when both sides emit 32 verdicts.
- Outlier detection fires correctly: `mesh-M` diverges in both per-completion accept/reject AND per-key score space, so its disagreement rate crosses `MESH_OUTLIER_RATE_GATE` in a single window.

## Deferred

- **Signed verdicts**: current `signature` field is empty; production wiring uses `protocol/signatures.verify_commit_signature` on a Bittensor hotkey. Integration pending the T2 E7 subnet-ownership claim (which brings real hotkeys with it).
- **R2 storage**: current harness reads/writes local filesystem paths. Production swaps to R2 via `storage/registry.py`; out of scope for the pure-aggregator epic.
- **Multi-window rolling state**: mesh aggregator is stateless per-window; persistent cross-window history (mirror of CopycatHistory) follows once validator service wiring is refactored.

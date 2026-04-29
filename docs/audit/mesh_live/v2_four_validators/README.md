# Mesh-Live v2 — 4-Validator Run (3 Honest Blackwell + 1 Malicious)

Second-generation mesh-live integration run, extending the initial 3-validator audit with a third real Blackwell node (staging2) and a larger per-window completion count (64 vs the v1 run's 32).

## Configuration

| Validator | Host | GPU | Stake | Scenario |
|---|---|---|---|---|
| mesh-A | dev server (rtx6000b alias) | RTX PRO 6000 Blackwell | 40.0 | honest |
| mesh-B | staging1                    | RTX PRO 6000 Blackwell | 35.0 | honest |
| mesh-C | staging2                    | RTX PRO 6000 Blackwell | 15.0 | honest |
| mesh-M | local (simulated) | — | 10.0 | malicious (rejects + emits far-from-median scores) |

Window id: `88888`. Each validator produces 64 completion verdicts → 256 total verdicts across 4 validators.

## Result ([mesh_report.json](mesh_report.json))

- **completions accepted**: 64 / 64
- **missing_validators**: []
- **gated_validators**: `["mesh-M"]`
- **disagreement_rates**: `{mesh-A: 0.0, mesh-B: 0.0, mesh-C: 0.0, mesh-M: 1.0}`

Three honest Blackwell validators — across three physically separate containers, three independent Python environments (one Miniforge-conda, two fresh `python3 -m venv`) — produce bit-identical honest verdicts (disagreement 0.0 across A/B/C). The stake-weighted median of their acceptance vote overrides the minority-stake malicious validator on every completion. `mesh-M`'s disagreement rate of 1.0 crosses `MESH_OUTLIER_RATE_GATE = 0.05` and triggers gating in a single window.

## What this strengthens vs v1

| Dimension | v1 | v2 |
|---|---|---|
| Honest validators | 2 | **3** (adds staging2) |
| Total real Blackwell hosts | 2 | **3** |
| Completions per validator | 32 | **64** |
| Total aggregated verdicts | 96 | **256** |
| Stake distribution | 40 / 40 / 10 | 40 / 35 / 15 / 10 (more realistic spread) |

The stake spread matters: v1 had two equal 40-stake honest validators plus a 10-stake malicious; v2 has three honest validators at asymmetric stakes (40, 35, 15) testing that the stake-weighted median still converges cleanly when honest stake isn't uniform.

## Artifacts

| File | Role |
|---|---|
| [verdicts_devserver_blackwell.json](verdicts_devserver_blackwell.json) | raw verdicts produced by `mesh-A` (dev server Blackwell) |
| [verdicts_staging1_blackwell.json](verdicts_staging1_blackwell.json) | raw verdicts produced by `mesh-B` (staging1 Blackwell) |
| [verdicts_staging2_blackwell.json](verdicts_staging2_blackwell.json) | raw verdicts produced by `mesh-C` (staging2 Blackwell) |
| [verdicts_simulated_malicious.json](verdicts_simulated_malicious.json) | synthetic malicious validator verdicts (no GPU) |
| [mesh_report.json](mesh_report.json) | aggregation output (median verdicts + disagreement rates + gated validators) |

## Reproducing

```bash
# on each validator host:
python -m reliquary_inference.validator.mesh_integration produce \
    --validator-hotkey mesh-X --validator-stake 40.0 \
    --scenario honest --window-id 88888 --count 64 \
    --output /tmp/mesh_verdicts_X.json

# orchestrator collects all files then:
python -m reliquary_inference.validator.mesh_integration aggregate \
    --input /tmp/*.json \
    --expected-hotkeys mesh-A=40.0 mesh-B=35.0 mesh-C=15.0 mesh-M=10.0 \
    --output mesh_report.json
```

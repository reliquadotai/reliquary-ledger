# Contributors

Reliquary Ledger is the work of the team maintaining the
[reliquadotai](https://github.com/reliquadotai) organization across three
companion repos:

- [reliquary-ledger](https://github.com/reliquadotai/reliquary-ledger) — inference + verification runtime (this repo)
- [reliquary-forge](https://github.com/reliquadotai/reliquary-forge) — training runtime (GRPO + distillation)
- [reliquary-protocol](https://github.com/reliquadotai/reliquary-protocol) — shared protocol package

## Maintainers

- [@0xgrizz](https://github.com/0xgrizz)
- [@0xgrizz](https://github.com/0xgrizz) (formerly active under another GitHub alias on early commits)

## Acknowledgments

### Upstream protocol heritage

The GRAIL hidden-state sketch primitive originates from the upstream
[`grail`](https://github.com/grail-the-game/grail) project. Reliquary Ledger
ports the proof primitive (`PRIME_Q`, `CHALLENGE_K=32`, `PROOF_TOPK=16`,
`PROOF_NUM_BUCKETS=8`, log-magnitude bucketing, sqrt-growth tolerance) from
that work. The verifier pipeline, mesh consensus, copycat detection,
distillation lane, env registry, and closed-loop bridge are independent
Reliquary additions.

### Parallel work — romain13190

[romain13190](https://github.com/romain13190) has been independently
iterating on a parallel SN81 implementation at
[romain13190/reliquary](https://github.com/romain13190/reliquary). That
codebase began on 2026-04-17 from its own planning documents, and forensic
review confirmed neither codebase imported from the other (both port from
upstream `grail`). The romain13190 repo led on operational polish that we
have since adopted, with explicit per-commit attribution in our git
history. Specific ideas harvested:

- σ ≥ 0.43 zone filter + 50-window prompt cooldown framing
  (his [8fb89ba](https://github.com/romain13190/reliquary/commit/8fb89ba),
  2026-04-21; ours [8e92521](https://github.com/reliquadotai/reliquary-ledger/commit/8e92521))
- Cheater-curve threshold calibration methodology
  (his [5a7b3eb](https://github.com/romain13190/reliquary/commit/5a7b3eb))
- Sketch-drift measurement tooling
  (his [013763e](https://github.com/romain13190/reliquary/commit/013763e))
- Event-driven window-seal pattern
  (his [1790e25](https://github.com/romain13190/reliquary/commit/1790e25))
- Continuous HF Hub checkpoint publishing every 10 windows
  (his [0b9795d](https://github.com/romain13190/reliquary/commit/0b9795d))
- CUDA 12.8 + flash-attn-2 Docker image
  (his [6ee8592](https://github.com/romain13190/reliquary/commit/6ee8592))
- Watchtower-friendly GHCR build pipeline
  (his [b78ef76](https://github.com/romain13190/reliquary/commit/b78ef76))
- `--resume-from sha:<hex> | path:<dir>` operator flag
  (his [1801544](https://github.com/romain13190/reliquary/commit/1801544),
  [e5f81ad](https://github.com/romain13190/reliquary/commit/e5f81ad))
- Validator weight-only mode (no GPU, R2-driven)
  (his [aca8661](https://github.com/romain13190/reliquary/commit/aca8661))
- W&B richer training-config logging
  (his [b65085b](https://github.com/romain13190/reliquary/commit/b65085b))

Where we tightened thresholds, the values are calibrated independently on
our cross-GPU staging fleet rather than copied directly.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development workflow and
[`SECURITY.md`](SECURITY.md) for vulnerability reporting.

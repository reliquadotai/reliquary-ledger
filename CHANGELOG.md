# Changelog

All notable changes to **Reliquary Ledger** (`reliquary-inference`) are
documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
we version jointly with [reliquary-protocol](https://github.com/0xgrizz/reliquary-protocol)
so both runtimes bump in lockstep.

## [Unreleased]

### Added
- DAPO zone filter module (`validator/zone_filter.py`) with `rewards_std`,
  `is_in_zone`, `filter_groups`, `zone_summary`. σ ≥ 0.43 steady / 0.33
  bootstrap threshold. Environment-hard-fail rollouts count toward σ
  (they're protocol-valid, just reward = 0) while protocol-invalid
  rollouts (signature/proof/tokens) are excluded.
- Per-prompt `CooldownMap` (`validator/cooldown.py`) with 50-window
  horizon, atomic save/load, R2 mirror + auto-restore on node rebuild.
- `MathTasksSource` (`dataset/task_sources/__init__.py`) +
  `MATHEnvironment` (`dataset/task_sources/math_env.py`): Hendrycks
  MATH (12 500 problems), balanced-brace `\boxed{…}` extractor,
  conservative LaTeX normalization, `evaluate_math_trace`. Bootstrap
  filter `RELIQUARY_INFERENCE_MATH_MAX_LEVEL` restricts the sampling
  pool to easier difficulty levels.
- Batched M-rollout miner generate (`MiningEngine.generate_m_completions`).
  One `model.generate()` call for all M samples per prompt with per-batch
  RNG seed; per-row EOS truncation prevents HF pad-token from leaking
  into downstream GRAIL verification. Expected speedup on H100: 5-7×.
- Batched validator proof verification (`validator/batched_verify.py`).
  Groups completions by `(miner, window)`, pads + runs one forward pass,
  caches per-completion hidden states so ProofStage skips its own
  forward. Feature-flagged via `RELIQUARY_INFERENCE_BATCHED_VERIFY`.
- Reparam-trick sanity guard (`shared/reparam_guard.py`) wired into
  `PolicyConsumer` pre-apply path. Finite / magnitude floor / per-layer
  scale-ratio checks reject deltas with RMSNorm × Linear rescaling
  artifacts before they mutate the cached model.
- `/healthz`, `/status`, `/dashboard` HTTP endpoints on the metrics
  exporter. CORS enabled. Static single-page HTML dashboard at
  `dashboard/index.html` that polls `/status` + `/healthz` every 10 s
  and renders chain state + zone filter + per-task-source rollups.
- `reliquary-audit-index.service` + `reliquary-audit-index.timer` —
  systemd timer rebuilds `{export_dir}/audit/index.json` every 10 min
  so rolling gauges stay fresh.
- 7 zone filter Prometheus gauges (`reliquary_rolling_zone_*`).
- Closed-loop bridge integration: `RolloutBundlePublisher` / `Fetcher`
  exchange training data with Forge; `policy_consumer` applies signed
  deltas at `effective_at_ledger_window` with reparam + smoke-hash
  + signature gates.
- Config knobs: `RELIQUARY_INFERENCE_ZONE_FILTER_BOOTSTRAP`,
  `RELIQUARY_INFERENCE_COOLDOWN_WINDOWS`, `RELIQUARY_INFERENCE_COOLDOWN_R2_KEY`,
  `RELIQUARY_INFERENCE_MATH_MAX_LEVEL`, `RELIQUARY_INFERENCE_GENERATION_TEMPERATURE`,
  `RELIQUARY_INFERENCE_GENERATION_TOP_P`, `RELIQUARY_INFERENCE_VALIDATOR_BACKFILL_HORIZON_WINDOWS`,
  `RELIQUARY_INFERENCE_BATCHED_VERIFY`, `RELIQUARY_INFERENCE_BATCHED_VERIFY_MAX_SIZE`,
  `RELIQUARY_INFERENCE_REPARAM_PROJ_MIN_MEAN_ABS`,
  `RELIQUARY_INFERENCE_REPARAM_LAYER_SCALE_RATIO_MAX`.
- New docs: `miner-quickstart.md`, `validator-quickstart.md`,
  `incentive.md`, `dashboard.md`.
- Validator backfill horizon (default 10 windows) — no more orphaned
  windows after a validator restart.
- Verdict payload now carries `miner_id` + `sample_index` so the forge
  GRPO trainer can join verdicts → completions correctly.
- Duplicate-submission check keys on `(task_id, sample_index)` instead of
  `task_id` alone — unblocks M > 1 rollouts per prompt.

### Changed
- Miner generation now defaults to T = 0.9, top_p = 1.0 when
  `samples_per_task > 1` so the M rollouts in a GRPO group aren't
  identical. Greedy decode remains the default for single-sample paths.
- `_latest_window_manifests` fetches only the top-N manifests by upload
  time instead of the full O(N) body scan. Audit-index rebuild dropped
  from 10-15 min to ~60 s on the live R2 corpus.
- `collect_metrics_snapshot` integrates zone filter totals into the
  rolling snapshot so Prometheus / Grafana surface training-signal
  density in real time.
- `service.validate_window` pre-computes batched hidden states before
  running the per-completion pipeline so ProofStage doesn't pay for
  M serial forward passes.

### Fixed
- Correctness source in `forge` (shipped in
  [reliquary](https://github.com/0xgrizz/reliquary) `run_forge_grpo_live.py`):
  read per-rollout reward from `verdict_bundles` instead of
  `completion_bundles` (miners don't compute their own correctness).
- `reliquary-protocol.storage.R2ObjectBackend._call` now retries 429 /
  5xx / socket-timeout with exp backoff (5 s → 10 s → 20 s) before
  surfacing. Default timeout raised 30 s → 120 s.
- `list_detailed` reads CF's `last_modified` field (previously misread
  `uploaded`, causing every sort-by-time call to fall back to O(N)
  body fetch).
- Audit-index only fetches the `limit` scorecards referenced by the
  selected manifests instead of every historical scorecard.

### Deprecated
- `reasoning_tasks` and `dataset_prompts` task sources are kept in the
  codebase for tests and low-resource fallbacks; the live target is
  `math`. Deprecation warning surfaces when the environment selects
  either of the legacy sources with no test marker.

### Security
- Reparam-trick sanity guard wired into `PolicyConsumer.poll_once` to
  block deltas that pass the signature + smoke-hash gates but carry
  RMSNorm × Linear rescaling exploits.
- R2 retry policy surfaces 4xx-other-than-429 immediately — those are
  caller-contract errors, not load-transient, and retrying would hide
  real bugs.

## [0.1.0] - 2026-04-18

Initial public release. See the [protocol paper](docs/paper/reliquary_protocol_paper.md)
for the v0.1 design spec. Core inference runtime: miner engine +
validator 9-stage pipeline + GRAIL sketch proofs + mesh consensus +
R2-backed artifact storage + public audit index.

# Public Status

This page is the sanitized public status surface for Reliquary Ledger.

## Current Snapshot

- Network: `test`
- Netuid: **462** ("Reliquary")
- Status: live testnet mesh
- Commodity: proof-carrying completions
- **Active task source**: `math` (Hendrycks MATH, 12 500 problems). Bootstrap
  filter `RELIQUARY_INFERENCE_MATH_MAX_LEVEL=2` restricts to Level 1-2 while
  the base model is learning. Legacy sources (`reasoning_tasks`,
  `dataset_prompts`) are kept in the codebase for tests + fallbacks only.
- **Current real-model baseline**: `Qwen/Qwen2.5-3B-Instruct`
- **GPU profile**: `cuda` + `bf16` (RTX 6000 Blackwell reference)
- **Storage backend**: R2 REST (`r2_rest`) with retry + exp backoff
- **Mesh size**: 4 validators (devserver + 3 Targon staging nodes)
- **Proof system**: GRAIL sketch commitments v5, HMAC signatures
- **DAPO zone filter**: bootstrap threshold σ ≥ 0.33
- **Closed-loop bridge**: `forge-grpo-*` policy deltas applied by miner +
  validators via `policy_consumer` at `effective_at_ledger_window`

## Live Endpoints

- **Dashboard**: `http://<validator>:9108/`
- **Health**: `http://<validator>:9108/healthz` (JSON)
- **Operator status**: `http://<validator>:9108/status` (JSON, superset of healthz)
- **Prometheus metrics**: `http://<validator>:9108/metrics`
- **Public audit (R2 CDN)**:
  - [HTML index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html)
  - [JSON index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.json)

All endpoints emit `Access-Control-Allow-Origin: *` so third-party dashboards
and auditors can scrape without operator cooperation.

## What Is Confirmed Working

- M=8 sampled rollouts per prompt in a single `model.generate()` call
  (5-7× faster than serial sampling; verified on Blackwell)
- 4-validator mesh consensus with verdict aggregation
- Zone filter correctly identifies in-zone groups (σ ≥ 0.33) over real
  mining data — observed 60% in-zone rate on live Level-1/2 MATH
- Forge GRPO trainer pulls zone-filtered groups, runs real optimizer
  steps (non-zero gradient), publishes a `forge-grpo-*` delta through
  the closed-loop bridge
- Miner + validator `policy_consumer` hot-swaps the delta at
  `effective_at_ledger_window` without service restart
- Audit index rebuild in ~60 s via key-sorted R2 listing (was 10-15 min)
- R2 retry + exp backoff absorbs rate-limit bursts without losing a cycle
- Reparam-trick sanity guard rejects deltas with NaN / dead projections /
  out-of-band layer-scale ratios before they touch the cached model

## Operational Posture

- inference-only runtime on this repo (training lives in Forge)
- proof-first and audit-first
- private artifacts with a public audit surface
- private monitoring surfaces through SSH tunneling
- finalized reasoning windows exported with additive semantic verdict
  fields for downstream training import
- explicit websocket endpoint support
- testnet-first deployment

## Known Limitations

- public testnet websocket endpoints can still rate-limit under load;
  a dedicated endpoint is recommended for sustained operations
- the current live public audit is intentionally lightweight, not a
  full explorer
- a fully separate public audit bucket or custom domain still depends
  on operator-managed bucket permissions
- alert rules are provisioned, but notification routing is intentionally
  left to the operator
- the repo is testnet-first today, not presented as a finished mainnet
  deployment; mainnet cutover requires ≥ 4 weeks of continuous-operation
  track record plus external cryptographer review of the protocol paper

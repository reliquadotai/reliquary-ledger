# Public Status

This page is the sanitized public status surface for Reliquary.

## Current Snapshot

- Network: `test`
- Status: live testnet staging
- Commodity: proof-carrying completions
- Active public task-source baseline: `reasoning_tasks`
- Current real-model staging baseline: `Qwen/Qwen2.5-1.5B-Instruct`
- Current GPU profile: `cuda` + `bf16`
- Current storage baseline: private local artifacts with a public audit target
- Public audit:
  - [HTML index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html)
  - [JSON index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.json)

## What Is Confirmed Working

- local demo roundtrip for both task sources
- readonly testnet smoke path
- real-model readonly smoke on GPU
- live miner and validator services on testnet
- on-chain weight publication
- audit index generation and publishing
- localhost-only Prometheus and Grafana monitoring with exporter-backed subnet metrics

## Operational Posture

- inference-only runtime
- proof-first and audit-first
- private artifacts with a public audit surface
- private monitoring surfaces through SSH tunneling
- explicit websocket endpoint support
- testnet-first deployment

## Known Limitations

- public testnet websocket endpoints can still rate-limit under load; a dedicated endpoint is recommended for sustained operations
- the current live public audit is intentionally lightweight, not a full explorer
- a fully separate public audit bucket or custom domain still depends on operator-managed bucket permissions
- alert rules are provisioned, but notification routing is intentionally left to the operator
- the repo is testnet-first today, not presented as a finished mainnet deployment

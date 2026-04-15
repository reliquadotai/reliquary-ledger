[![CPU CI](https://github.com/0xgrizz/reliquary-inference/actions/workflows/cpu-ci.yml/badge.svg)](https://github.com/0xgrizz/reliquary-inference/actions/workflows/cpu-ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Testnet Live](https://img.shields.io/badge/testnet-live-brightgreen)](docs/status.md)

# Reliquary

This repository contains the inference runtime for Reliquary: a proof-carrying Bittensor subnet for **verifiable completions**. Miners submit completions tied to deterministic tasks, validators replay hidden-state sketch proofs and task bindings, and weights are set only on accepted unique work.

![Reliquary audit preview](docs/assets/audit-index-preview.svg)

## Live Status

- Live on Bittensor `test`
- GPU-backed real-model miner and validator staging path
- Hidden-state sketch verification over shared Hugging Face replay
- R2-backed artifact storage
- Public audit:
  - [HTML index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html)
  - [JSON index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.json)

The current sanitized status surface lives in [docs/status.md](docs/status.md).

## What Reliquary Does

Reliquary treats **accepted proof-carrying completions** as the commodity:

- deterministic task batches
- completion bundles with signature bindings
- validator-side replay with the same forward path
- duplicate and copycat rejection
- auditable manifests and public score summaries

The runtime stays narrow by design: clear to inspect, easy to test, and straightforward to operate.

## Verify It Yourself

```bash
git clone https://github.com/0xgrizz/reliquary-inference.git
cd reliquary-inference
cp env.example .env
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]" && reliquary-inference demo-local
```

For a live chain-read path without on-chain writes:

```bash
bash deploy/testnet-readonly-smoke.sh
```

## Scope

Included:

- `dataset_prompts` and `reasoning_tasks`
- single-GPU Hugging Face miner mode
- optional dual-engine miner mode with `vLLM` generation and Hugging Face proof replay
- local registry for development
- R2-backed registry for staging and testnet
- public audit index with stable `audit/index.json` and `audit/index.html`
- private-first monitoring with Prometheus, Grafana, node exporter, and a local metrics exporter

Excluded:

- distillation packs
- learner checkpoints
- policy authority
- RL rollouts
- async GRPO

## Documentation

- [Overview](docs/overview.md)
- [Current status](docs/status.md)
- [FAQ](docs/faq.md)
- [Architecture](docs/architecture.md)
- [Protocol](docs/protocol.md)
- [Scoring](docs/scoring.md)
- [Audit surface](docs/audit.md)
- [Monitoring](docs/monitoring.md)
- [Deployment guide](docs/deployment.md)
- [Readiness review](docs/readiness-review.md)
- [Review findings](docs/review-findings.md)
- [Release checklist](docs/release-checklist.md)

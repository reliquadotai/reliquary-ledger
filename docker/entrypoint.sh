#!/usr/bin/env bash
# Entrypoint for the Reliquary Ledger image.
#
# Reads RELIQUARY_INFERENCE_ROLE to decide which command to exec:
#   miner       → reliquary-inference run-miner
#   validator   → reliquary-inference run-validator
#   demo-local  → reliquary-inference demo-local
#   metrics     → reliquary-inference metrics-exporter
#
# All other config flows through env vars (RELIQUARY_INFERENCE_*, BT_*).
# See env.testnet.example for the full operator contract.
#
# Parallel-work credit: romain13190/reliquary@ed6da6c. the upstream entrypoint
# inspired the env-driven dispatch shape; the role taxonomy is ours
# (separate miner/validator/metrics commands rather than a single
# `validate` with --no-train).
set -euo pipefail

ROLE="${RELIQUARY_INFERENCE_ROLE:-miner}"

# Bind health + metrics to all interfaces inside the container so the
# operator's -p mapping can publish them. The defaults in env.example
# are 127.0.0.1 (safe for bare-metal); we override here.
export RELIQUARY_INFERENCE_METRICS_BIND="${RELIQUARY_INFERENCE_METRICS_BIND:-0.0.0.0}"
export RELIQUARY_INFERENCE_HEALTH_BIND="${RELIQUARY_INFERENCE_HEALTH_BIND:-0.0.0.0}"

extra_args=()
if [[ "${RELIQUARY_INFERENCE_ONCE:-0}" == "1" ]]; then
  extra_args+=(--once)
fi
if [[ -n "${RELIQUARY_INFERENCE_RESUME_FROM:-}" ]]; then
  extra_args+=(--resume-from "${RELIQUARY_INFERENCE_RESUME_FROM}")
fi

case "${ROLE}" in
  miner)
    echo "[entrypoint] launching reliquary-inference run-miner ${extra_args[*]:-}"
    exec reliquary-inference run-miner "${extra_args[@]}"
    ;;
  validator)
    echo "[entrypoint] launching reliquary-inference run-validator ${extra_args[*]:-}"
    exec reliquary-inference run-validator "${extra_args[@]}"
    ;;
  metrics)
    echo "[entrypoint] launching reliquary-inference metrics-exporter"
    exec reliquary-inference metrics-exporter
    ;;
  demo-local)
    echo "[entrypoint] launching reliquary-inference demo-local"
    exec reliquary-inference demo-local
    ;;
  *)
    echo "[entrypoint] unknown RELIQUARY_INFERENCE_ROLE=${ROLE}; expected miner|validator|metrics|demo-local" >&2
    exit 64
    ;;
esac

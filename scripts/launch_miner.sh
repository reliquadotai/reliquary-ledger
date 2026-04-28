#!/usr/bin/env bash
# Launch a Reliquary Ledger miner in the background.
#
# Quickstart (testnet 462):
#   cp env.testnet.example .env  # then edit netuid, R2 creds, wallet
#   bash scripts/launch_miner.sh
#
# Or with Docker:
#   docker compose up -d miner
#
# Variables (from .env, overridable):
#   RELIQUARY_INSTALL_DIR — checkout root (default: pwd)
#   RELIQUARY_INFERENCE_LOG_DIR — where logs go (default: ./data/logs)
#   RELIQUARY_INFERENCE_RESUME_FROM — optional: sha:<hex> | path:<dir>
#
# Parallel-work credit: romain13190/reliquary@5f8fbb3 — script ergonomics
# (pkill prior, clear __pycache__, nohup, PID file). Our CLI is env-driven
# so the script doesn't need to forward most flags.

set -euo pipefail

# Resolve repo root from script location so the script works regardless
# of where the operator invokes it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Source .env if present so `bash scripts/launch_miner.sh` works without
# requiring the operator to `source .env` first.
if [[ -f "${INSTALL_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${INSTALL_DIR}/.env"
  set +a
fi

LOG_DIR="${RELIQUARY_INFERENCE_LOG_DIR:-${INSTALL_DIR}/data/logs}"
LOG_FILE="${LOG_DIR}/miner.log"
PID_FILE="${LOG_DIR}/miner.pid"
mkdir -p "${LOG_DIR}"

# Required for any chain-touching mode.
: "${RELIQUARY_INFERENCE_NETWORK:?RELIQUARY_INFERENCE_NETWORK must be set (e.g. test, finney)}"
: "${RELIQUARY_INFERENCE_NETUID:?RELIQUARY_INFERENCE_NETUID must be set (e.g. 462)}"

cd "${INSTALL_DIR}"

# Editable installs pick up source changes but stale __pycache__ can mask
# them — clear on every launch.
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Stop any prior miner under this checkout.
if [[ -f "${PID_FILE}" ]]; then
  prior_pid="$(cat "${PID_FILE}")"
  if kill -0 "${prior_pid}" 2>/dev/null; then
    echo "stopping prior miner PID=${prior_pid}"
    kill "${prior_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${prior_pid}" 2>/dev/null || true
  fi
fi
pkill -9 -f 'reliquary-inference run-miner' 2>/dev/null || true
sleep 1
rm -f "${LOG_FILE}" "${PID_FILE}"

# Pick the venv python if present, otherwise system python.
PYTHON="${INSTALL_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

extra_args=()
if [[ -n "${RELIQUARY_INFERENCE_RESUME_FROM:-}" ]]; then
  extra_args+=(--resume-from "${RELIQUARY_INFERENCE_RESUME_FROM}")
fi

echo "launching miner: network=${RELIQUARY_INFERENCE_NETWORK} netuid=${RELIQUARY_INFERENCE_NETUID} log=${LOG_FILE}"
nohup "${PYTHON}" -m reliquary_inference.cli run-miner "${extra_args[@]}" \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "miner PID=$(cat "${PID_FILE}") log=${LOG_FILE}"

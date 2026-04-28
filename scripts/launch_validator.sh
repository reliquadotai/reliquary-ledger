#!/usr/bin/env bash
# Launch a Reliquary Ledger validator in the background.
#
# Quickstart (testnet 462):
#   cp env.testnet.example .env  # then edit netuid, R2 creds, wallet
#   bash scripts/launch_validator.sh
#
# Or with Docker:
#   docker compose up -d validator
#
# Variables (from .env, overridable):
#   RELIQUARY_INSTALL_DIR — checkout root (default: pwd)
#   RELIQUARY_INFERENCE_LOG_DIR — where logs go (default: ./data/logs)
#   RELIQUARY_INFERENCE_RESUME_FROM — optional: sha:<hex> | path:<dir>
#
# Parallel-work credit: romain13190/reliquary@5f8fbb3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -f "${INSTALL_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${INSTALL_DIR}/.env"
  set +a
fi

LOG_DIR="${RELIQUARY_INFERENCE_LOG_DIR:-${INSTALL_DIR}/data/logs}"
LOG_FILE="${LOG_DIR}/validator.log"
PID_FILE="${LOG_DIR}/validator.pid"
mkdir -p "${LOG_DIR}"

: "${RELIQUARY_INFERENCE_NETWORK:?RELIQUARY_INFERENCE_NETWORK must be set (e.g. test, finney)}"
: "${RELIQUARY_INFERENCE_NETUID:?RELIQUARY_INFERENCE_NETUID must be set (e.g. 462)}"

cd "${INSTALL_DIR}"

find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

if [[ -f "${PID_FILE}" ]]; then
  prior_pid="$(cat "${PID_FILE}")"
  if kill -0 "${prior_pid}" 2>/dev/null; then
    echo "stopping prior validator PID=${prior_pid}"
    kill "${prior_pid}" 2>/dev/null || true
    sleep 2
    kill -9 "${prior_pid}" 2>/dev/null || true
  fi
fi
pkill -9 -f 'reliquary-inference run-validator' 2>/dev/null || true
sleep 1
rm -f "${LOG_FILE}" "${PID_FILE}"

PYTHON="${INSTALL_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

extra_args=()
if [[ -n "${RELIQUARY_INFERENCE_RESUME_FROM:-}" ]]; then
  extra_args+=(--resume-from "${RELIQUARY_INFERENCE_RESUME_FROM}")
fi

echo "launching validator: network=${RELIQUARY_INFERENCE_NETWORK} netuid=${RELIQUARY_INFERENCE_NETUID} log=${LOG_FILE}"
nohup "${PYTHON}" -m reliquary_inference.cli run-validator "${extra_args[@]}" \
    > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "validator PID=$(cat "${PID_FILE}") log=${LOG_FILE}"

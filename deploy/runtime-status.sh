#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/reliquary-inference/current}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

cd "${REPO_ROOT}"

if command -v systemctl >/dev/null 2>&1; then
  echo "== services =="
  systemctl is-active inference-miner.service inference-validator.service || true
  echo
fi

echo "== reliquary-inference status =="
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" reliquary-inference status --json

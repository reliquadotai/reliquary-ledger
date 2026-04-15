#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
REPO_ROOT="${REPO_ROOT:-/srv/reliquary-inference/current}"
APP_ENV="${APP_ENV:-reliquary-inference}"
EXTRAS="${EXTRAS:-dev}"

if ! "${MINIFORGE_ROOT}/bin/conda" env list | awk '{print $1}' | grep -qx "${APP_ENV}"; then
  "${MINIFORGE_ROOT}/bin/conda" create -y -n "${APP_ENV}" python=3.12 pip
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -m pip install --upgrade pip setuptools wheel
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -m pip install -e "${REPO_ROOT}[${EXTRAS}]"

echo "Environment installed:"
"${MINIFORGE_ROOT}/bin/conda" env list

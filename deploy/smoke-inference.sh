#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -c "import torch; print(torch.cuda.is_available())"
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" reliquary-inference demo-local --source reasoning_tasks --count 2

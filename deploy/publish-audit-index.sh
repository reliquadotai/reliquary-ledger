#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/reliquary-inference/current}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
LIMIT="${LIMIT:-25}"
PUBLISH="${PUBLISH:-true}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

cd "${REPO_ROOT}"

args=(build-audit-index --limit "${LIMIT}")
if [ "${PUBLISH}" = "true" ]; then
  args+=(--publish)
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" reliquary-inference "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
RESTART_SERVICES="${RESTART_SERVICES:-false}"

mkdir -p "$(dirname "${ENV_FILE}")"

if [ -f "${ENV_FILE}" ]; then
  cp "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - "${ENV_FILE}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

env_path = Path(sys.argv[1])
updates = {
    "RELIQUARY_INFERENCE_STORAGE_BACKEND": "local",
    "RELIQUARY_INFERENCE_EXPOSE_PUBLIC_ARTIFACT_URLS": "false",
}

lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
seen: set[str] = set()
rendered: list[str] = []

for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        rendered.append(line)
        continue
    key, _, _value = line.partition("=")
    if key in updates:
        rendered.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        rendered.append(line)

for key, value in updates.items():
    if key not in seen:
        rendered.append(f"{key}={value}")

env_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
PY

chmod 600 "${ENV_FILE}"
echo "Applied local storage profile to ${ENV_FILE}"

if [ "${RESTART_SERVICES}" = "true" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart inference-miner.service inference-validator.service
  systemctl --no-pager --full status inference-miner.service inference-validator.service || true
fi

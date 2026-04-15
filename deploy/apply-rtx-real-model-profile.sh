#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
MODEL_REF="${MODEL_REF:-Qwen/Qwen2.5-1.5B-Instruct}"
DEVICE="${DEVICE:-cuda}"
LOAD_DTYPE="${LOAD_DTYPE:-bf16}"
TASK_SOURCE="${TASK_SOURCE:-reasoning_tasks}"
TASK_COUNT="${TASK_COUNT:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
MINER_MODE="${MINER_MODE:-single_gpu_hf}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
RESTART_SERVICES="${RESTART_SERVICES:-false}"

mkdir -p "$(dirname "${ENV_FILE}")"

if [ -f "${ENV_FILE}" ]; then
  cp "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - "${ENV_FILE}" "${MODEL_REF}" "${DEVICE}" "${LOAD_DTYPE}" "${TASK_SOURCE}" "${TASK_COUNT}" "${MAX_NEW_TOKENS}" "${MINER_MODE}" "${POLL_INTERVAL}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

env_path = Path(sys.argv[1])
updates = {
    "RELIQUARY_INFERENCE_MODEL_REF": sys.argv[2],
    "RELIQUARY_INFERENCE_DEVICE": sys.argv[3],
    "RELIQUARY_INFERENCE_LOAD_DTYPE": sys.argv[4],
    "RELIQUARY_INFERENCE_TASK_SOURCE": sys.argv[5],
    "RELIQUARY_INFERENCE_TASK_COUNT": sys.argv[6],
    "RELIQUARY_INFERENCE_MAX_NEW_TOKENS": sys.argv[7],
    "RELIQUARY_INFERENCE_MINER_MODE": sys.argv[8],
    "RELIQUARY_INFERENCE_POLL_INTERVAL": sys.argv[9],
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

echo "Applied RTX real-model profile to ${ENV_FILE}"

if [ "${RESTART_SERVICES}" = "true" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart inference-miner.service inference-validator.service
  systemctl --no-pager --full status inference-miner.service inference-validator.service || true
fi

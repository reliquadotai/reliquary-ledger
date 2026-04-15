#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
CHAIN_ENDPOINT="${CHAIN_ENDPOINT:-}"
NETWORK="${NETWORK:-test}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
RESTART_SERVICES="${RESTART_SERVICES:-false}"

if [ -z "${CHAIN_ENDPOINT}" ]; then
  echo "CHAIN_ENDPOINT is required." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_FILE}")"

if [ -f "${ENV_FILE}" ]; then
  cp "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - "${CHAIN_ENDPOINT}" "${NETWORK}" <<'PY'
from __future__ import annotations

import sys

import bittensor as bt

chain_endpoint = sys.argv[1]
network = sys.argv[2]

config = bt.Subtensor.config()
config.subtensor.chain_endpoint = chain_endpoint
subtensor = bt.Subtensor(network=None, config=config)
print(f"validated endpoint {chain_endpoint} on network {network} at block {int(subtensor.block)}")
PY

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - "${ENV_FILE}" "${CHAIN_ENDPOINT}" "${POLL_INTERVAL}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

env_path = Path(sys.argv[1])
updates = {
    "RELIQUARY_INFERENCE_CHAIN_ENDPOINT": sys.argv[2],
    "BT_SUBTENSOR_CHAIN_ENDPOINT": sys.argv[2],
    "RELIQUARY_INFERENCE_POLL_INTERVAL": sys.argv[3],
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
echo "Applied chain endpoint profile to ${ENV_FILE}"

if [ "${RESTART_SERVICES}" = "true" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart inference-miner.service inference-validator.service
  systemctl --no-pager --full status inference-miner.service inference-validator.service || true
fi

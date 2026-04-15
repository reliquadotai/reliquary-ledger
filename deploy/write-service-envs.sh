#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/srv/reliquary-inference/state}"
SERVICE_USER="${SERVICE_USER:-reliquary}"
MINER_HOTKEY="${MINER_HOTKEY:-miner}"
VALIDATOR_HOTKEY="${VALIDATOR_HOTKEY:-validator}"
PUBLIC_FILE="${PUBLIC_FILE:-${STATE_ROOT}/bootstrap/wallet-public.json}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
MINER_ID="${MINER_ID:-}"
VALIDATOR_ID="${VALIDATOR_ID:-}"
SIGNATURE_SCHEME="${SIGNATURE_SCHEME:-bittensor_hotkey}"

if [ -f "${PUBLIC_FILE}" ]; then
  ids_json="$("${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${PUBLIC_FILE}").read_text(encoding="utf-8"))
print(json.dumps({
    "miner_id": payload.get("miner_hotkey_ss58", ""),
    "validator_id": payload.get("validator_hotkey_ss58", ""),
}))
PY
)"
  if [ -z "${MINER_ID}" ]; then
    MINER_ID="$(printf '%s' "${ids_json}" | "${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -c 'import json,sys; print(json.loads(sys.stdin.read()).get("miner_id",""))')"
  fi
  if [ -z "${VALIDATOR_ID}" ]; then
    VALIDATOR_ID="$(printf '%s' "${ids_json}" | "${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python -c 'import json,sys; print(json.loads(sys.stdin.read()).get("validator_id",""))')"
  fi
fi

MINER_ID="${MINER_ID:-rtx-miner}"
VALIDATOR_ID="${VALIDATOR_ID:-rtx-validator}"

cat > "${STATE_ROOT}/inference-miner.env" <<EOF
HOTKEY_NAME=${MINER_HOTKEY}
RELIQUARY_INFERENCE_MINER_ID=${MINER_ID}
RELIQUARY_INFERENCE_SIGNATURE_SCHEME=${SIGNATURE_SCHEME}
EOF

cat > "${STATE_ROOT}/inference-validator.env" <<EOF
HOTKEY_NAME=${VALIDATOR_HOTKEY}
RELIQUARY_INFERENCE_VALIDATOR_ID=${VALIDATOR_ID}
EOF

chown "${SERVICE_USER}:${SERVICE_USER}" \
  "${STATE_ROOT}/inference-miner.env" \
  "${STATE_ROOT}/inference-validator.env"
chmod 640 "${STATE_ROOT}/inference-miner.env" "${STATE_ROOT}/inference-validator.env"

#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
SERVICE_USER="${SERVICE_USER:-reliquary}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

NETWORK="${RELIQUARY_INFERENCE_NETWORK:-${BT_SUBTENSOR_NETWORK:-test}}"
WALLET_NAME="${WALLET_NAME:-reliquary-inference}"
WALLET_PATH="${BT_WALLET_PATH:-/home/${SERVICE_USER}/.bittensor/wallets}"

runuser -u "${SERVICE_USER}" -- /bin/bash -lc "
  export RELIQUARY_INFERENCE_NETWORK='${NETWORK}'
  export BT_SUBTENSOR_NETWORK='${NETWORK}'
  export WALLET_NAME='${WALLET_NAME}'
  export BT_WALLET_PATH='${WALLET_PATH}'
  /opt/miniforge/bin/conda run --no-capture-output -n '${APP_ENV}' python - <<'PY'
import json
import os

import bittensor as bt

network = os.environ['RELIQUARY_INFERENCE_NETWORK']
wallet_name = os.environ['WALLET_NAME']
wallet_path = os.environ['BT_WALLET_PATH']

wallet = bt.Wallet(name=wallet_name, hotkey='validator', path=wallet_path)
subtensor = bt.Subtensor(network=network)

response = subtensor.register_subnet(
    wallet=wallet,
    wait_for_inclusion=True,
    wait_for_finalization=False,
)
payload = {
    'network': network,
    'wallet_name': wallet_name,
    'wallet_path': wallet_path,
    'response': str(response),
    'response_type': type(response).__name__,
}
print(json.dumps(payload, indent=2))
PY
"

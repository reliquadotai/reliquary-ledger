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
NETUID="${RELIQUARY_INFERENCE_NETUID:-1}"
WALLET_NAME="${WALLET_NAME:-reliquary-inference}"
WALLET_PATH="${BT_WALLET_PATH:-/home/${SERVICE_USER}/.bittensor/wallets}"
VALIDATOR_HOTKEY="${VALIDATOR_HOTKEY:-validator}"
MINER_HOTKEY="${MINER_HOTKEY:-miner}"

runuser -u "${SERVICE_USER}" -- /bin/bash -lc "
  export RELIQUARY_INFERENCE_NETWORK='${NETWORK}'
  export RELIQUARY_INFERENCE_NETUID='${NETUID}'
  export WALLET_NAME='${WALLET_NAME}'
  export BT_WALLET_PATH='${WALLET_PATH}'
  export VALIDATOR_HOTKEY='${VALIDATOR_HOTKEY}'
  export MINER_HOTKEY='${MINER_HOTKEY}'
  '${MINIFORGE_ROOT}/bin/conda' run --no-capture-output -n '${APP_ENV}' python - <<'PY'
import json
import os

import bittensor as bt

network = os.environ['RELIQUARY_INFERENCE_NETWORK']
netuid = int(os.environ['RELIQUARY_INFERENCE_NETUID'])
wallet_name = os.environ['WALLET_NAME']
wallet_path = os.environ['BT_WALLET_PATH']
hotkeys = [os.environ['VALIDATOR_HOTKEY'], os.environ['MINER_HOTKEY']]

subtensor = bt.Subtensor(network=network)
results = []
for hotkey_name in hotkeys:
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    hotkey_ss58 = wallet.hotkey.ss58_address
    already_registered = subtensor.is_hotkey_registered_on_subnet(hotkey_ss58, netuid)
    if already_registered:
        results.append(
            {
                'hotkey_name': hotkey_name,
                'hotkey_ss58': hotkey_ss58,
                'status': 'already_registered',
            }
        )
        continue
    response = subtensor.burned_register(
        wallet=wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )
    results.append(
        {
            'hotkey_name': hotkey_name,
            'hotkey_ss58': hotkey_ss58,
            'status': 'submitted',
            'response': str(response),
            'response_type': type(response).__name__,
        }
    )

print(json.dumps(
    {
        'network': network,
        'netuid': netuid,
        'wallet_name': wallet_name,
        'wallet_path': wallet_path,
        'results': results,
    },
    indent=2,
))
PY
"

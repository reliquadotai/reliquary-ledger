#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
SERVICE_USER="${SERVICE_USER:-reliquary}"
STATE_ROOT="${STATE_ROOT:-/srv/reliquary-inference/state}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${STATE_ROOT}/bootstrap}"
WALLET_PATH="${BT_WALLET_PATH:-/home/${SERVICE_USER}/.bittensor/wallets}"
WALLET_NAME="${WALLET_NAME:-reliquary-inference}"
VALIDATOR_HOTKEY="${VALIDATOR_HOTKEY:-validator}"
MINER_HOTKEY="${MINER_HOTKEY:-miner}"
PUBLIC_FILE="${PUBLIC_FILE:-${BOOTSTRAP_DIR}/wallet-public.json}"
BITTENSOR_ROOT="/home/${SERVICE_USER}/.bittensor"

mkdir -p "${BOOTSTRAP_DIR}" "${BITTENSOR_ROOT}/miners" "${WALLET_PATH}"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${BOOTSTRAP_DIR}" "${BITTENSOR_ROOT}" "${WALLET_PATH}"

runuser -u "${SERVICE_USER}" -- /bin/bash -lc "
  export BT_WALLET_PATH='${WALLET_PATH}'
  export WALLET_NAME='${WALLET_NAME}'
  export VALIDATOR_HOTKEY='${VALIDATOR_HOTKEY}'
  export MINER_HOTKEY='${MINER_HOTKEY}'
  /opt/miniforge/bin/conda run --no-capture-output -n '${APP_ENV}' python - <<'PY'
import json
import os
from pathlib import Path

import bittensor as bt

wallet_path = os.environ['BT_WALLET_PATH']
wallet_name = os.environ['WALLET_NAME']
validator_hotkey = os.environ['VALIDATOR_HOTKEY']
miner_hotkey = os.environ['MINER_HOTKEY']

validator_wallet = bt.Wallet(name=wallet_name, hotkey=validator_hotkey, path=wallet_path)
validator_wallet.create_if_non_existent(
    coldkey_use_password=False,
    hotkey_use_password=False,
    suppress=True,
)

miner_wallet = bt.Wallet(name=wallet_name, hotkey=miner_hotkey, path=wallet_path)
miner_wallet.create_new_hotkey(
    use_password=False,
    overwrite=False,
    suppress=True,
)

payload = {
    'wallet_name': wallet_name,
    'wallet_path': wallet_path,
    'coldkey_ss58': validator_wallet.coldkeypub.ss58_address,
    'validator_hotkey_name': validator_hotkey,
    'validator_hotkey_ss58': validator_wallet.hotkey.ss58_address,
    'miner_hotkey_name': miner_hotkey,
    'miner_hotkey_ss58': miner_wallet.hotkey.ss58_address,
}
Path('${PUBLIC_FILE}').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(json.dumps(payload, indent=2))
PY
"

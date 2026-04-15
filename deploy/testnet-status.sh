#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
PUBLIC_FILE="${PUBLIC_FILE:-/srv/reliquary-inference/state/bootstrap/wallet-public.json}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - <<'PY'
import json
import os
from pathlib import Path

import bittensor as bt

network = os.getenv("RELIQUARY_INFERENCE_NETWORK", os.getenv("BT_SUBTENSOR_NETWORK", "test"))
netuid = int(os.getenv("RELIQUARY_INFERENCE_NETUID", "1"))
public_file = Path(os.getenv("PUBLIC_FILE", "/srv/reliquary-inference/state/bootstrap/wallet-public.json"))

subtensor = bt.Subtensor(network=network)
status = {
    "network": network,
    "block": int(subtensor.block),
    "total_subnets": int(subtensor.get_total_subnets()),
    "subnet_burn_cost": str(subtensor.get_subnet_burn_cost()),
    "netuid": netuid,
    "subnet_exists": bool(subtensor.subnet_exists(netuid)),
}

if public_file.exists():
    wallet_info = json.loads(public_file.read_text(encoding="utf-8"))
    coldkey = wallet_info["coldkey_ss58"]
    validator_hotkey = wallet_info["validator_hotkey_ss58"]
    miner_hotkey = wallet_info["miner_hotkey_ss58"]
    status["wallet"] = wallet_info
    status["coldkey_balance"] = str(subtensor.get_balance(coldkey))
    status["validator_registered_any"] = bool(subtensor.is_hotkey_registered_any(validator_hotkey))
    status["miner_registered_any"] = bool(subtensor.is_hotkey_registered_any(miner_hotkey))
    if status["subnet_exists"]:
        status["validator_registered"] = bool(subtensor.is_hotkey_registered_on_subnet(validator_hotkey, netuid))
        status["miner_registered"] = bool(subtensor.is_hotkey_registered_on_subnet(miner_hotkey, netuid))

print(json.dumps(status, indent=2))
PY

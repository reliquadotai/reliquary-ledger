#!/usr/bin/env bash
# Apply the Reliquary Ledger mainnet (SN81) profile.
#
# This profile targets Bittensor finney / SN81 with autonomous commit +
# push semantics; force-push and mainnet btcli mutations are still gated
# by the user-level Claude Code hook + repo settings.
#
# Safety: deliberately refuses to run without ALLOW_MAINNET=1 so a stray
# invocation does not point a miner/validator at real TAO.

set -euo pipefail

if [[ "${ALLOW_MAINNET:-}" != "1" ]]; then
  echo "refusing to apply mainnet profile without ALLOW_MAINNET=1"
  echo "re-run with: ALLOW_MAINNET=1 $0"
  exit 1
fi

ENV_FILE="${1:-./deploy/mainnet/sn81.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE"
  echo "copy deploy/mainnet/sn81.env.template first and fill in wallet + endpoint details"
  exit 2
fi

echo "applying mainnet SN81 profile from $ENV_FILE"
# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

python3 -m reliquary_inference.cli diagnose-config || true

echo "profile applied in current shell; launch services with this env active:"
echo "  systemctl --user start reliquary-ledger-miner-mainnet"
echo "  systemctl --user start reliquary-ledger-validator-mainnet"

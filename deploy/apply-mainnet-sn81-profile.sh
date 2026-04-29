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

# Pre-flight: run diagnose-config and propagate the exit code. The
# operator wants the script to abort if the env file is missing
# required fields; the previous version swallowed failures with
# `|| true` and printed a misleading "profile applied" line even when
# config validation tripped. Use the entrypoint binary so we don't
# rely on `python -m` semantics (the typer app is wired via
# [project.scripts] in pyproject.toml, not as a __main__ block).
if command -v reliquary-inference >/dev/null 2>&1; then
  reliquary-inference diagnose-config
else
  echo "reliquary-inference entrypoint not found in PATH; aborting" >&2
  echo "ensure the venv (e.g. /opt/reliquary-venv) is activated before re-running" >&2
  exit 3
fi

echo "profile applied in current shell; launch services with this env active:"
echo "  systemctl --user start reliquary-ledger-miner-mainnet"
echo "  systemctl --user start reliquary-ledger-validator-mainnet"

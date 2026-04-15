#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
STATE_ROOT="${STATE_ROOT:-/srv/reliquary-inference/state}"
CREATE_SUBNET="${CREATE_SUBNET:-false}"
ENABLE_SERVICES="${ENABLE_SERVICES:-true}"
SERVICE_USER="${SERVICE_USER:-reliquary}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

if [ "${RELIQUARY_INFERENCE_NETWORK:-local}" = "local" ]; then
  echo "RELIQUARY_INFERENCE_NETWORK is local; set it to test before activation." >&2
  exit 1
fi

echo "== testnet status =="
status_json="$(bash /srv/reliquary-inference/current/deploy/testnet-status.sh)"
echo "${status_json}"

balance="$(printf '%s\n' "${status_json}" | jq -r '.coldkey_balance // ""')"
subnet_exists="$(printf '%s\n' "${status_json}" | jq -r '.subnet_exists // false')"

if [ "${CREATE_SUBNET}" != "true" ] && [ "${subnet_exists}" != "true" ]; then
  echo "Target subnet does not exist and CREATE_SUBNET is not true." >&2
  exit 1
fi

case "${balance}" in
  ""|"τ0"*|"0"* )
    echo "Coldkey is not funded on the target network. Fund the wallet before activation." >&2
    exit 1
    ;;
esac

if [ "${CREATE_SUBNET}" = "true" ]; then
  echo "== creating subnet =="
  bash /srv/reliquary-inference/current/deploy/register-test-subnet.sh
  echo "Update RELIQUARY_INFERENCE_NETUID in ${ENV_FILE} if a new subnet was created."
fi

echo "== registering hotkeys =="
bash /srv/reliquary-inference/current/deploy/register-hotkeys.sh

echo "== readonly smoke =="
bash /srv/reliquary-inference/current/deploy/testnet-readonly-smoke.sh

echo "== writing service envs =="
bash /srv/reliquary-inference/current/deploy/write-service-envs.sh

echo "== installing systemd units =="
install -m 644 /srv/reliquary-inference/current/deploy/systemd/inference-miner.service /etc/systemd/system/inference-miner.service
install -m 644 /srv/reliquary-inference/current/deploy/systemd/inference-validator.service /etc/systemd/system/inference-validator.service
systemctl daemon-reload

if [ "${ENABLE_SERVICES}" = "true" ]; then
  echo "== enabling services =="
  systemctl enable --now inference-miner.service inference-validator.service
else
  echo "Skipping service enable because ENABLE_SERVICES=${ENABLE_SERVICES}"
fi

echo "== final service status =="
systemctl --no-pager --full status inference-miner.service inference-validator.service || true

echo "Activation flow complete."

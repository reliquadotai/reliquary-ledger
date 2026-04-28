#!/usr/bin/env bash
# Bootstrap a Reliquary Ledger node on a fresh Linux GPU box.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/reliquadotai/reliquary-ledger/main/scripts/setup_remote_box.sh | bash
#   # or, after cloning:
#   bash scripts/setup_remote_box.sh
#
# What it does:
#   1. Installs system deps (build tools, python3.12, git, curl, jq).
#   2. Clones (or updates) reliquary-ledger to /opt/reliquary-ledger.
#   3. Creates a Python 3.12 venv at .venv with editable install.
#   4. Drops env.testnet.example -> .env if .env doesn't exist.
#   5. Installs the systemd template units (commented; operator enables).
#
# Idempotent: safe to re-run on existing boxes.
#
# Parallel-work credit: romain13190/reliquary@5f8fbb3 — same shape, our
# install paths and CLI invocations.

set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/reliquary-ledger}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
REPO_URL="${REPO_URL:-https://github.com/reliquadotai/reliquary-ledger.git}"
BRANCH="${BRANCH:-main}"

log() { echo "[setup_remote_box] $*"; }

if [[ "${EUID}" -ne 0 ]]; then
  log "warning: not running as root; system package install will be skipped."
  log "  re-run with sudo to enable apt-get install steps."
fi

# ────────  1. system deps  ────────
if [[ "${EUID}" -eq 0 ]]; then
  log "installing system deps via apt-get"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq \
    git build-essential wget curl ca-certificates jq \
    python3.12 python3.12-venv python3-pip
fi

# ────────  2. clone / update  ────────
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  log "updating existing checkout at ${INSTALL_DIR}"
  cd "${INSTALL_DIR}"
  git fetch origin
  git checkout "${BRANCH}"
  git pull origin "${BRANCH}"
else
  log "cloning ${REPO_URL} to ${INSTALL_DIR}"
  if [[ "${EUID}" -eq 0 ]]; then
    install -d -m 755 "${INSTALL_DIR}"
  else
    mkdir -p "${INSTALL_DIR}" || sudo install -d -m 755 -o "${USER}" "${INSTALL_DIR}"
  fi
  git clone -b "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"

# ────────  3. venv + editable install  ────────
if [[ ! -d "${INSTALL_DIR}/.venv" ]]; then
  log "creating venv at ${INSTALL_DIR}/.venv"
  "${PYTHON_BIN}" -m venv "${INSTALL_DIR}/.venv"
fi

log "installing reliquary-inference (editable) into venv"
"${INSTALL_DIR}/.venv/bin/pip" install --upgrade pip wheel setuptools >/dev/null
"${INSTALL_DIR}/.venv/bin/pip" install -e ".[dev]" >/dev/null

# ────────  4. seed .env  ────────
if [[ ! -f "${INSTALL_DIR}/.env" ]]; then
  log "seeding .env from env.testnet.example (edit before launch)"
  cp "${INSTALL_DIR}/env.testnet.example" "${INSTALL_DIR}/.env"
  chmod 600 "${INSTALL_DIR}/.env"
fi

# ────────  5. systemd templates (do not enable)  ────────
if [[ "${EUID}" -eq 0 && -d "${INSTALL_DIR}/deploy/systemd" ]]; then
  log "copying systemd unit templates to /etc/systemd/system (not enabled)"
  for unit in "${INSTALL_DIR}"/deploy/systemd/*.service; do
    [[ -f "${unit}" ]] || continue
    install -m 644 "${unit}" "/etc/systemd/system/$(basename "${unit}")"
  done
  systemctl daemon-reload
fi

cat <<EOF

[setup_remote_box] complete.

Next steps:
  1. Edit ${INSTALL_DIR}/.env (netuid, wallet path, R2 creds)
  2. Verify: ${INSTALL_DIR}/.venv/bin/reliquary-inference status
  3. Launch:
       bash ${INSTALL_DIR}/scripts/launch_validator.sh   # or launch_miner.sh
     or with systemd:
       sudo systemctl enable --now reliquary-ledger-validator-mainnet.service
  4. Health check:
       curl http://localhost:9180/healthz

Logs:    ${INSTALL_DIR}/data/logs/
Docs:    https://github.com/reliquadotai/reliquary-ledger/blob/main/START_HERE.md
EOF

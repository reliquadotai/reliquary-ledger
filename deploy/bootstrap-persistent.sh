#!/usr/bin/env bash
# Restore a Reliquary Ledger devserver onto a persistent volume (default: /save).
#
# Designed for Targon-style nested containers where /srv, /home, /opt, and /etc
# all live on the ephemeral overlay and get wiped on every container restart.
# The persistent volume holds the durable bits (repos, venv, wallet, HF cache,
# prometheus data) and this script re-wires the standard paths into it on every
# container boot.
#
# Run on every container restart:
#   bash /save/bootstrap-persistent.sh
#
# Required environment (or defaults):
#   SAVE=/save                  (root of the persistent volume)
#   VENV=/save/venv/reliquary   (python venv — pre-built)
#   REPOS=/save/repos           (git checkouts)
#   STATE=/save/state           (env files + artifacts + logs)
#   WALLETS=/save/.bittensor    (Bittensor wallet dir)
#   HF_CACHE=/save/hf-cache
#   SYSTEMD_SRC=/save/systemd   (optional — unit files to install)
#
# The script:
#   1. apt-installs base tools (python3, git, tmux, jq, rsync, prometheus,
#      grafana, prometheus-node-exporter)
#   2. creates the `reliquary` system user (services run as this)
#   3. symlinks /srv/reliquary-inference/current, /srv/reliquary/current,
#      /opt/reliquary-venv, /root/.bittensor, /home/reliquary/.bittensor
#      into ${SAVE}
#   4. writes /etc/profile.d/reliquary-paths.sh exporting HF_HOME +
#      BITTENSOR_HOME + PATH
#   5. installs systemd units from ${SYSTEMD_SRC} + drop-in overrides
#      that pin prometheus + grafana data dirs under ${SAVE}
#   6. creates empty env-file templates at ${STATE}/*/*.env (never overwrites)
#
# Does NOT:
#   - create / restore wallets (operator scp's ~/.bittensor in)
#   - populate env files (they're operator-managed; secrets)
#   - systemctl enable any service (needs validated wallet + env first)

set -euo pipefail

SAVE="${SAVE:-/save}"
REPOS="${REPOS:-$SAVE/repos}"
VENV="${VENV:-$SAVE/venv/reliquary}"
WALLETS="${WALLETS:-$SAVE/.bittensor}"
STATE="${STATE:-$SAVE/state}"
HF_CACHE="${HF_CACHE:-$SAVE/hf-cache}"
SYSTEMD_SRC="${SYSTEMD_SRC:-$SAVE/systemd}"
SYSTEMD_OVERRIDES="${SYSTEMD_OVERRIDES:-$SAVE/systemd/overrides}"

if [[ ! -d "$SAVE" ]]; then
  echo "fatal: $SAVE not mounted — persistent volume missing" >&2
  exit 2
fi
if [[ ! -d "$REPOS/reliquary-inference" ]]; then
  echo "fatal: $REPOS/reliquary-inference missing — restore repos first" >&2
  exit 3
fi

log() { printf '[bootstrap %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# -----------------------------------------------------------------------
# 1. Base apt packages
# -----------------------------------------------------------------------
log "apt: install base tooling"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  build-essential ca-certificates git curl wget rsync jq tmux \
  prometheus prometheus-node-exporter >/dev/null

# Grafana lives in a custom apt repo — install once, re-use after reboot.
if ! command -v grafana-server >/dev/null; then
  log "apt: install grafana"
  apt-get install -y --no-install-recommends \
    gnupg2 apt-transport-https software-properties-common >/dev/null
  mkdir -p /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/grafana.gpg ]]; then
    wget -qO - https://apt.grafana.com/gpg.key | gpg --dearmor > /etc/apt/keyrings/grafana.gpg
    printf 'deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main\n' \
      > /etc/apt/sources.list.d/grafana.list
    apt-get update -q >/dev/null
  fi
  apt-get install -y --no-install-recommends grafana >/dev/null
fi

# -----------------------------------------------------------------------
# 2. reliquary system user
# -----------------------------------------------------------------------
if ! id -u reliquary >/dev/null 2>&1; then
  log "user: create reliquary"
  useradd --system --shell /bin/bash --home-dir "$SAVE" --no-create-home reliquary
fi

# -----------------------------------------------------------------------
# 3. /srv/ + /opt/ + wallet symlinks
# -----------------------------------------------------------------------
log "symlink: /srv/{reliquary-inference,reliquary}/current"
mkdir -p /srv/reliquary-inference /srv/reliquary
ln -sfn "$REPOS/reliquary-inference" /srv/reliquary-inference/current
ln -sfn "$REPOS/reliquary"           /srv/reliquary/current

mkdir -p "$STATE/reliquary-inference/bootstrap" "$STATE/reliquary"
ln -sfn "$STATE/reliquary-inference" /srv/reliquary-inference/state
ln -sfn "$STATE/reliquary"           /srv/reliquary/state

log "symlink: /opt/reliquary-venv"
ln -sfn "$VENV" /opt/reliquary-venv

log "symlink: /root/.bittensor + /home/reliquary/.bittensor → $WALLETS"
mkdir -p "$WALLETS"
ln -sfn "$WALLETS" /root/.bittensor
mkdir -p /home/reliquary
ln -sfn "$WALLETS" /home/reliquary/.bittensor
chown -h reliquary:reliquary /home/reliquary/.bittensor 2>/dev/null || true
chown -R reliquary:reliquary "$WALLETS" 2>/dev/null || true

# -----------------------------------------------------------------------
# 4. /etc/profile.d + systemd EnvironmentFile for HF + wallet paths
# -----------------------------------------------------------------------
log "profile: HF_HOME + BITTENSOR_HOME + PATH"
mkdir -p "$HF_CACHE" "$SAVE/logs" "$SAVE/logs/grafana"
install -m 0644 /dev/stdin /etc/profile.d/reliquary-paths.sh <<EOF
# Managed by deploy/bootstrap-persistent.sh — do not edit by hand.
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"
export BITTENSOR_HOME="$WALLETS"
export PATH="$VENV/bin:/srv/reliquary-inference/current/scripts:/srv/reliquary/current/scripts:\$PATH"
export PYTHONPATH="/srv/reliquary-inference/current:/srv/reliquary/current:\${PYTHONPATH:-}"
EOF

install -m 0644 /dev/stdin "$STATE/reliquary-paths.env" <<EOF
HF_HOME=$HF_CACHE
HUGGINGFACE_HUB_CACHE=$HF_CACHE/hub
TRANSFORMERS_CACHE=$HF_CACHE/transformers
BITTENSOR_HOME=$WALLETS
PATH=$VENV/bin:/srv/reliquary-inference/current/scripts:/srv/reliquary/current/scripts:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EOF

# -----------------------------------------------------------------------
# 5. Systemd units + overrides
# -----------------------------------------------------------------------
if [[ -d "$SYSTEMD_SRC" ]]; then
  log "systemd: install reliquary units from $SYSTEMD_SRC"
  shopt -s nullglob
  for unit in "$SYSTEMD_SRC"/*.service "$SYSTEMD_SRC"/*.timer "$SYSTEMD_SRC"/*.path; do
    install -m 0644 "$unit" "/etc/systemd/system/$(basename "$unit")"
  done
  shopt -u nullglob
fi

if [[ -d "$SYSTEMD_OVERRIDES" ]]; then
  log "systemd: install prometheus + grafana overrides (data → $SAVE)"
  for dir in "$SYSTEMD_OVERRIDES"/*.d; do
    [[ -d "$dir" ]] || continue
    name=$(basename "$dir")
    mkdir -p "/etc/systemd/system/$name"
    install -m 0644 "$dir"/*.conf "/etc/systemd/system/$name/"
  done
fi

systemctl daemon-reload

# -----------------------------------------------------------------------
# 6. Env-file templates (never overwrite existing)
# -----------------------------------------------------------------------
log "state: ensure env templates (non-destructive)"
for target in \
  "$STATE/reliquary-inference/reliquary-inference.env" \
  "$STATE/reliquary-inference/inference-miner.env" \
  "$STATE/reliquary-inference/inference-validator.env" \
  "$STATE/reliquary/reliquary.env"; do
  if [[ ! -f "$target" ]]; then
    mkdir -p "$(dirname "$target")"
    cat > "$target" <<EOF
# $(basename "$target") — fill in real values before systemctl enable.
# Created by bootstrap-persistent.sh on $(date -u +%Y-%m-%d).
# Never commit — holds hotkey seeds, API tokens, R2 creds, shared HMAC secret.
EOF
    chmod 600 "$target"
    chown reliquary:reliquary "$target" 2>/dev/null || true
  fi
done

# -----------------------------------------------------------------------
# 7. Prometheus + Grafana data dirs
# -----------------------------------------------------------------------
log "prometheus + grafana: ensure data dirs on $SAVE"
mkdir -p "$SAVE/prometheus/data" "$SAVE/prometheus/config" \
         "$SAVE/grafana/data" "$SAVE/grafana/provisioning/datasources" \
         "$SAVE/grafana/provisioning/dashboards" "$SAVE/grafana/dashboards" \
         "$SAVE/grafana/plugins" "$SAVE/logs/grafana"
chown -R prometheus:prometheus "$SAVE/prometheus/data" "$SAVE/prometheus/config" 2>/dev/null || true
chown -R grafana:grafana "$SAVE/grafana" "$SAVE/logs/grafana" 2>/dev/null || true

# -----------------------------------------------------------------------
# 8. Git safe.directory — repos are root-owned but different users access
# -----------------------------------------------------------------------
git config --global --add safe.directory '*' 2>/dev/null || true

log "done — $(date -u)"
printf '  venv:     %s (%s)\n' "$VENV" "$([ -x $VENV/bin/python ] && $VENV/bin/python --version 2>/dev/null || echo 'NOT BUILT')"
printf '  repos:    %s\n' "$REPOS"
printf '  wallets:  %s (%s)\n' "$WALLETS" "$(ls -1 "$WALLETS/wallets" 2>/dev/null | head -c 60 || echo 'empty — scp ~/.bittensor in')"
printf '  hf cache: %s\n' "$HF_CACHE"

cat <<NEXT

Next (operator):
  1. Create / restore wallet — either:
       btcli wallet create --wallet-name <name> --no-use-password
     OR
       scp -rp ~/.bittensor root@<host>:$SAVE/
  2. Fill env files at $STATE/*/*.env — include
     RELIQUARY_INFERENCE_SIGNING_SECRET (MUST be identical across
     miner + validator + all mesh validators).
  3. systemctl enable --now inference-miner inference-validator
  4. Monitoring (optional):
     systemctl enable --now prometheus grafana-server prometheus-node-exporter

NEXT

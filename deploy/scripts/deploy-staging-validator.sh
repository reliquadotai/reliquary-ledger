#!/usr/bin/env bash
# Deploy one Reliquary Ledger validator to a remote staging host.
# Run from the operator's laptop.
#
# Usage:
#   deploy-staging-validator.sh <ssh_user_host> <hotkey_name>
#
# Example:
#   deploy-staging-validator.sh wrk-abc123@ssh.deployments.targon.com staging1
#
# Expectations (operator-managed before running this):
#   - $HOME/.bittensor/wallets/reliquary-validator/coldkeypub.txt exists
#   - $HOME/.bittensor/wallets/reliquary-validator/hotkeys/<hotkey_name> exists
#   - $HOME/.bittensor/wallets/reliquary-validator/hotkeys/<hotkey_name>pub.txt exists
#   - A local ../.env.local with RELIQUARY_R2_CF_API_TOKEN (optional)
#   - A local /tmp/reliquary-signing-secret (shared HMAC secret — MUST match
#     the secret set on the devserver miner + validator)
#   - GRAIL repos at /Users/$USER/GRAIL/{reliquary-inference,reliquary-protocol}
#
# What this script does on the remote:
#   1. apt install python3-pip, build-essential, git, jq, tmux (if missing)
#   2. rsync reliquary-inference + reliquary-protocol into /opt/reliquary-repos
#   3. pip install -e both into the existing /opt/reliquary-venv
#   4. scp the specific hotkey + coldkeypub into /opt/reliquary-state/.bittensor
#      (never scp'd: the coldkey SEED stays on the laptop)
#   5. pipe the R2 CF API token + signing secret into /opt/reliquary-secrets
#   6. write /opt/reliquary-state/env/reliquary-inference.env
#   7. install reliquary-staging-validator.service + systemctl enable --now

set -euo pipefail

USER_HOST="${1:?usage: $0 <ssh_user_host> <hotkey_name>}"
HOTKEY_NAME="${2:?usage: $0 <ssh_user_host> <hotkey_name>}"

: "${HOME_BITTENSOR:=$HOME/.bittensor/wallets/reliquary-validator}"
: "${GRAIL_ROOT:=$HOME/GRAIL}"
: "${R2_ENV_LOCAL:=$GRAIL_ROOT/reliquary-protocol/.env.local}"
: "${SIGNING_SECRET_PATH:=/tmp/reliquary-signing-secret}"

for f in "$HOME_BITTENSOR/coldkeypub.txt" \
         "$HOME_BITTENSOR/hotkeys/$HOTKEY_NAME" \
         "$HOME_BITTENSOR/hotkeys/${HOTKEY_NAME}pub.txt"; do
  [[ -f "$f" ]] || { echo "fatal: $f missing" >&2; exit 2; }
done

SSH="ssh -o ConnectTimeout=15 -i $HOME/.ssh/private_key.pem $USER_HOST"
SCP="scp -qp -i $HOME/.ssh/private_key.pem"

echo "[deploy] 1/6 pre-create dirs on $USER_HOST"
$SSH "mkdir -p /opt/reliquary-repos /opt/reliquary-secrets /opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys /opt/reliquary-state/env /opt/reliquary-state/logs /opt/reliquary-state/artifacts /opt/reliquary-state/exports /opt/reliquary-state/hf-cache"

echo "[deploy] 2/6 tar + stream repos"
(cd "$GRAIL_ROOT" && COPYFILE_DISABLE=1 tar \
   --exclude='._*' --exclude='.DS_Store' --exclude='__pycache__' \
   --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.mypy_cache' \
   --exclude='.hypothesis' --exclude='.venv' --exclude='.env' \
   --exclude='.env.local' --exclude='.env.*.local' \
   -czf - reliquary-inference reliquary-protocol) | \
  $SSH "cd /opt/reliquary-repos && rm -rf reliquary-inference reliquary-protocol && tar -xzf - && find . -name '._*' -delete 2>/dev/null; true"

echo "[deploy] 3/6 pip install into /opt/reliquary-venv"
$SSH '
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip build-essential git jq rsync tmux curl >/dev/null 2>&1
  /opt/reliquary-venv/bin/pip install --quiet \
    -e /opt/reliquary-repos/reliquary-inference \
    -e /opt/reliquary-repos/reliquary-protocol \
    bittensor bittensor-cli bittensor-wallet \
    aiobotocore boto3 fastapi uvicorn prometheus-client python-dotenv
'

echo "[deploy] 4/6 scp hotkey $HOTKEY_NAME + coldkeypub (no coldkey seed)"
$SCP "$HOME_BITTENSOR/coldkeypub.txt" \
     "$USER_HOST:/opt/reliquary-state/.bittensor/wallets/reliquary-validator/coldkeypub.txt"
$SCP "$HOME_BITTENSOR/hotkeys/$HOTKEY_NAME" \
     "$USER_HOST:/opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys/$HOTKEY_NAME"
$SCP "$HOME_BITTENSOR/hotkeys/${HOTKEY_NAME}pub.txt" \
     "$USER_HOST:/opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys/${HOTKEY_NAME}pub.txt"

echo "[deploy] 5/6 pipe R2 token + signing secret into /opt/reliquary-secrets"
if [[ -r "$R2_ENV_LOCAL" ]]; then
  grep '^RELIQUARY_R2_CF_API_TOKEN=' "$R2_ENV_LOCAL" | cut -d= -f2- | \
    $SSH 'cat > /opt/reliquary-secrets/r2-cf-api-token && chmod 600 /opt/reliquary-secrets/r2-cf-api-token'
else
  echo "  WARN: $R2_ENV_LOCAL missing — R2 writes will fail until populated"
fi
if [[ -r "$SIGNING_SECRET_PATH" ]]; then
  $SCP "$SIGNING_SECRET_PATH" \
       "$USER_HOST:/opt/reliquary-secrets/signing-secret"
  $SSH 'chmod 600 /opt/reliquary-secrets/signing-secret'
else
  echo "  WARN: $SIGNING_SECRET_PATH missing — signatures will not verify (verdicts will be rejected as invalid_signature)"
fi

echo "[deploy] 6/6 render env, install systemd unit, start"
# Inlined so operator doesn't need another scp round. Heredoc substitutes
# $HOTKEY_NAME before it hits the remote shell.
cat > /tmp/staging-env-stub.sh <<REMOTE
#!/usr/bin/env bash
set -euo pipefail
TOKEN=\$(cat /opt/reliquary-secrets/r2-cf-api-token 2>/dev/null || echo "")
SIGNING=\$(cat /opt/reliquary-secrets/signing-secret 2>/dev/null || echo "")
cat > /opt/reliquary-state/env/reliquary-inference.env <<ENV
RELIQUARY_INFERENCE_NETUID=\${NETUID:-462}
RELIQUARY_INFERENCE_NETWORK=\${NETWORK:-test}
BT_SUBTENSOR_NETWORK=\${NETWORK:-test}
BT_WALLET_PATH=/opt/reliquary-state/.bittensor/wallets
WALLET_NAME=reliquary-validator
HOTKEY_NAME=$HOTKEY_NAME
RELIQUARY_INFERENCE_VALIDATOR_ID=\$(jq -r .ss58Address /opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys/${HOTKEY_NAME}pub.txt)
RELIQUARY_INFERENCE_STATE_ROOT=/opt/reliquary-state
RELIQUARY_INFERENCE_ARTIFACT_DIR=/opt/reliquary-state/artifacts
RELIQUARY_INFERENCE_EXPORT_DIR=/opt/reliquary-state/exports
RELIQUARY_INFERENCE_LOG_DIR=/opt/reliquary-state/logs
RELIQUARY_INFERENCE_DEVICE=cuda:0
RELIQUARY_INFERENCE_LOAD_DTYPE=bfloat16
RELIQUARY_INFERENCE_MODEL_REF=\${MODEL_REF:-Qwen/Qwen2.5-3B-Instruct}
HF_HOME=/opt/reliquary-state/hf-cache
HUGGINGFACE_HUB_CACHE=/opt/reliquary-state/hf-cache/hub
RELIQUARY_INFERENCE_TASK_SOURCE=reasoning_tasks
RELIQUARY_INFERENCE_TASK_COUNT=8
RELIQUARY_INFERENCE_POLL_INTERVAL=30
RELIQUARY_INFERENCE_SAMPLES_PER_TASK=1
RELIQUARY_INFERENCE_MAX_NEW_TOKENS=48
RELIQUARY_INFERENCE_METRICS_BIND=127.0.0.1
RELIQUARY_INFERENCE_METRICS_PORT=9108
RELIQUARY_INFERENCE_METRICS_REFRESH_INTERVAL=120
RELIQUARY_INFERENCE_METRICS_WINDOW_COUNT=1
RELIQUARY_INFERENCE_STORAGE_BACKEND=r2_rest
RELIQUARY_INFERENCE_R2_ACCOUNT_ID=\${R2_ACCOUNT_ID:-d5332aea7e3780d0f2391a4e4f6ddfbc}
RELIQUARY_INFERENCE_R2_BUCKET=\${R2_BUCKET:-reliquary}
RELIQUARY_INFERENCE_R2_PUBLIC_URL=\${R2_PUBLIC_URL:-https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev}
RELIQUARY_INFERENCE_R2_CF_API_TOKEN=\$TOKEN
RELIQUARY_INFERENCE_SIGNATURE_SCHEME=local_hmac
RELIQUARY_INFERENCE_SIGNING_SECRET=\$SIGNING
RELIQUARY_INFERENCE_USE_DRAND=false
ENV
chmod 600 /opt/reliquary-state/env/reliquary-inference.env

# Install systemd unit + enable
install -m 0644 /opt/reliquary-repos/reliquary-inference/deploy/systemd/reliquary-staging-validator.service \
  /etc/systemd/system/reliquary-staging-validator.service
systemctl daemon-reload
systemctl enable --now reliquary-staging-validator
sleep 3
echo "validator active: \$(systemctl is-active reliquary-staging-validator)"
REMOTE
$SCP /tmp/staging-env-stub.sh "$USER_HOST:/opt/reliquary-env-stub.sh"
$SSH "chmod +x /opt/reliquary-env-stub.sh && bash /opt/reliquary-env-stub.sh"

echo "[deploy] done."

#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
CF_ACCOUNT_ID="${CF_ACCOUNT_ID:-}"
CF_API_TOKEN="${CF_API_TOKEN:-}"
AUDIT_BUCKET="${AUDIT_BUCKET:-}"
AUDIT_ENDPOINT_URL="${AUDIT_ENDPOINT_URL:-}"
AUDIT_ACCESS_KEY_ID="${AUDIT_ACCESS_KEY_ID:-}"
AUDIT_SECRET_ACCESS_KEY="${AUDIT_SECRET_ACCESS_KEY:-}"
PUBLIC_AUDIT_BASE_URL="${PUBLIC_AUDIT_BASE_URL:-}"
EXPOSE_PUBLIC_ARTIFACT_URLS="${EXPOSE_PUBLIC_ARTIFACT_URLS:-false}"
RESTART_SERVICES="${RESTART_SERVICES:-false}"

if [ -z "${AUDIT_BUCKET}" ] || [ -z "${PUBLIC_AUDIT_BASE_URL}" ]; then
  echo "AUDIT_BUCKET and PUBLIC_AUDIT_BASE_URL are required." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_FILE}")"

if [ -z "${AUDIT_ACCESS_KEY_ID}" ] || [ -z "${AUDIT_SECRET_ACCESS_KEY}" ]; then
  if [ -z "${CF_ACCOUNT_ID}" ] || [ -z "${CF_API_TOKEN}" ]; then
    echo "Provide AUDIT_ACCESS_KEY_ID/AUDIT_SECRET_ACCESS_KEY or CF_ACCOUNT_ID/CF_API_TOKEN." >&2
    exit 1
  fi
  token_json="$(curl -sS -X GET "https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/tokens/verify" \
    -H "Authorization: Bearer ${CF_API_TOKEN}")"
else
  token_json='{}'
fi

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - "${ENV_FILE}" "${CF_ACCOUNT_ID}" "${CF_API_TOKEN}" "${AUDIT_BUCKET}" "${AUDIT_ENDPOINT_URL}" "${AUDIT_ACCESS_KEY_ID}" "${AUDIT_SECRET_ACCESS_KEY}" "${PUBLIC_AUDIT_BASE_URL}" "${EXPOSE_PUBLIC_ARTIFACT_URLS}" "${token_json}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
account_id = sys.argv[2]
token_value = sys.argv[3]
bucket = sys.argv[4]
endpoint_override = sys.argv[5]
access_key_id = sys.argv[6]
secret_access_key = sys.argv[7]
public_base_url = sys.argv[8]
expose_public_artifact_urls = sys.argv[9]
verify_payload = json.loads(sys.argv[10])

if not access_key_id or not secret_access_key:
    token_id = verify_payload["result"]["id"]
    access_key_id = token_id
    secret_access_key = hashlib.sha256(token_value.encode("utf-8")).hexdigest()

if endpoint_override:
    endpoint_url = endpoint_override
elif account_id:
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
else:
    raise SystemExit("AUDIT_ENDPOINT_URL is required when direct audit credentials are used without CF_ACCOUNT_ID.")

updates = {
    "RELIQUARY_INFERENCE_AUDIT_BUCKET": bucket,
    "RELIQUARY_INFERENCE_AUDIT_ENDPOINT_URL": endpoint_url,
    "RELIQUARY_INFERENCE_AUDIT_ACCESS_KEY_ID": access_key_id,
    "RELIQUARY_INFERENCE_AUDIT_SECRET_ACCESS_KEY": secret_access_key,
    "RELIQUARY_INFERENCE_PUBLIC_AUDIT_BASE_URL": public_base_url,
    "RELIQUARY_INFERENCE_EXPOSE_PUBLIC_ARTIFACT_URLS": expose_public_artifact_urls.lower(),
}

lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
seen: set[str] = set()
rendered: list[str] = []

for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        rendered.append(line)
        continue
    key, _, _value = line.partition("=")
    if key in updates:
        rendered.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        rendered.append(line)

for key, value in updates.items():
    if key not in seen:
        rendered.append(f"{key}={value}")

env_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
PY

chmod 600 "${ENV_FILE}"
echo "Applied audit profile to ${ENV_FILE}"

if [ "${RESTART_SERVICES}" = "true" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart inference-miner.service inference-validator.service
  systemctl --no-pager --full status inference-miner.service inference-validator.service || true
fi

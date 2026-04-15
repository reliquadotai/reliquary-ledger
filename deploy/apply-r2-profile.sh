#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
CF_ACCOUNT_ID="${CF_ACCOUNT_ID:-}"
CF_API_TOKEN="${CF_API_TOKEN:-}"
R2_BUCKET="${R2_BUCKET:-reliquary}"
R2_ENDPOINT_URL="${R2_ENDPOINT_URL:-}"
RESTART_SERVICES="${RESTART_SERVICES:-false}"

if [ -z "${CF_ACCOUNT_ID}" ] || [ -z "${CF_API_TOKEN}" ]; then
  echo "CF_ACCOUNT_ID and CF_API_TOKEN are required." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENV_FILE}")"

if [ -f "${ENV_FILE}" ]; then
  cp "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
fi

token_json="$(curl -sS -X GET "https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT_ID}/tokens/verify" \
  -H "Authorization: Bearer ${CF_API_TOKEN}")"

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - "${ENV_FILE}" "${CF_ACCOUNT_ID}" "${CF_API_TOKEN}" "${R2_BUCKET}" "${R2_ENDPOINT_URL}" "${token_json}" <<'PY'
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
verify_payload = json.loads(sys.argv[6])

token_id = verify_payload["result"]["id"]
derived_secret = hashlib.sha256(token_value.encode("utf-8")).hexdigest()
endpoint_url = endpoint_override or f"https://{account_id}.r2.cloudflarestorage.com"

updates = {
    "RELIQUARY_INFERENCE_STORAGE_BACKEND": "r2",
    "RELIQUARY_INFERENCE_R2_BUCKET": bucket,
    "RELIQUARY_INFERENCE_R2_ENDPOINT_URL": endpoint_url,
    "RELIQUARY_INFERENCE_R2_ACCESS_KEY_ID": token_id,
    "RELIQUARY_INFERENCE_R2_SECRET_ACCESS_KEY": derived_secret,
}
updates["RELIQUARY_INFERENCE_EXPOSE_PUBLIC_ARTIFACT_URLS"] = "false"

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
echo "Applied R2 profile to ${ENV_FILE}"

if [ "${RESTART_SERVICES}" = "true" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl restart inference-miner.service inference-validator.service
  systemctl --no-pager --full status inference-miner.service inference-validator.service || true
fi

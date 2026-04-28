#!/usr/bin/env bash
# Upload the static public status page to the audit R2 bucket.
#
# Run this on a host with R2 / aws-s3 credentials configured, or over a
# Cloudflare API token via the apply-audit-profile.sh flow.
#
# Usage:
#   bash scripts/upload_public_status.sh
#
# Variables (from .env or environment):
#   RELIQUARY_INFERENCE_AUDIT_BUCKET — destination bucket name (required)
#   RELIQUARY_INFERENCE_AUDIT_ENDPOINT_URL — R2 endpoint (required)
#   RELIQUARY_INFERENCE_AUDIT_ACCESS_KEY_ID — R2 key (required)
#   RELIQUARY_INFERENCE_AUDIT_SECRET_ACCESS_KEY — R2 secret (required)
#
# Output: public page lands at <PUBLIC_AUDIT_BASE_URL>/status/index.html
#
# This is a one-shot uploader. For continuous freshness, add it to a cron
# or systemd timer (e.g. every 5 min). The page itself polls the public
# audit JSON every 60 s in-browser, so per-second freshness is not the
# upload cadence — this is just for landing-page changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE="${INSTALL_DIR}/dashboard/public/index.html"

if [[ -f "${INSTALL_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "${INSTALL_DIR}/.env"
  set +a
fi

: "${RELIQUARY_INFERENCE_AUDIT_BUCKET:?required}"
: "${RELIQUARY_INFERENCE_AUDIT_ENDPOINT_URL:?required}"
: "${RELIQUARY_INFERENCE_AUDIT_ACCESS_KEY_ID:?required}"
: "${RELIQUARY_INFERENCE_AUDIT_SECRET_ACCESS_KEY:?required}"

if [[ ! -f "${SOURCE}" ]]; then
  echo "source page missing: ${SOURCE}" >&2
  exit 1
fi

# Use aws CLI if available; else fall back to curl + s3 sigv4 (omitted for brevity).
if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not installed; install with 'pip install awscli' or 'brew install awscli'" >&2
  exit 1
fi

export AWS_ACCESS_KEY_ID="${RELIQUARY_INFERENCE_AUDIT_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${RELIQUARY_INFERENCE_AUDIT_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="auto"

echo "uploading ${SOURCE} -> s3://${RELIQUARY_INFERENCE_AUDIT_BUCKET}/status/index.html"
aws s3 cp "${SOURCE}" "s3://${RELIQUARY_INFERENCE_AUDIT_BUCKET}/status/index.html" \
  --endpoint-url "${RELIQUARY_INFERENCE_AUDIT_ENDPOINT_URL}" \
  --content-type "text/html; charset=utf-8" \
  --cache-control "public, max-age=60"

echo "done. public URL: ${RELIQUARY_INFERENCE_PUBLIC_AUDIT_BASE_URL:-<PUBLIC_AUDIT_BASE_URL>}/status/index.html"

#!/usr/bin/env bash
set -euo pipefail

PROMETHEUS_URL="${PROMETHEUS_URL:-http://127.0.0.1:9090}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:9108/metrics}"

echo "== reliquary runtime status =="
bash "$(dirname "$0")/runtime-status.sh"

echo
echo "== service health =="
systemctl --no-pager --full --lines=0 status \
  inference-miner.service \
  inference-validator.service \
  reliquary-metrics-exporter.service \
  prometheus-node-exporter \
  prometheus \
  grafana-server || true

echo
echo "== metrics endpoint =="
curl -fsSL "${METRICS_URL}" | grep -E 'reliquary_(latest_window_mined|latest_weight_publication_window|audit_publish_age_seconds|chain_scrape_age_seconds)' || true

echo
echo "== prometheus target health =="
curl -fsSL "${PROMETHEUS_URL}/api/v1/targets?state=active" | jq '.data.activeTargets[] | {job: .labels.job, health: .health, scrapeUrl: .scrapeUrl, lastError: .lastError}'

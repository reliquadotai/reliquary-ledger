#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Run as root." >&2
  exit 1
fi

export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
apt-get update
apt-get install -y curl jq ca-certificates gnupg apt-transport-https prometheus prometheus-node-exporter

if ! command -v grafana-server >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  if [ ! -f /etc/apt/keyrings/grafana.gpg ]; then
    curl -fsSL https://apt.grafana.com/gpg.key | gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
  fi
  cat >/etc/apt/sources.list.d/grafana.list <<'EOF'
deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main
EOF
  apt-get update
  apt-get install -y grafana
fi

install -d -m 0755 /var/lib/node_exporter/textfile_collector
install -d -m 0755 /etc/prometheus/rules.d
install -d -m 0755 /var/lib/grafana/dashboards/reliquary
install -d -m 0755 /etc/grafana/provisioning/datasources
install -d -m 0755 /etc/grafana/provisioning/dashboards

echo "Monitoring packages installed."

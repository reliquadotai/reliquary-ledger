#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
  echo "Run as root." >&2
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/srv/reliquary-inference/current}"
STATE_ROOT="${STATE_ROOT:-/srv/reliquary-inference/state}"
ENV_FILE="${ENV_FILE:-${STATE_ROOT}/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
PROMETHEUS_BIND="${PROMETHEUS_BIND:-127.0.0.1:9090}"
NODE_EXPORTER_BIND="${NODE_EXPORTER_BIND:-127.0.0.1:9100}"
GRAFANA_BIND="${GRAFANA_BIND:-127.0.0.1:3000}"
METRICS_BIND="${METRICS_BIND:-127.0.0.1}"
METRICS_PORT="${METRICS_PORT:-9108}"
CONTROL_PLANE_TARGET="${CONTROL_PLANE_TARGET:-127.0.0.1:8000}"
TEXTFILE_DIR="${TEXTFILE_DIR:-/var/lib/node_exporter/textfile_collector}"
RESTART_SERVICES="${RESTART_SERVICES:-true}"

python3 - <<PY
from pathlib import Path

path = Path("${ENV_FILE}")
text = path.read_text(encoding="utf-8") if path.exists() else ""
lines = [line for line in text.splitlines() if line.strip()]
updates = {
    "RELIQUARY_INFERENCE_METRICS_BIND": "${METRICS_BIND}",
    "RELIQUARY_INFERENCE_METRICS_PORT": "${METRICS_PORT}",
    "RELIQUARY_INFERENCE_METRICS_REFRESH_INTERVAL": "15",
}
existing = {}
for line in lines:
    if "=" in line:
        key, value = line.split("=", 1)
        existing[key] = value
existing.update(updates)
path.write_text("".join(f"{key}={value}\n" for key, value in sorted(existing.items())), encoding="utf-8")
PY

cat >/etc/default/prometheus-node-exporter <<EOF
ARGS="--web.listen-address=${NODE_EXPORTER_BIND} --collector.textfile.directory=${TEXTFILE_DIR}"
EOF

cat >/etc/default/prometheus <<EOF
ARGS="--config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus --web.listen-address=${PROMETHEUS_BIND} --web.enable-lifecycle"
EOF

python3 - <<PY
from pathlib import Path

template = Path("${ROOT_DIR}/deploy/monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8")
template = template.replace("__METRICS_TARGET__", "${METRICS_BIND}:${METRICS_PORT}")
template = template.replace("__CONTROL_PLANE_TARGET__", "${CONTROL_PLANE_TARGET}")
Path("/etc/prometheus/prometheus.yml").write_text(template, encoding="utf-8")
PY
install -m 0644 "${ROOT_DIR}/deploy/monitoring/prometheus/alerts.yml" /etc/prometheus/rules.d/reliquary-alerts.yml

install -m 0755 -d /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards /var/lib/grafana/dashboards/reliquary
install -m 0644 "${ROOT_DIR}/deploy/monitoring/grafana/provisioning/datasources/prometheus.yml" /etc/grafana/provisioning/datasources/reliquary-prometheus.yml
install -m 0644 "${ROOT_DIR}/deploy/monitoring/grafana/provisioning/dashboards/reliquary.yml" /etc/grafana/provisioning/dashboards/reliquary.yml
install -m 0644 "${ROOT_DIR}/deploy/monitoring/grafana/dashboards/"*.json /var/lib/grafana/dashboards/reliquary/

promtool check config /etc/prometheus/prometheus.yml
promtool check rules /etc/prometheus/rules.d/reliquary-alerts.yml

python3 - <<PY
from pathlib import Path

grafana_ini = Path("/etc/grafana/grafana.ini")
text = grafana_ini.read_text(encoding="utf-8")
replacements = {
    ";http_addr =": "http_addr = 127.0.0.1",
    ";http_port = 3000": "http_port = 3000",
    ";domain = localhost": "domain = localhost",
}
for old, new in replacements.items():
    text = text.replace(old, new)
if "http_addr = 127.0.0.1" not in text and "[server]" in text:
    text = text.replace("[server]", "[server]\nhttp_addr = 127.0.0.1\nhttp_port = 3000", 1)
grafana_ini.write_text(text, encoding="utf-8")
PY

install -m 0644 "${ROOT_DIR}/deploy/systemd/reliquary-metrics-exporter.service" /etc/systemd/system/reliquary-metrics-exporter.service
install -m 0644 "${ROOT_DIR}/deploy/systemd/reliquary-gpu-metrics.service" /etc/systemd/system/reliquary-gpu-metrics.service
install -m 0644 "${ROOT_DIR}/deploy/systemd/reliquary-gpu-metrics.timer" /etc/systemd/system/reliquary-gpu-metrics.timer
install -m 0755 "${ROOT_DIR}/deploy/monitoring/bin/collect-gpu-metrics.sh" /usr/local/bin/reliquary-collect-gpu-metrics

python3 - <<PY
from pathlib import Path

path = Path("/etc/systemd/system/reliquary-gpu-metrics.service")
text = path.read_text(encoding="utf-8").replace(
    "/srv/reliquary-inference/current/deploy/monitoring/bin/collect-gpu-metrics.sh",
    "/usr/local/bin/reliquary-collect-gpu-metrics",
)
path.write_text(text, encoding="utf-8")
PY

systemctl daemon-reload
systemctl enable --now prometheus-node-exporter prometheus grafana-server reliquary-metrics-exporter.service reliquary-gpu-metrics.timer

if [ "${RESTART_SERVICES}" = "true" ]; then
  systemctl restart prometheus-node-exporter prometheus grafana-server reliquary-metrics-exporter.service
  systemctl start reliquary-gpu-metrics.service
fi

systemctl --no-pager --full status prometheus-node-exporter prometheus grafana-server reliquary-metrics-exporter.service reliquary-gpu-metrics.timer || true

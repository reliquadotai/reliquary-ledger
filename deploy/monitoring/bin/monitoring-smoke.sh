#!/usr/bin/env bash
# Tier 2 Epic 5 acceptance smoke: verify the local monitoring stack is healthy.
#
# Checks:
#   - Prometheus /-/healthy returns 200
#   - Grafana /api/health returns 200
#   - Jaeger UI root returns 200
#   - Prometheus can scrape a supplied --metrics-url endpoint once (optional).
#
# Usage:
#   monitoring-smoke.sh [--metrics-url http://host:port/metrics]
#
# Exits non-zero on any failure so CI / operators can gate rollout.

set -euo pipefail

PROM_URL="${PROM_URL:-http://127.0.0.1:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://127.0.0.1:3000}"
JAEGER_URL="${JAEGER_URL:-http://127.0.0.1:16686}"
METRICS_URL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --metrics-url)
            METRICS_URL="$2"
            shift 2
            ;;
        --prom-url)
            PROM_URL="$2"
            shift 2
            ;;
        --grafana-url)
            GRAFANA_URL="$2"
            shift 2
            ;;
        --jaeger-url)
            JAEGER_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "monitoring-smoke.sh [--metrics-url URL] [--prom-url URL] [--grafana-url URL] [--jaeger-url URL]"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

check() {
    local label="$1"
    local url="$2"
    local expected="${3:-200}"
    local code
    code=$(curl -fsS -o /dev/null -w "%{http_code}" --max-time 10 "$url" || echo "000")
    if [[ "$code" != "$expected" ]]; then
        echo "FAIL ${label}: ${url} returned ${code} (expected ${expected})"
        return 1
    fi
    echo "OK   ${label}: ${url} returned ${code}"
}

failures=0

check "prometheus"  "${PROM_URL}/-/healthy"          || failures=$((failures + 1))
check "prom-ready"  "${PROM_URL}/-/ready"            || failures=$((failures + 1))
check "grafana"     "${GRAFANA_URL}/api/health"      || failures=$((failures + 1))
check "jaeger"      "${JAEGER_URL}/"                 || failures=$((failures + 1))

# Prometheus list-datasources sanity (checks Grafana provisioning picked it up).
if ! check "grafana-prom-datasource" \
        "${GRAFANA_URL}/api/datasources/name/Prometheus" 200; then
    echo "  note: Grafana admin auth required when basic-auth is on; treat as warning"
fi

if [[ -n "${METRICS_URL}" ]]; then
    if ! curl -fsS --max-time 10 "${METRICS_URL}" | head -n 20 >/dev/null; then
        echo "FAIL metrics-endpoint: ${METRICS_URL} not scrapable"
        failures=$((failures + 1))
    else
        echo "OK   metrics-endpoint: ${METRICS_URL} scrape returned data"
    fi
fi

if [[ ${failures} -ne 0 ]]; then
    echo
    echo "monitoring smoke FAILED (${failures} failure(s))"
    exit 1
fi

echo
echo "monitoring smoke PASSED"

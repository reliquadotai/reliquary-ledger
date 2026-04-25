# Monitoring

Reliquary uses a private-first single-node monitoring stack:

- Prometheus on `127.0.0.1:9090`
- Grafana on `127.0.0.1:3000`
- `node_exporter` on `127.0.0.1:9100`
- `reliquary-inference metrics-exporter` on `127.0.0.1:9108`
- host GPU and service restart metrics through the node exporter textfile collector

The public audit surface stays separate. Monitoring is not exposed publicly.

## Install

```bash
sudo bash deploy/install-monitoring.sh
sudo bash deploy/apply-monitoring-profile.sh
```

## Operator Path

Use this order on the RTX staging node:

1. `bash deploy/bootstrap-node.sh`
2. `EXTRAS=dev,gpu bash deploy/install-environment.sh`
3. `bash deploy/create-wallet.sh`
4. `bash deploy/write-service-envs.sh`
5. `sudo RESTART_SERVICES=true bash deploy/apply-rtx-real-model-profile.sh`
6. `CHAIN_ENDPOINT=wss://your-endpoint.example bash deploy/apply-chain-endpoint-profile.sh`
7. choose private artifact storage:
   - `sudo RESTART_SERVICES=true bash deploy/apply-local-storage-profile.sh`
   - or `CF_ACCOUNT_ID=... CF_API_TOKEN=... R2_BUCKET=private-artifacts bash deploy/apply-r2-profile.sh`
8. configure public audit publishing:
   - `CF_ACCOUNT_ID=... CF_API_TOKEN=... AUDIT_BUCKET=public-audit PUBLIC_AUDIT_BASE_URL=https://audit.example.com bash deploy/apply-audit-profile.sh`
9. `sudo bash deploy/install-monitoring.sh`
10. `sudo bash deploy/apply-monitoring-profile.sh`
11. `bash deploy/runtime-health.sh`
12. `bash deploy/testnet-readonly-smoke.sh`
13. `bash deploy/real-model-readonly-smoke.sh`
14. `sudo CREATE_SUBNET=false ENABLE_SERVICES=true bash deploy/activate-testnet.sh`

## SSH Tunnels

Use SSH tunnels instead of public ports:

```bash
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 -L 9108:127.0.0.1:9108 <node>
```

Then open:

- `http://127.0.0.1:3000` for Grafana
- `http://127.0.0.1:9090` for Prometheus
- `http://127.0.0.1:9108/metrics` for raw exporter output

## Dashboards

- `Reliquary Live Ops`
- `Reliquary Host And GPU`
- `Reliquary Chain State`
- `Reliquary Platform`

The platform dashboard is expected to show partial data unless the broader Reliquary control plane is running on the same host.

## Runtime Health

```bash
bash deploy/runtime-health.sh
```

This combines:

- `reliquary-inference status`
- systemd health for miner, validator, metrics exporter, Prometheus, Grafana, and node exporter
- raw metrics freshness
- Prometheus scrape target health

The exporter also surfaces:

- latest finalized window available for import
- import lag between mined and finalized windows
- task-source submitted and accepted totals
- rolling reasoning correctness, format, and policy-compliance rates

## Rollback

- revert the model profile by restoring the previous env backup under `/srv/reliquary-inference/state`
- unset `RELIQUARY_INFERENCE_CHAIN_ENDPOINT` and `BT_SUBTENSOR_CHAIN_ENDPOINT` to return to the network default endpoint
- rerun the storage profile scripts with the intended local or R2 settings
- stop monitoring cleanly with:

```bash
sudo systemctl stop reliquary-metrics-exporter.service reliquary-gpu-metrics.timer prometheus prometheus-node-exporter grafana-server
```

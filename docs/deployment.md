# Deployment

`reliquary-inference` is designed for host-level Python services. The default public posture is:

- private artifact storage
- separate public audit publishing
- one miner service
- one validator service
- one clearly documented operator path

## Local Development

```bash
cp env.example .env
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
reliquary-inference demo-local
```

## Golden Path

Use this order for a clean staging or testnet bring-up:

1. `bash deploy/bootstrap-node.sh`
2. `EXTRAS=dev,gpu bash deploy/install-environment.sh`
3. `bash deploy/create-wallet.sh`
4. `bash deploy/write-service-envs.sh`
5. `sudo RESTART_SERVICES=true bash deploy/apply-rtx-real-model-profile.sh`
6. `CHAIN_ENDPOINT=wss://your-endpoint.example bash deploy/apply-chain-endpoint-profile.sh`
7. choose one private artifact path:
   - `sudo RESTART_SERVICES=true bash deploy/apply-local-storage-profile.sh`
   - or `CF_ACCOUNT_ID=... CF_API_TOKEN=... R2_BUCKET=private-artifacts bash deploy/apply-r2-profile.sh`
8. configure a public audit target:
   - `AUDIT_BUCKET=public-audit AUDIT_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com AUDIT_ACCESS_KEY_ID=... AUDIT_SECRET_ACCESS_KEY=... PUBLIC_AUDIT_BASE_URL=https://audit.example.com bash deploy/apply-audit-profile.sh`
   - or `CF_ACCOUNT_ID=... CF_API_TOKEN=... AUDIT_BUCKET=public-audit PUBLIC_AUDIT_BASE_URL=https://audit.example.com bash deploy/apply-audit-profile.sh`
9. `bash deploy/runtime-status.sh`
10. `bash deploy/testnet-readonly-smoke.sh`
11. `bash deploy/real-model-readonly-smoke.sh`
12. `sudo bash deploy/install-monitoring.sh`
13. `sudo bash deploy/apply-monitoring-profile.sh`
14. `bash deploy/runtime-health.sh`
15. `sudo CREATE_SUBNET=false ENABLE_SERVICES=true bash deploy/activate-testnet.sh`

If you are switching an existing node from R2-backed artifacts to local private artifacts, mirror the current window state first:

```bash
bash deploy/mirror-r2-state-to-local.sh
```

## Runtime Profiles

### Real-model RTX baseline

```bash
sudo RESTART_SERVICES=true bash deploy/apply-rtx-real-model-profile.sh
```

This applies the current recommended baseline:

- `Qwen/Qwen2.5-3B-Instruct`
- `cuda`
- `bf16`
- `math` task source (live; `RELIQUARY_INFERENCE_MATH_MAX_LEVEL=2` for bootstrap)
- `10` second poll interval

### Dedicated chain endpoint

```bash
CHAIN_ENDPOINT=wss://your-endpoint.example bash deploy/apply-chain-endpoint-profile.sh
```

Use this when you want to reduce pressure on public websocket infrastructure and keep long-running services stable.

### Private artifact storage

For a private single-node staging baseline:

```bash
sudo RESTART_SERVICES=true bash deploy/apply-local-storage-profile.sh
```

This keeps artifacts on-node and leaves public publishing to the dedicated audit target.

### Private artifact bucket

```bash
CF_ACCOUNT_ID=... CF_API_TOKEN=... R2_BUCKET=private-artifacts bash deploy/apply-r2-profile.sh
```

This configures the primary artifact registry.

### Separate public audit bucket

```bash
AUDIT_BUCKET=public-audit AUDIT_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com AUDIT_ACCESS_KEY_ID=... AUDIT_SECRET_ACCESS_KEY=... PUBLIC_AUDIT_BASE_URL=https://audit.example.com bash deploy/apply-audit-profile.sh
```

Or derive audit credentials from a Cloudflare token at deploy time:

```bash
CF_ACCOUNT_ID=... CF_API_TOKEN=... AUDIT_BUCKET=public-audit PUBLIC_AUDIT_BASE_URL=https://audit.example.com bash deploy/apply-audit-profile.sh
```

This configures a dedicated audit publishing target and keeps raw artifact links hidden by default.

## Services

- `inference-miner.service`
- `inference-validator.service`

The base env file is:

- `/srv/reliquary-inference/state/reliquary-inference.env`

Per-service override files are:

- `/srv/reliquary-inference/state/inference-miner.env`
- `/srv/reliquary-inference/state/inference-validator.env`

## Health And Status

Use:

```bash
bash deploy/runtime-status.sh
```

The runtime status path summarizes:

- latest window mined
- latest weight publication
- current model
- current network
- current storage and bucket mode
- chain endpoint mode
- public audit target and bucket selection

## Monitoring

Use the dedicated monitoring runbook for the full Grafana and Prometheus setup:

- [monitoring.md](monitoring.md)

The supported baseline is:

- Prometheus on `127.0.0.1:9090`
- Grafana on `127.0.0.1:3000`
- node exporter on `127.0.0.1:9100`
- `reliquary-inference metrics-exporter` on `127.0.0.1:9108`

Access those services through SSH tunneling rather than public exposure.

## Rollback

Use these rollback moves when a profile or config change misbehaves:

- revert model profile:
  - restore the previous env backup from `/srv/reliquary-inference/state/reliquary-inference.env.bak.*`
- revert chain endpoint:
  - unset `RELIQUARY_INFERENCE_CHAIN_ENDPOINT` and `BT_SUBTENSOR_CHAIN_ENDPOINT`, then restart services
- revert bucket config:
  - restore the previous env backup or rerun the profile scripts with the intended values
- stop services safely:
  - `systemctl stop inference-miner.service inference-validator.service`

## Public And Private Boundaries

- keep raw artifact buckets private by default
- publish only the audit index publicly unless you explicitly want raw object exposure
- do not commit wallet names, hostnames, addresses, or runtime secrets into the repo

## Release Checklist

Use [release-checklist.md](release-checklist.md) before widening operator access or public visibility.

# Validator Quickstart — Reliquary Ledger (testnet netuid 462)

A validator runs the nine-stage verifier pipeline over miner completions,
publishes signed verdicts + scorecards to R2, and sets weights on-chain.
Validators also run `policy_consumer` to hot-swap Forge's policy deltas.

## Requirements

- Linux host with CUDA GPU (same model family as miners; cross-GPU
  determinism audit covers RTX 6000B ↔ H100)
- Python 3.12 + uv
- Bittensor wallet registered on netuid 462
- R2 bucket shared with the mesh
- Shared HMAC signing secret (coordinate with mesh operator)

## Clone + install

```bash
git clone https://github.com/reliquadotai/reliquary-inference.git
cd reliquary-inference
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,gpu]"
cp env.example .env
```

## Configure

Edit `.env`:

```
# Chain
RELIQUARY_INFERENCE_NETUID=462
RELIQUARY_INFERENCE_NETWORK=test
BT_SUBTENSOR_NETWORK=test
WALLET_NAME=reliquary-validator
HOTKEY_NAME=default

# Task source must match every other validator
RELIQUARY_INFERENCE_TASK_SOURCE=math
RELIQUARY_INFERENCE_TASK_COUNT=1
RELIQUARY_INFERENCE_SAMPLES_PER_TASK=8
RELIQUARY_INFERENCE_MAX_NEW_TOKENS=1024
RELIQUARY_INFERENCE_MATH_MAX_LEVEL=2

# Model (must match miners)
RELIQUARY_INFERENCE_MODEL_REF=Qwen/Qwen2.5-3B-Instruct
RELIQUARY_INFERENCE_DEVICE=cuda
RELIQUARY_INFERENCE_LOAD_DTYPE=bfloat16

# Zone filter + cooldown
RELIQUARY_INFERENCE_ZONE_FILTER_BOOTSTRAP=1
RELIQUARY_INFERENCE_COOLDOWN_WINDOWS=50
RELIQUARY_INFERENCE_COOLDOWN_R2_KEY=cooldown/cooldown.json

# Batched proof verification
RELIQUARY_INFERENCE_BATCHED_VERIFY=true
RELIQUARY_INFERENCE_BATCHED_VERIFY_MAX_SIZE=8

# Backfill + metrics
RELIQUARY_INFERENCE_VALIDATOR_BACKFILL_HORIZON_WINDOWS=10
RELIQUARY_INFERENCE_METRICS_WINDOW_COUNT=10
RELIQUARY_INFERENCE_METRICS_REFRESH_INTERVAL=30

# Storage
RELIQUARY_INFERENCE_STORAGE_BACKEND=r2_rest
RELIQUARY_INFERENCE_R2_ACCOUNT_ID=<cf-account>
RELIQUARY_INFERENCE_R2_BUCKET=reliquary
RELIQUARY_INFERENCE_R2_CF_API_TOKEN=<cf-token>

# Closed-loop bridge
RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED=true
RELIQUARY_INFERENCE_POLICY_AUTHORITY_HOTKEY=reliquary-policy-authority-v1
RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET=<shared>
RELIQUARY_INFERENCE_TRAINING_NETUID=462

# Signing
RELIQUARY_INFERENCE_SIGNATURE_SCHEME=local_hmac
RELIQUARY_INFERENCE_SIGNING_SECRET=<shared-mesh-hmac-secret>
RELIQUARY_INFERENCE_VALIDATOR_ID=<your-hotkey>
```

## Register

```bash
btcli subnet register --netuid 462 --network test \
  --wallet.name reliquary-validator --wallet.hotkey default
```

You need enough test TAO to cover the burn price + stake. See
[btcli docs](https://docs.bittensor.com/getting-started/testnet-faucet)
for the testnet faucet.

## Smoke test

```bash
bash deploy/testnet-readonly-smoke.sh
```

Reads chain, metagraph, and the most recent window bundle without any
on-chain writes. Expect a summary with `window_id` + mesh peer count.

## Run the validator service

```bash
sudo cp deploy/systemd/reliquary-ledger-validator.service /etc/systemd/system/
sudo sed -i 's|/path/to/.env|'"$PWD"'/.env|' \
  /etc/systemd/system/reliquary-ledger-validator.service
sudo systemctl daemon-reload
sudo systemctl enable --now reliquary-ledger-validator
sudo journalctl -u reliquary-ledger-validator -f
```

## Run the metrics exporter

```bash
sudo cp deploy/systemd/reliquary-metrics-exporter.service \
  /etc/systemd/system/
sudo systemctl enable --now reliquary-metrics-exporter
```

Endpoints (localhost by default; bind to `0.0.0.0` to expose):

- `GET /healthz` — liveness probe + zone metrics
- `GET /status` — full operator state JSON
- `GET /metrics` — Prometheus text
- `GET /` — static HTML dashboard (see [dashboard.md](dashboard.md))

## Run the audit index timer

```bash
sudo cp deploy/systemd/reliquary-audit-index.service /etc/systemd/system/
sudo cp deploy/systemd/reliquary-audit-index.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now reliquary-audit-index.timer
```

Rebuilds `{export_dir}/audit/index.json` every 10 min. The metrics exporter
reads that file to populate the rolling gauges.

## Joining the mesh

Coordinate with an existing mesh operator to:

1. Share the `RELIQUARY_INFERENCE_SIGNING_SECRET` (HMAC secret used for
   mesh verdicts).
2. Share the `RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET` (used to verify
   Forge's `CheckpointAttestation`).
3. Get added to the R2 bucket's access list.

Once your validator is running, the other mesh members will see your
verdicts under `verdict_bundles/window-NNNNNNNN/<your-hotkey>.json.gz`
on the next window.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Validator hangs at "activating" for > 10 min | R2 rate-limit on initial scorecard scan | Bump timeout drop-in to 30 min: `/etc/systemd/system/reliquary-ledger-validator.service.d/timeout.conf` with `[Service]\nTimeoutStartSec=1800` |
| `published weights for window X: {'miner': 0.0}` | Miner got 0/M correct on this window | Expected on hard windows; next windows should self-correct |
| `invalid_signature` hard-fail on every verdict | Mesh HMAC secret mismatch | Sync with mesh operator |
| Zone gauges stuck at 0 in `/status` | `METRICS_WINDOW_COUNT=0` (old workaround) | Set `RELIQUARY_INFERENCE_METRICS_WINDOW_COUNT=10` + restart exporter |
| `reparam_guard:projection_magnitude_below_floor` on policy apply | Forge published a suspicious delta | Good — that's the guard working. Contact trainer operator. |

See [monitoring.md](monitoring.md) for Prometheus/Grafana wiring and
[incentive.md](incentive.md) for the weight-derivation formula.

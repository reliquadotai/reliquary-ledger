# Miner Quickstart — Reliquary Ledger (testnet netuid 462)

A miner on Reliquary Ledger takes deterministic task batches, produces
M=8 sampled rollouts per prompt with GRAIL sketch commitments + HMAC
signatures, and uploads completion bundles to R2 for the validator mesh
to verify.

## Requirements

- Linux host with CUDA GPU (H100 / RTX 6000 Blackwell tested; smaller
  GPUs fine for small models)
- Python 3.12
- [uv](https://github.com/astral-sh/uv) (`pip install uv`)
- Bittensor wallet with some test TAO (testnet faucet)
- ~20 GB free disk for model cache + state
- R2 bucket + Cloudflare API token (shared across the mesh)

## Clone + install

```bash
git clone https://github.com/0xgrizz/reliquary-inference.git
cd reliquary-inference
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,gpu]"
```

## Configure

```bash
cp env.example .env
```

Edit `.env` — at minimum:

```
# Chain
RELIQUARY_INFERENCE_NETUID=462
RELIQUARY_INFERENCE_NETWORK=test
BT_SUBTENSOR_NETWORK=test
WALLET_NAME=reliquary-miner
HOTKEY_NAME=default
BT_WALLET_PATH=~/.bittensor/wallets

# Task source (live)
RELIQUARY_INFERENCE_TASK_SOURCE=math
RELIQUARY_INFERENCE_TASK_COUNT=1
RELIQUARY_INFERENCE_SAMPLES_PER_TASK=8
RELIQUARY_INFERENCE_MAX_NEW_TOKENS=1024
RELIQUARY_INFERENCE_MATH_MAX_LEVEL=2

# Miner
RELIQUARY_INFERENCE_MINER_ID=<your-miner-id>
RELIQUARY_INFERENCE_MODEL_REF=Qwen/Qwen2.5-3B-Instruct
RELIQUARY_INFERENCE_DEVICE=cuda
RELIQUARY_INFERENCE_LOAD_DTYPE=bfloat16
RELIQUARY_INFERENCE_GENERATION_TEMPERATURE=0.9
RELIQUARY_INFERENCE_GENERATION_TOP_P=1.0

# Storage (share R2 creds with the validator mesh)
RELIQUARY_INFERENCE_STORAGE_BACKEND=r2_rest
RELIQUARY_INFERENCE_R2_ACCOUNT_ID=<cf-account>
RELIQUARY_INFERENCE_R2_BUCKET=reliquary
RELIQUARY_INFERENCE_R2_CF_API_TOKEN=<cf-token>
RELIQUARY_INFERENCE_R2_PUBLIC_URL=https://pub-<hash>.r2.dev

# Bridge (mandatory — synchronised with the validator mesh)
RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED=true
RELIQUARY_INFERENCE_POLICY_AUTHORITY_HOTKEY=reliquary-policy-authority-v1
RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET=<shared-hmac-secret>
RELIQUARY_INFERENCE_TRAINING_NETUID=462

# Signing (mesh HMAC secret)
RELIQUARY_INFERENCE_SIGNATURE_SCHEME=local_hmac
RELIQUARY_INFERENCE_SIGNING_SECRET=<shared-mesh-hmac-secret>
```

## Register on the subnet

```bash
btcli subnet register --netuid 462 --network test \
  --wallet.name reliquary-miner --wallet.hotkey default
```

## Run one window locally (smoke test)

```bash
reliquary-inference mine-window --once
```

Expect: `mined 8 completions for window <ledger_window>` within ~3-5 min.

## Run as a systemd service (long-running)

Use `deploy/systemd/` templates:

```bash
sudo cp deploy/systemd/reliquary-ledger-miner.service /etc/systemd/system/
sudo sed -i 's|/path/to/.env|'"$PWD"'/.env|' /etc/systemd/system/reliquary-ledger-miner.service
sudo systemctl daemon-reload
sudo systemctl enable --now reliquary-ledger-miner
sudo journalctl -u reliquary-ledger-miner -f
```

## Verify liveness

Once the miner has submitted completions, check any validator's `/healthz`:

```bash
curl -s http://<validator-host>:9108/healthz | jq .latest_window_mined
```

Should advance by 1 every 6 min (30-block window cadence × 12 s block time).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `miner loop error: 429` | R2 rate-limit burst | Already retried with exp backoff; wait 60 s. If persistent, share the R2 quota with fewer miners. |
| All M rollouts identical (sample_index 0-7 same text) | Greedy decode (T=0) | Set `RELIQUARY_INFERENCE_GENERATION_TEMPERATURE=0.9`. |
| `termination_no_eos` rejects every rollout | `MAX_NEW_TOKENS` too small | Bump to 1024 for MATH. |
| `invalid_signature` hard-fails | Wrong `SIGNING_SECRET` vs the mesh | Sync with the mesh operator. |
| Weight drops to 0 | Miner's answers are all wrong | Expected on Level 4-5; set `MATH_MAX_LEVEL=2` for bootstrap. |

See [monitoring.md](monitoring.md) for Prometheus wiring.

# Discord-ready deploy snippet

For posting in the Bittensor Discord / Telegram / X thread.

## Mine on testnet 462 in 4 commands

```bash
git clone https://github.com/reliquadotai/reliquary-ledger
cd reliquary-ledger && cp env.testnet.example .env   # edit netuid, R2 creds, wallet path
export BT_WALLETS_DIR=$HOME/.bittensor/wallets
docker compose up -d miner

curl http://localhost:9180/healthz
```

Validator path is identical — swap `miner` for `validator`. See [docs/validator-quickstart.md](validator-quickstart.md) for the mesh-join coordination step.

## What `.env` needs

You can mine with just these set (rest have safe testnet defaults):

```dotenv
# Wallet (must already exist; create with `btcli w new_coldkey` then `new_hotkey`)
BT_WALLET_PATH=$HOME/.bittensor/wallets
WALLET_NAME=reliquary-ledger
HOTKEY_NAME=miner

# R2 storage (sign up at https://dash.cloudflare.com → R2 → Create bucket)
RELIQUARY_INFERENCE_R2_BUCKET=your-bucket-name
RELIQUARY_INFERENCE_R2_ENDPOINT_URL=https://YOUR_ACCOUNT.r2.cloudflarestorage.com
RELIQUARY_INFERENCE_R2_ACCESS_KEY_ID=...
RELIQUARY_INFERENCE_R2_SECRET_ACCESS_KEY=...
```

Hotkey must be registered on `netuid=462` (testnet) before submission. Use
`btcli s register --netuid 462 --network test` to register.

## What you get back

- `:9180/healthz` — JSON health (chain connected, model loaded, last window verified)
- `:9108/metrics` — Prometheus text format
- `:9108/dashboard` — embedded operator dashboard (window count, mining/validation totals, zone-filter gauges)
- Submissions visible in the [public R2 audit index](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html) within minutes.

## Auto-update (opt-in)

Watchtower polls GHCR every 5 min and rolls forward only containers labeled
`com.centurylinklabs.watchtower.enable=true`. By default that's the miner only — validators are excluded so weight-setting authority isn't auto-rolled mid-window.

```bash
docker compose -f docker-compose.yml \
  -f docker/docker-compose.watchtower.yml up -d
```

## Bare-metal (no Docker)

```bash
git clone https://github.com/reliquadotai/reliquary-ledger
cd reliquary-ledger
bash scripts/setup_remote_box.sh   # installs deps, venv, .env
# edit .env
bash scripts/launch_miner.sh
tail -f data/logs/miner.log
```

## Help

- [docs/miner-quickstart.md](miner-quickstart.md) — full step-by-step.
- [docs/validator-quickstart.md](validator-quickstart.md) — validator + mesh join.
- [docs/runbook-mainnet.md](runbook-mainnet.md) — incident scenarios.
- Discord: `#reliquary` channel on Bittensor.

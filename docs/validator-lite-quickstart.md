# Validator-lite quickstart

A CPU-only Reliquary validator. Runs the 6 CPU stages of the 9-stage
verifier independently and borrows the 3 GPU stages (sketch proof,
logprob replay, distribution KL) from a quorum of full validators'
published verdicts. No GPU. No env file. ~1.3 GB image, ~1 GB RAM,
$5/month VPS class.

The image lives at `ghcr.io/reliquadotai/reliquary-validator-lite:latest`
and is rebuilt automatically on every commit to `main` via
`.github/workflows/docker-ghcr-publish-lite.yml`. Watchtower
auto-pulls fresh tags.

If you don't know whether you want lite, full, or mirror: **want
lite.** Full validator needs an H100-class GPU + full setup; mirror
is even lower-friction but contributes zero independent verification
signal. Lite is the sweet spot.

## Three validator tiers

| Tier | What it runs | Image | GPU? | Earns weights? | Use case |
|---|---|---|---|---|---|
| Full | All 9 stages independently | `ghcr.io/reliquadotai/reliquary-ledger:latest` | yes | yes | the canonical mainnet validator we run |
| **Lite** | 6 CPU stages independently + 3 GPU stages borrowed from quorum | `ghcr.io/reliquadotai/reliquary-validator-lite:latest` | **no** | **yes** | external operator default |
| Mirror | Zero stages — pure quorum aggregator | `ghcr.io/reliquadotai/reliquary-validator-lite:latest` (env override) | no | yes (no independent signal) | maximally low-friction; trust mesh fully |

To switch a lite container into mirror mode: add `-e RELIQUARY_INFERENCE_VALIDATOR_MODE=mirror`
to the run command.

## What lite does (vs full)

| Stage | Full | Lite |
|---|---|---|
| 1. Schema | independent CPU | independent CPU |
| 2. Tokens | independent CPU | independent CPU |
| 3. Prompt binding | independent CPU | independent CPU |
| 4. **Proof (sketch)** | independent GPU | borrowed from quorum |
| 5. Termination | independent CPU | independent CPU |
| 6. Environment | independent CPU | independent CPU |
| 7. Reward | independent CPU | independent CPU |
| 8. **Logprob** | independent GPU | borrowed from quorum |
| 9. **Distribution** | independent GPU (soft) | borrowed from quorum |

**Independent veto.** Lite validator rejects anything its 6 CPU
stages catch — no quorum can override that. The borrow only kicks
in when the CPU stages all passed and the GPU stages need a verdict.

**Quorum rule.** Lite accepts iff ≥ 2 full validators' published
verdicts also accepted. Lite rejects iff ≥ 2 full validators
rejected on a GPU stage. Otherwise lite abstains.

**Yuma independence preserved.** Each lite validator signs and
publishes its own verdict bundle, sets its own weights. The mesh
sees N independent weight vectors, not 1 amplified vector.

## Setup (testnet 462)

Two prerequisites:

1. **Docker + Docker Compose** on the host.
2. **A registered hotkey** on netuid 462. If you don't have one:
   ```bash
   btcli wallet new-coldkey --wallet-name reliquary --n-words 12
   btcli wallet new-hotkey  --wallet-name reliquary --hotkey v1 --n-words 12
   # Fund the coldkey (testnet faucet) then register:
   btcli subnet register \
     --wallet-name reliquary --hotkey v1 \
     --netuid 462 --subtensor.network test
   ```

Then bring it up:

```bash
curl -fsSLo docker-compose.validator-lite.yml \
  https://raw.githubusercontent.com/reliquadotai/reliquary-ledger/main/docker-compose.validator-lite.yml

WALLET_NAME=reliquary HOTKEY_NAME=v1 \
  docker compose -f docker-compose.validator-lite.yml up -d
```

That's it. The container hardcodes every other config (chain endpoint,
netuid, R2 audit URL, model_ref, σ filter band, polling interval).
Watchtower polls the registry every 5 minutes; new releases auto-update.

Verify:

```bash
curl -fsS http://127.0.0.1:9180/healthz
docker logs -f reliquary-validator-lite
```

## Setup (mainnet)

When mainnet cutover happens, set three additional env vars:

```bash
WALLET_NAME=reliquary HOTKEY_NAME=v1 \
  RELIQUARY_INFERENCE_NETWORK=finney \
  RELIQUARY_INFERENCE_NETUID=81 \
  RELIQUARY_INFERENCE_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443 \
  BT_SUBTENSOR_NETWORK=finney \
  docker compose -f docker-compose.validator-lite.yml up -d
```

## What you DON'T need to set

- `RELIQUARY_INFERENCE_R2_*` — lite validator only reads from the
  public bucket; no R2 credentials required.
- `RELIQUARY_INFERENCE_MODEL_REF` — hardcoded in the image.
- `RELIQUARY_INFERENCE_TASK_SOURCE` — hardcoded.
- `RELIQUARY_INFERENCE_DEVICE` — hardcoded to CPU.
- `RELIQUARY_INFERENCE_LITE_QUORUM` — defaults to 2; rarely needs
  changing.

## What lite mode does NOT give you

- **No miner reward.** Lite validators don't earn miner rewards (they
  don't generate rollouts).
- **No "I caught the cheater alone" credit on GPU stages.** A
  validator running full mode will be the one whose proof-stage
  reject signal counts. Lite validators ride on the full validators'
  signal for those three stages.
- **No quorum signal until ≥ 2 full validators have published.** If
  you're the first lite validator on a brand-new mainnet with only
  one full validator, you'll abstain on every completion until a
  second full validator publishes its bundle. Workaround: run
  alongside ≥ 2 full validators (which is the standard mesh shape).

## Switching from lite to full later

If you scale up your operation and want to run full mode:

```bash
# Stop lite
docker compose -f docker-compose.validator-lite.yml down

# Switch to the full image (needs GPU + env file with R2 creds)
# See docker-compose.yml + env.testnet.example.
```

Your hotkey carries over — same on-chain identity, different mode.

## How to verify your lite validator is on the mesh

After ~3 windows (a few minutes on testnet 462), your hotkey should
appear in the audit index. The lite validator's verdicts are
published to R2 the same way full validators' are. Check:

```bash
curl -s https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.json \
  | jq '.windows[].chain_publish_result.uids'
```

Your validator UID should be present in `uids` for recent windows.

## Failure modes + diagnostics

| Symptom | Likely cause | Fix |
|---|---|---|
| `lite_quorum_abstain` on every completion | < 2 full validators publishing verdicts | wait for the mesh to fill, or check the audit index for live full validators |
| `borrowed_gpu_reject_quorum` | full validators' GPU stages legitimately rejected the rollout | this is correct behavior — the rollout is bad |
| CPU stage rejects | lite caught a defect independently | this is correct behavior — lite is doing its job |
| `model_config_unavailable` | tokenizer or AutoConfig couldn't load | check internet access to HF Hub from the container |
| Image fails to start | missing wallet mount / wrong env | check `BT_WALLET_PATH` resolves and your hotkey exists at `<wallet>/<HOTKEY_NAME>` |

## Architecture references

- [`reliquary_inference/validator/mode.py`](../reliquary_inference/validator/mode.py) — mode constants + quorum decision function
- [`reliquary_inference/validator/lite_verifier.py`](../reliquary_inference/validator/lite_verifier.py) — lite verification entry point
- [`reliquary_inference/validator/pipeline.py`](../reliquary_inference/validator/pipeline.py) — the 9-stage pipeline (StagePolicy.enabled_stages is the lite knob)
- `Dockerfile.validator-lite` — CPU-only image, ~200 MB
- `docker-compose.validator-lite.yml` — Watchtower-ready compose template

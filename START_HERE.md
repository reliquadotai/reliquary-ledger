# Start here

Reliquary is a Bittensor subnet (live on testnet **netuid 462**) that runs two
companion runtimes under a single netuid: **Ledger** (proof-carrying inference,
this repo) and **Forge** (training). The reference implementation is mainnet-
ready as a self-contained operator deployment — cutover is gated only on
internal proof-of-system criteria (code green, cross-GPU bit-exact verified,
trainer firing, mesh quorum stable). Subnet identity for mainnet is tracked
separately and does not block readiness.

Pick your role:

| Role        | Hardware                | Quickstart                                                                  |
|-------------|-------------------------|-----------------------------------------------------------------------------|
| Miner       | 1× GPU ≥ 24 GB VRAM     | [docs/miner-quickstart.md](docs/miner-quickstart.md)                        |
| Validator   | 1× A100/H100/B200       | [docs/validator-quickstart.md](docs/validator-quickstart.md)                |
| Trainer     | 1–2× GPU on Forge       | [reliquary-forge START_HERE.md](https://github.com/reliquadotai/reliquary-forge/blob/main/START_HERE.md) |
| Spectator   | a browser               | [Public status page](https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/audit/index.html) |

## 5-minute path

```bash
# 1. Clone and configure for testnet 462.
git clone https://github.com/reliquadotai/reliquary-ledger.git
cd reliquary-ledger
cp env.testnet.example .env       # edit netuid, R2 creds, wallet path

# 2. Pick a role and launch.
export BT_WALLETS_DIR="$HOME/.bittensor/wallets"
docker compose up -d miner        # or `validator`

# 3. Verify.
curl http://localhost:9180/healthz
```

For the bare-metal path (systemd, no Docker) see [docs/deployment.md](docs/deployment.md).

## How it fits together

```
┌──────────────────────────────── netuid 462 (testnet) ────────────────────────────────┐
│                                                                                      │
│   Miners  ──(rollouts + GRAIL sketch + HMAC)─→  Validator mesh  ──verdicts──→  R2    │
│                                                       │                              │
│                                                weights│                              │
│                                                       ▼                              │
│                                                Bittensor chain                       │
│                                                                                      │
│   ────────────── closed-loop bridge (signed PolicyCommitment) ──────────────         │
│                                                       │                              │
│   Forge (reliquary-forge) ──GRPO step──→  delta ckpt ─┘                              │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

- **Miners** generate completions for deterministic Hendrycks-MATH tasks, attach a
  GRAIL sketch + Ed25519 / HMAC signature, and submit to R2.
- **Validators** run a 9-stage verifier pipeline (schema → tokens → prompt → proof →
  termination → environment → reward → logprob → distribution) per completion, then
  the 4-validator mesh aggregates verdicts via stake-weighted median.
- **Forge** consumes in-zone (σ ≥ 0.33 / 0.43) rollout groups, runs PPO-clipped GRPO
  with KL penalty, and publishes signed delta checkpoints back to miners.
- **Closed-loop bridge** lets miners hot-swap to the new policy at the next window
  boundary without a process restart.

## What's where

| Repo                                                                  | Role                                                  |
|-----------------------------------------------------------------------|-------------------------------------------------------|
| [reliquary-ledger](https://github.com/reliquadotai/reliquary-ledger)  | Inference + verification runtime (this repo)          |
| [reliquary-forge](https://github.com/reliquadotai/reliquary-forge)    | Training runtime (GRPO + distillation)                |
| [reliquary-protocol](https://github.com/reliquadotai/reliquary-protocol) | Shared protocol package (signatures, R2, bridge)   |

## Next steps

- [docs/miner-quickstart.md](docs/miner-quickstart.md) — clone-to-mining in 15 min.
- [docs/validator-quickstart.md](docs/validator-quickstart.md) — clone-to-validating + mesh join.
- [docs/protocol.md](docs/protocol.md) — the 9-stage pipeline.
- [docs/audit.md](docs/audit.md) — public R2 audit index.
- [CONTRIBUTORS.md](CONTRIBUTORS.md) — credit, attribution, parallel work.

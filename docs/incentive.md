# Incentive Mechanism

Reliquary Ledger pays miners for **unique accepted proof-carrying completions**
and nothing else. No raw token volume. No self-reported rewards. Every scored
rollout passed the 9-stage verifier pipeline and at least one mesh validator's
independent replay.

## Weight formula

```
raw_scoreᵢ = min(uniqueᵢ, UNIQUE_ROLLOUTS_CAP) ^ SUPERLINEAR_EXPONENT
weightᵢ    = raw_scoreᵢ / Σⱼ raw_scoreⱼ
```

Where:

- `uniqueᵢ` = count of rollouts from miner i in the rolling scoring window
  that passed every hard check AND every soft check (no duplicate digest,
  no dataset copycat, no duplicate task submission).
- `SUPERLINEAR_EXPONENT` = 4.0 (configurable; see
  `reliquary_inference/constants.py`). Gives Sybil resistance: 2 miners
  each doing half the work earn ≈ 12.5% of what 1 miner doing the full
  work earns (`(½)⁴ × 2 = ⅛`), so whale-splitting is unrewarding.
- `UNIQUE_ROLLOUTS_CAP` = 5000 per window. Caps the upside so one miner
  can't lock out the rest of the fleet by throwing compute at it.

## Rolling window

Weights are reset each on-chain `WEIGHT_SUBMISSION_INTERVAL` (default
360 blocks on testnet). Within a weight submission interval the
validator aggregates **verification totals** from every window it
processes. EMA smoothing is applied across the last ROLLING_WINDOWS = 72
windows so a miner's score has ≈ 25-window half-life.

## What qualifies as "unique accepted"

A rollout counts toward `uniqueᵢ` iff its verdict payload has:

- `accepted: true` (passed every hard + soft check)

Specifically, a verdict fails to count when any of:

| Stage | Reason | Classification |
|---|---|---|
| 1 schema | missing field / version mismatch / duplicate nonce | hard |
| 2 tokens | out of vocab / length exceeded | hard |
| 3 prompt | binding mismatch / unknown source | hard |
| 4 proof | GRAIL sketch mismatch / no positions checked | hard |
| 5 termination | no EOS / overflow | hard |
| 6 environment | evaluate_math_trace returned accepted=False | hard |
| 7 reward | contract violation / missing | hard |
| 8 logprob | drift exceeded / missing | hard |
| 9 distribution | median out of band | **soft** |
| — | signature | pre-pipeline hard |
| — | duplicate task submission | **soft** |
| — | dataset copycat | **soft** |
| — | duplicate completion digest | **soft** |

Soft fails produce an artifact but don't count toward the weight.

## Why these choices

- **Exponent 4**: makes Sybil-splitting strictly worse than consolidating.
  Tested against the 5-miner archetype showcase (honest high, honest avg,
  low-diversity, weak-policy, malformed-proof) — the honest-high miner
  consistently wins with ≈ 60% share.
- **Cap 5000**: at the current window cadence of 6 min, 5000 rollouts
  represents ≥ 50× a realistic single-GPU output rate, so the cap only
  binds in concentrated-attack scenarios.
- **Rolling window 72**: matches the WEIGHT_SUBMISSION_INTERVAL=360 / 5
  block cadence so a miner that stops contributing loses half its score
  in about 25 windows (≈ 2.5 hours on testnet cadence).

## Cross-check with mesh verdicts

The weight formula runs on every validator independently. For a miner to
receive weight from validator V:

1. V must have `accepted: true` for the rollout.
2. If you ask multiple validators to publish weights, you get the stake-
   weighted mean of each validator's independent tally (standard
   Bittensor Yuma).

The mesh consensus layer (stake cap 10%, > 50% honest-stake majority
required for upgrade) prevents any single validator from dominating the
weight-setting.

## Policy commitments + training rewards

Ledger's weight formula covers **inference** work only. The companion
Forge trainer consumes in-zone rollout groups from Ledger and publishes
policy checkpoints via the closed-loop bridge. Forge's incentive system
(trainer-quorum weighting, benchmark-gated promotion) lives in
[reliquary/docs/scoring.md](https://github.com/reliquadotai/reliquary/blob/main/docs/scoring.md)
and is independent of the per-completion weight here.

## Relevant constants

See `reliquary_inference/constants.py`:

- `SUPERLINEAR_EXPONENT = 4.0`
- `UNIQUE_ROLLOUTS_CAP = 5000`
- `UNIQUE_ROLLOUTS_CAP_ENABLED = True`
- `WEIGHT_SUBMISSION_INTERVAL = 360`  # blocks
- `ROLLING_WINDOWS = 72`
- `EMA_ALPHA = 2.0 / (ROLLING_WINDOWS + 1)`  # ≈ 0.0274

Upgrade cadence for these constants is version-pinned in
`reliquary-protocol`; any change requires a coordinated bump of both
subnets and a 24-hour testnet bake before mainnet.

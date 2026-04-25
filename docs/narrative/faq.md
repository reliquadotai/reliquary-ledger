# Reliquary — Frequently Asked Questions

## Protocol

### What is proof-carrying inference?

Every miner completion carries a compact cryptographic fingerprint (a
*sketch*) of the hidden states at a designated transformer layer,
bound by HMAC to the miner's submission. Validators re-run one
forward pass at 32 sampled positions to verify the sketch matches —
orders of magnitude cheaper than re-running the full decode, and
sufficient to reject any gross tampering. Subtle per-position tampering
is caught by the combined nine-stage pipeline.

### Why 32 sampled positions?

`CHALLENGE_K = 32`. Empirically covers typical completion lengths
(64-256 tokens) with ≥ 12% sampling density. Positions are derived
from the miner's tokens + per-window randomness, so a miner cannot
predict which positions will be checked before committing the sketch
+ tokens together. Targeted per-position tampering requires tampering
32 positions without the validator noticing — infeasible under
`PRIME_Q = 2_147_483_647` with `PROOF_SKETCH_TOLERANCE_BASE = 6000`.

### Why the Mersenne M31 prime?

`2^31 - 1` fits in int32 for fast mod-reduction on commodity hardware;
empirically no observable modular collisions in 430K+ trials across
Grail's prior art + our own audits. Upgrade path to a larger prime is
documented (protocol version bump + coordinated 24h testnet bake).

### How does a validator mesh reach consensus?

Stake-weighted median of per-completion verdicts, with stake capped at
10% per validator before aggregation. Quorum requires ≥ 50% of the
capped total. Validators whose verdict vector is > 0.25 euclidean
distance from the median are flagged as outliers; > 5% outlier rate
over a window gates that validator for 12 subsequent windows.

### What prevents a colluding minority from rewriting consensus?

- Stake cap at 10% per validator means no single whale can dominate.
- Outlier detection is automatic + signed by the majority of honest
  stake.
- Disagreement rate gate kicks in automatically — no governance
  action required.
- Live audit: 4 validators including 1 simulated malicious (10%
  stake); malicious validator 1.0 disagreement, cleanly gated,
  zero effect on consensus across 256 verdicts.

### What stops a miner forging a proof?

The sketch material `r` is derived from per-window randomness
(subtensor-committed) via a PRF. The miner cannot compute valid
sketches without running the policy model's forward pass. HMAC-SHA256
binds the sketch to the miner's hotkey secret. Forgery requires
breaking SHA-256.

### What about a miner running a smaller model and hoping for tolerance slack?

The nine-stage pipeline's logprob drift stage (stage 8) rejects if
≥ 51% of positions show > 0.15 log-prob drift vs the declared policy
model. Running a different model produces catastrophic logprob drift;
the sketch alone might fall within tolerance, but stage 8 catches it.

## Operations

### What does it cost to run a miner?

One commodity-grade GPU (RTX 4090 / A100 / Blackwell), reasonable
network (≤ 100 ms to subtensor endpoint), ≥ 32 GB RAM. Storage for the
policy checkpoint is bounded by the delta-checkpoint compression
(~50% of fp16 baseline per window; full snapshots every 64 windows).

### What does it cost to run a validator?

One Blackwell or H100 per validator hotkey (verifier forward pass
needs fp16/bf16 + flash-attn2 for the proof-layer recomputation);
~256 GB RAM for metagraph cache + artifact registry. Network ~1 Gbps
for shard-parallel checkpoint download.

### How do I verify what a miner sent me without waiting for mesh consensus?

Run `reliquary-inference verify --submission path/to/submission.json
--policy <policy-id>`. This runs all nine stages locally with your
own model download; returns accept / reject + the explanation struct
that records which stage fired.

### What happens if my validator crashes mid-window?

The verdict-storage pipeline spools un-published verdicts to local
disk atomically via `tempfile` + `os.replace` + fsync. On restart,
`flush_spool()` pushes the spool to the R2 backend. No window data
is lost on crash; at worst a late-arriving verdict reaches the mesh
after the window has closed, in which case the aggregator ignores it
(later windows continue).

## Economics

### Is there a token?

Reliquary runs on Bittensor's TAO + subnet alpha token system. Mining
rewards follow Bittensor's per-subnet emission curve, distributed via
the subnet's consensus weights. No separate token.

### How is stake slashed?

A validator whose onchain commitment diverges from the mesh-consensus
merkle root is slash-eligible under the Tier 4 governance charter
(TBD wired). A miner submitting detectable forgery loses their share
of the window's emissions (stage-level reject) and is eligible for
the copycat-detection gate if the submission matches another miner's
canonical first submission.

### How is an env author's stake returned?

Authors lock TAO at submission time. If the env stays in good standing
(ACTIVE status, no security incidents) for 30 days, stake unlocks.
If the env is revoked (validators vote revoke, or security incident
triggers automatic revocation), stake is partially slashed per the
governance charter.

## Governance

### Who controls the subnet?

Subnet ownership follows Bittensor's standard subnet-owner mechanics.
Acceptance of any miner or validator submission is determined by the
protocol's verifier pipeline + stake-weighted mesh consensus.
Protocol upgrades require a coordinated package version bump with
testnet bake + onchain commit of upgrade intent.

### Can a validator-mesh majority push a bad upgrade?

No. Upgrades require:
1. `reliquary-protocol` version bump (committed to source).
2. Matching pin bumps in BOTH runtimes' pyproject.
3. 24h testnet bake.
4. Mainnet rollout at a pre-announced block height.
5. Validators commit upgrade intent onchain BEFORE the cutover.

A majority in one runtime cannot force a change that breaks the other
because both runtimes' canonical bytes must match — a forked runtime
produces orphan verdicts that the other side rejects.

## Road ahead

### What's the next major milestone?

Full 100K empirical audit on Blackwell fleet. Mainnet Ledger cutover
after the continuous-operation track record on testnet netuid 462
crosses 4 weeks of uninterrupted mesh consensus.

### When will Forge come online at full scale?

Forge runs as a companion protocol on the same subnet as Ledger; the
closed-loop bridge is already live on testnet. Full-scale activation
(mainnet, production GRPO cadence, distillation lane fully wired)
tracks Ledger stability — once Ledger has demonstrated a ~4-week
continuous-operation record on mainnet, Forge ramps to production
cadence with shared protocol + divergent learner configuration.
Target window: Tier 3 Epic 5.

### What are the open research questions?

Listed in section 14 of the protocol paper: sketch-tolerance lower
bound, delta-checkpoint compression ceiling (FP8 migration), WASM
sandbox alternative, Ed25519 signature migration.

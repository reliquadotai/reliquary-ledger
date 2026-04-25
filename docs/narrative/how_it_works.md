# How Reliquary Works — in one page

AI runs on someone else's servers. How do you know the server actually
ran the model you asked for, and didn't hand you cheap garbage?

Re-running the AI yourself is expensive. **Reliquary** solves this
with **proof-carrying inference**: every answer ships with a tiny
cryptographic receipt. Anyone can check the receipt in milliseconds
and know the answer was computed honestly.

## The cast

- **Miners** run the model and submit `(answer, receipt)`.
- **Validators** check receipts, vote on each answer, and pay miners.
- **The chain** records every vote so anyone can audit later.

## One request, end to end

```
      ┌────────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────┐
      │   Policy   │───▶│  Miner   │───▶│  Validator  │───▶│  Chain  │
      │  authority │    │          │    │    mesh     │    │         │
      └────────────┘    └──────────┘    └─────────────┘    └─────────┘
       publish          run model        verify 9 stages     commit
       checkpoint       emit receipt     vote in mesh        merkle root
```

1. **Policy authority** publishes the blessed model for the current
   window (e.g. "Qwen3-4B, window 42, hash abc…").
2. **Miners** download the model, run requests, and send back:
   - the answer + tokens
   - a **sketch** = hidden-state fingerprint at one layer
   - a signature binding it all to their wallet.
3. **Validators** check the receipt through 9 cheap stages:
   schema / tokens / prompt / **proof** / termination / environment /
   reward / logprob drift / distribution band. One layer of forward
   pass at 32 sampled positions — milliseconds, not seconds.
4. **The mesh** aggregates votes with stake-weighted median.
   Whale-proof: 10 % per-validator cap. Cheat-proof: disagreeing
   validators auto-gated after one bad window.
5. **The chain** commits the median's Merkle root per window. Anyone
   can read the chain later and reconstruct every verdict.

## Why it can't be cheated

| Attack | Defense |
|---|---|
| Forge a sketch without running the model | Sketch depends on per-window randomness the miner can't predict — it's PRF-bound to a value that's only revealed after the miner has already committed tokens. |
| Run a smaller / different model to save cost | 9-stage pipeline includes log-probability drift check: running a different model blows past the 0.15 log-prob tolerance on ≥ 51 % of positions. |
| Copy another miner's answer | Content-hash copycat detector with a 2-second ambiguity window and 12-window gating. First submitter wins; copier loses weight for 12 windows. |
| One big validator rigs consensus | Stake cap at **10 % per validator** before aggregation — 80 % stake buys the same voice as 10 %. |
| Colluding validators outvote honest | Mesh needs **5 independent signers** to cross the cap-weighted quorum; each costs TAO to register. Disagreeing validators auto-gated. |
| Tamper with stored verdicts | Every verdict is SHA-256-signed + stored content-addressably. Merkle root of the window goes onchain. A tampered copy fails reconstruction. |
| Write a malicious task environment | Env code runs in a 5-layer sandbox (restricted Python + AST audit + allowlisted builtins + subprocess + CPU/memory caps). No network, no filesystem, no import. |

## How miners are paid

```
weight ∝ (unique_accepted_rollouts)^4         capped at 5 000 rollouts
reward = 0.70·correctness + 0.15·novelty + 0.15·difficulty
```

- **Correct answers earn more than partial**. Wrong answers earn zero.
- **Novel answers earn more than duplicates**. Copying pays nothing.
- **Harder tasks earn more than easy**. Difficulty set by the env.
- **Sybil is punished by math**: 1 miner at 5 000 rollouts earns
  5 000⁴ weight; 2 miners at 2 500 rollouts each earn 2 × 2 500⁴ — an
  order of magnitude less. Splitting identity costs you.

## What's actually true today (empirical, on real hardware)

- **4 different GPU architectures** (3× RTX PRO 6000 Blackwell + 1× H100,
  Targon fleet) produced **bit-identical proofs** across **90 samples**.
  Same byte, every time, on every machine.
- **4-validator mesh-live run** (3 real Blackwells + 1 simulated
  malicious) processed 256 verdicts in one window. Malicious validator
  was **cleanly auto-gated** without affecting consensus.
- **1 000 honest trials**: zero false positives. The verifier never
  flagged a honest miner.
- **9-stage pipeline rejects** every class of tamper we threw at it
  (tokenization, hidden-state replacement, different-model substitution).

## What makes Reliquary different from today's subnets

| Most subnets | Reliquary |
|---|---|
| Trust one validator per request | Stake-weighted median across the mesh, with 10 % cap so no whale dominates |
| Re-run the full model to verify | Verify 32 positions out of hundreds — ~99 % cheaper, same security |
| Hard-code the task environment | Permissionless env registry — anyone can propose a new task |
| No receipts, no audit trail | Every verdict has a content-addressable hash and onchain Merkle commit |
| Training happens behind closed doors | Every training checkpoint is a delta bundle with Merkle root onchain |

## The one sentence pitch

**Run the model, prove you did, submit the proof. Anyone can verify
in milliseconds. Cheating is mathematically unprofitable.**

---

*Technical details: see the [protocol paper](../paper/reliquary_protocol_paper.md).
Source: [reliquary-inference](https://github.com/reliquadotai/reliquary-ledger),
[reliquary](https://github.com/reliquadotai/reliquary-forge),
[reliquary-protocol](https://github.com/reliquadotai/reliquary-protocol).*

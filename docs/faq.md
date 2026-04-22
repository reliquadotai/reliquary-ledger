# FAQ

## What is Reliquary Ledger?

The inference runtime for Reliquary, a two-subnet Bittensor platform for
proof-carrying AI training. Ledger produces verifiable completions that a
sibling trainer (Forge) turns into policy checkpoints. The two subnets share
a [protocol package](https://github.com/0xgrizz/reliquary-protocol) so their
bytes cross-verify deterministically.

Ledger is live on testnet netuid 462. Mainnet cutover is gated on 4 weeks
of continuous-operation track record plus external cryptographer review of
the protocol paper.

## Which task source is live?

`math` — the Hendrycks MATH benchmark (12 500 problems across 5 difficulty
levels and 7 subjects). Correctness = last `\boxed{…}` content exact-matched
against the reference answer after conservative LaTeX normalization. A
bootstrap filter (`RELIQUARY_INFERENCE_MATH_MAX_LEVEL=2`) restricts the
sampling pool to easier problems until the base model can solve Level 3+.

Legacy sources `reasoning_tasks` and `dataset_prompts` are retained in the
codebase for tests and low-resource fallbacks but are not the live target.

## Why MATH instead of synthetic or arithmetic tasks?

Two reasons:

1. **Real gradient signal.** Qwen2.5-3B on Hendrycks MATH splits 2-6 correct
   out of 8 on a meaningful fraction of problems, which is exactly where the
   DAPO zone filter (σ ≥ 0.33 bootstrap, 0.43 steady) pulls useful GRPO
   advantage. Synthetic arithmetic collapses to all-correct or all-wrong.

2. **Verifier-friendly.** Boxed-answer extraction + LaTeX normalization is
   objective and deterministic — no LLM judge, no reward model, no cross-GPU
   drift.

## What's the DAPO zone filter?

A rollout group is `(miner_id, task_id)` — M samples of one miner on one
prompt. The group's within-group reward std σ is the gradient-signal
indicator. Groups with σ below threshold are dropped upfront; only σ ≥ σ_min
groups feed the trainer. For binary 0/1 rewards the steady-state threshold
0.43 corresponds to Bernoulli k ∈ [2, 6] correct — the canonical GRPO-OOA
zone.

## How does the closed-loop bridge work?

1. Forge pulls in-zone rollout groups from R2 via a local cursor (900
   R2 gets/cycle → ~10 in steady state).
2. Forge runs GRPO (PPO-clipped surrogate + KL k3 estimator against a
   frozen reference) on the groups.
3. Forge publishes a `DeltaBundle` + `CheckpointAttestation` (HMAC-signed
   training provenance) + on-chain `PolicyCommitment`.
4. Ledger's `policy_consumer` polls commitments every ~10 s, verifies the
   attestation signature via `BridgeVerifier`, runs the delta through the
   reparam-trick sanity guard (finite / magnitude / layer-scale ratio),
   then mutates cached model bundles in-place at `effective_at_window`.

See [architecture.md](architecture.md) for the full diagram.

## What prevents reparameterization attacks on policy deltas?

Our attack surface is narrower than checkpoint-upload subnets because we
only consume HMAC-signed deltas from a trusted Forge trainer. But
defense-in-depth: before any delta applies to the cached model,
`reliquary_inference.shared.reparam_guard` rejects:

1. NaN / inf in any shard.
2. Projection tensors with mean |w| below a floor (default 1e-4).
3. Per-layer scale-ratio imbalance above a bound (default 1e5) — catches
   the RMSNorm×Linear rescaling exploit that leaves one tensor scaled to ~0
   and another to ~∞.

## Why proofs?

They reduce the trust gap between what miners claim and what validators
can replay. Reliquary scores completions that stay bound to deterministic
tasks, signature bindings, and hidden-state sketch checks — not just
tokens.

## Why use Bittensor instead of a regular API marketplace?

Bittensor is the coordination + incentive layer. It provides subnet
membership, identity, and weight publication. Reliquary keeps large
artifacts off-chain (R2) and uses the chain for what it's good at:
coordination, incentives, and policy commitments.

## Why are artifacts off-chain?

Artifacts are larger, more numerous, and more inspectable than what
belongs on-chain. Reliquary uses R2 + CDN for the artifact layer and
publishes weights + compact commitment references through the chain.

## Who controls the subnet?

No single entity. The coldkey is a multi-sig (3-of-5; signers listed in
the governance charter). Protocol upgrades require a coordinated
`reliquary-protocol` version bump with testnet bake + on-chain commit of
upgrade intent. Governance operates entirely via the on-chain charter
committed per-window through the subnet identity interface — no external
delegation, no single-key authority.

## Can a validator-mesh majority push a bad upgrade?

No. Upgrades require:

1. `reliquary-protocol` version bump (committed to source).
2. Matching pin bumps in both subnets' `pyproject.toml`.
3. 24h testnet bake.
4. Mainnet rollout at a pre-announced block height.
5. Validators commit upgrade intent on-chain BEFORE the cutover.

A majority in one subnet cannot force a change that breaks the other
subnet because the two subnets' canonical bytes must match — a forked
subnet produces orphan verdicts that the other side rejects.

## Why is the repository named `reliquary-inference` if the product name is Reliquary Ledger?

Historical — the repo was created before the two-subnet split was
finalised. The package inside it is `reliquary_inference`; the product-
facing name is **Reliquary Ledger**. The sibling repo
[`reliquary`](https://github.com/0xgrizz/reliquary) is **Reliquary Forge**.

# Threat Model

**Scope**: Reliquary's proof-carrying inference + closed-loop RL
training pipeline as deployed on Bittensor testnet netuid 462.

## Trust assumptions

We assume:
- **Bittensor chain liveness + safety** — the subtensor itself is
  honest; finalized blocks are immutable; on-chain commitments are
  genuinely available.
- **Hash function soundness** — SHA-256 is collision-resistant and
  pre-image-resistant over the message lengths we use.
- **HMAC-SHA256 PRF properties** — an adversary without the secret
  cannot forge a valid HMAC over a payload they don't control.
- **TLS to R2 + the subtensor RPC** — the underlying transport
  prevents trivial tampering in transit. (We do not rely on TLS for
  authenticity; every authenticated artifact is signed inside the
  envelope.)

We do **not** assume:
- Any individual operator (miner, validator, policy authority) is
  honest.
- Any individual GPU executes correctly — bit-exact cross-GPU
  determinism is an empirical result, not a cryptographic assumption
  (see `../audit/cross_gpu/`).
- Any third-party storage (R2) is append-only or tamper-proof; every
  artifact hash is committed onchain before it matters.

## Adversaries we defend against

### A1. Dishonest miner
**Threat**: Submits a completion that wasn't produced by the declared
policy, or claims a reward they didn't earn.

**Defenses**:
- **GRAIL sketch commitment** — every completion carries a compact
  fingerprint of the hidden states at the declared proof layer, bound
  by HMAC to the miner's hotkey. A forged completion would require
  solving the discrete log of the miner's sketch over a Mersenne
  M31 prime without the model's actual forward pass.
- **9-stage verifier pipeline** — validators recompute the forward
  pass at `CHALLENGE_K = 32` sampled positions (derived from the
  miner's own tokens + per-window public randomness, so the miner
  cannot predict positions before committing). Stages: schema →
  tokens → prompt → proof → termination → environment → reward →
  logprob → distribution. Passing all 9 without actually running
  the policy requires breaking SHA-256.
- **Logprob drift stage (stage 8)** — rejects when ≥ 51% of
  positions show > 0.15 log-prob drift vs the declared policy model.
  A miner running a smaller / different model produces catastrophic
  drift even when the sketch alone might fall within tolerance.

### A2. Dishonest validator (or validator coalition up to 49% of stake)
**Threat**: Publishes false verdicts to shape weights in their favor;
gates an honest miner; passes a malicious miner's forged completion.

**Defenses**:
- **Stake-weighted median consensus with 10% stake cap per validator
  per completion** — no single whale can dominate.
- **Outlier gate** — validators whose per-completion verdict vector
  is > 0.25 Euclidean distance from the median are flagged; > 5%
  outlier rate over a rolling window gates that validator for 12
  subsequent windows.
- **Mesh cross-check at apply time** — every Ledger validator
  applies policy deltas independently; one compromised validator
  can't push bad weights unilaterally.
- **Empirical**: Tier 2 Epic 1 v2 audit ran 3 honest Blackwells + 1
  simulated-malicious validator across 256 verdicts; the malicious
  validator was cleanly gated (1.0 disagreement rate) with zero
  effect on consensus. See `../audit/mesh_live/`.

### A3. Dishonest Forge trainer (policy authority)
**Threat**: Publishes a training delta that wasn't actually produced
by GRPO on the claimed rollout windows; poisons the policy with a
backdoor; rescales weights to exploit the reparam symmetry.

**Defenses**:
- **HMAC-signed CheckpointAttestation** binds the merkle root of the
  delta to the exact Ledger rollout window range consumed. The
  `consumed_rollout_windows` + `consumed_rollout_keys` lists are in
  the canonical bytes the signature covers.
- **Smoke hash** recomputed on apply — `sha256(manifest_bytes ||
  sorted shard payload_sha256s)`. Corruption in transit rejected.
- **Reparam-trick sanity guard** — `reliquary_inference.shared.reparam_guard`
  rejects deltas with:
  - NaN / inf in any shard.
  - Projection tensors with mean |w| below a floor (default 1e-4).
  - Per-layer scale-ratio imbalance above a bound (default 1e5) —
    catches RMSNorm×Linear rescaling that leaves one tensor scaled
    to ~0 and another to ~∞.
- **Policy authority hotkey allowlist** — `BridgeVerifier` rejects
  any commitment not signed by an allowlisted authority key.
- **Mesh cross-check** — every validator applies the same delta.

### A4. Malicious task environment author (permissionless registry)
**Threat**: Submits a task spec whose reward function exfiltrates
miner outputs, or claims an environment with a backdoored grader.

**Defenses** (Tier 3 Epic 2, design + sandbox shipped):
- **Restricted-Python sandbox** — author code runs under a
  5-layer bypass-resistant sandbox (AST audit + builtins whitelist +
  subprocess blocker + rlimit + RestrictedPython).
- **Stake commitment** — authors lock TAO at submission; slashed
  on detected misbehavior.
- **Validator shadow-mode admission** — validators replay the
  author's declared sample in the sandbox; reject if reward
  disagrees with author's expected value.
- **30-day ACTIVE probation** before full reward flow.

### A5. Chain-level governance attack
**Threat**: Majority of subnet coldkey signers push a malicious
protocol upgrade.

**Defenses**:
- **3-of-5 multi-sig coldkey** (drilled + tested on testnet).
- **Coordinated protocol version pin** — a version bump requires
  matching pins on `reliquary-inference` + `reliquary` +
  `reliquary-protocol` in a single coordinated release.
- **24h testnet bake** at every upgrade.
- **Mainnet rollout at a pre-announced block height**; validators
  commit upgrade intent onchain BEFORE cutover. Any runtime still on
  the old version after cutover produces orphan verdicts rejected
  by up-to-date peers.

## Out of scope for this review

- Bittensor subtensor internals (consensus, finality, validator
  election). We depend on it; we don't audit it.
- Third-party dependencies (torch, transformers, boto3). Supply-chain
  attacks on these are acknowledged and mitigated by the DCO +
  allowlist policy in `../legal/licensing.md` but not under active
  Reliquary-level scrutiny.
- Economic game theory of the emission curve / staking dynamics.
  That's a Bittensor-level concern; Reliquary inherits it.
- Side-channel attacks on GPU kernels (timing, power, thermal).
  Out of scope; no PII is processed by the subnet.
- Quantum-resistance. HMAC-SHA256 is not post-quantum; migration to
  an alternative is documented as an open research question in
  the protocol paper §14.

## Specific properties the reviewer is asked to verify

1. **Completion integrity** — given a verdict_bundle + the declared
   policy_artifact_id, can an adversary construct a completion that
   passes all 9 verifier stages without actually running the
   declared model?

2. **Mesh consensus integrity** — given 3 of 4 validators colluding,
   can they shift weights more than the 10% cap allows? Can they
   cleanly exclude an honest 4th validator without the outlier gate
   catching them?

3. **Bridge signature integrity** — given access to the R2 artifact
   bucket but NOT the policy authority HMAC secret, can an
   adversary publish a `CheckpointAttestation` + `PolicyCommitment`
   that the miner's `policy_consumer` will accept?

4. **Reparam defense completeness** — does the reparam guard catch
   every instance of the RMSNorm × Linear scale exploit described
   in the paper §9? Are there other weight-preserving symmetries
   it misses?

5. **Holdout isolation** — given the public seed constants in
   `reliquary_protocol.eval_holdout`, can an adversary construct a
   miner sampling strategy that exposes the policy to a holdout
   problem before the eval harness scores it? (This is a
   training-data-leakage attack on the benchmark's independence.)

6. **EvalBundle authenticity** — given the signed `EvalBundle`
   artifacts, can an adversary fake an improved-accuracy time-series
   without actually running the MATH holdout against the declared
   policy?

## What "accept" means

We do not expect the reviewer to endorse Reliquary as a whole. We
expect a structured report:

- **Critical findings** — an active attack that violates any of
  properties 1-6 above. Must be fixed before mainnet cutover.
- **Major findings** — a defense that relies on an undocumented
  assumption, or a parameter choice the reviewer thinks should be
  tightened (e.g. reparam guard bounds).
- **Minor findings** — spec ambiguities, missing test coverage,
  operational hardening suggestions.

The mainnet gate is: zero unresolved criticals + documented triage
path for every major.

---

*Review scope + methodology in [`scope.md`](scope.md). Response
protocol for findings in [`response-protocol.md`](response-protocol.md).*

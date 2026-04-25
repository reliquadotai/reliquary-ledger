# Reliquary: Proof-Carrying Inference for Decentralized AI Subnets

**Draft protocol paper — v0.1 (2026-04-18)**

## Abstract

Reliquary is a proof-carrying-inference protocol for Bittensor-class
decentralized AI subnets. Miners generate completions under a
deterministic inference envelope and submit a compact *sketch proof*
that a validator can re-execute in milliseconds to confirm the
inference was actually computed with the declared model on the declared
inputs. A nine-stage verifier pipeline layered on top of the sketch
verifies schema, tokens, prompt construction, proof integrity, termination,
environment execution, reward, per-position log-probabilities, and
output distribution. A mesh of validators aggregates the per-completion
accept/reject decisions into a stake-weighted median consensus with
explicit outlier detection and gating. The protocol is content-
addressable end-to-end — every artifact is identified by sha256 over
its canonical JSON form — and every hash a validator accepts is
committed onchain so external auditors can reconstruct the entire
subnet state from a single block-height snapshot.

This paper documents the protocol at the level of detail necessary for
a cryptographer or a distributed-systems practitioner to reproduce a
working implementation from scratch.

## 1. Design goals

- **Verifiability without re-execution cost**: a validator must be able
  to confirm a miner's claim in milliseconds per completion, not seconds.
- **Cross-hardware determinism**: the same proof must verify identically
  on different GPU architectures (A100, H100, Hopper, Blackwell) and
  different PyTorch+CUDA versions.
- **Permissionless onboarding**: miners + validators are pseudonymous
  hotkeys; no gatekeeper controls who runs the protocol.
- **Byzantine fault tolerance in the validator mesh**: a single
  compromised validator cannot corrupt consensus; colluding minority
  subsets cannot either, up to a stake-weighted 10% cap.
- **Content-addressable artifacts end-to-end**: every payload a validator
  accepts is identified by a sha256 hash over its canonical JSON form;
  the same payload hashes identically across any validator anywhere.
- **Onchain auditability**: every hash a validator accepts is committed
  onchain so external auditors can reconstruct the subnet's entire
  state from a block-height snapshot + R2 reads against the committed
  hashes.
- **Lazy upgradeability**: a shared protocol package pins version; both
  runtimes coordinate upgrades via a single semver bump.

## 2. System model

**Actors.**
- **Miners** generate completions from a deterministic model checkpoint
  on deterministic inputs. They submit `(completion, proof)` tuples
  back to the validator mesh.
- **Validators** verify submissions through the nine-stage pipeline
  and publish per-completion verdicts signed under their hotkey.
- **Policy authority** (HMAC-signed quorum of trainers, rotated via
  the subnet identity commit interface) publishes the authoritative
  policy checkpoint each training window. Miners download the policy
  and lock to it for that window; validators use the same policy to
  re-derive the proof material.
- **Chain**: Bittensor subtensor, carrying commitments + weight
  submissions.

**Trust.**
- Miners are untrusted; they may submit any bytes and we must detect.
- Validators are partially trusted: stake-weighted with a 10% cap so
  no single validator can dominate; a > 50% honest-stake majority is
  required for consensus.
- Chain is trusted for inclusion + ordering.
- Shared protocol package is trusted; upgrade cadence is documented
  and coordinated.

## 3. Proof protocol

### 3.1 Proof sketch

The proof is a *sketch* — a compact fingerprint of the model's
attention hidden states at the designated proof layer — plus a signature
binding the sketch to the miner's submission.

Given a transformer model and an input token sequence, let
`H[i] ∈ R^d_model` be the hidden-state vector at layer `LAYER_INDEX`
after position `i`. The sketch for position `i` is:

```
s_i = ⟨H[i], r⟩  mod  PRIME_Q
```

where `r ∈ Z^{d_model}` is a pseudorandom integer vector derived from
per-window randomness (the `"sketch"` PRF label) and `PRIME_Q =
2_147_483_647` is the Mersenne prime M31. Scaling is fixed: `round(H[i]
* 1024)` cast to int64 before the dot-product, giving a bounded-range
quantization that cross-GPU-deterministic.

The miner emits:

- `s_vals`: one `s_i` for every position in the full token sequence
  (prompt + completion).
- `signature`: HMAC-SHA256 over the canonical byte representation of
  `s_vals` under the miner's HMAC secret.
- `randomness`: the window randomness used to derive `r`.
- `indices`: a deterministic sample of positions selected via the
  `"open"` PRF label, used by the verifier during sketch verification.

### 3.2 Sketch verification

The validator re-runs the forward pass on the same token sequence using
the same policy checkpoint (downloaded via Merkle-committed delta
checkpoints). For each `i ∈ indices`, it:

1. Recomputes `H'[i]` locally.
2. Computes `s'_i = ⟨H'[i], r⟩ mod PRIME_Q`.
3. Accepts iff `|s'_i - s_i| ≤ PROOF_SKETCH_TOLERANCE_BASE` (default
   6000) after the modular reduction.

Tolerance accommodates long-sequence attention drift accumulating
across transformer layers; empirical cross-GPU runs (Tier 2 Epic 6
extension) show bit-exact agreement across 4 hosts (3 Blackwell + 1
H100) at 90 samples under this tolerance envelope — the envelope is
protecting long-sequence attention drift, not cross-hardware numerical
differences. Stretch attackers who tamper wholesale with hidden states
produce sketch diffs orders of magnitude larger than the tolerance,
so the sketch alone is sufficient to reject gross tampering. Subtle
per-position tampering within the tolerance envelope is caught by the
combined nine-stage pipeline (stages 8 + 9).

### 3.3 Position index derivation

The validator checks only a sampled subset of positions, not the entire
sequence. The subset is derived deterministically from the token
content + per-window randomness:

```
open_randomness = sha256(randomness || ":open")
token_hash = sha256(int_to_bytes(tok) for tok in tokens)
seed = PRF("open", token_hash, open_randomness, out_bytes=32)
indices = sorted(Random(seed).sample(range(seq_len), CHALLENGE_K))
```

with `CHALLENGE_K = 32`. A miner cannot predict which positions the
validator will check until after committing the sketch + tokens
together, so targeted per-position tampering requires tampering 32
positions (not 1) without the validator noticing.

## 4. Nine-stage verifier pipeline

Every submission flows through a strict pipeline of ordered stages.
Any stage returning `reject` short-circuits downstream; any stage
returning `soft_flag` records the flag in the verdict explanation but
does not reject (used for distribution drift where the validator is
uncertain).

1. **Schema**: validate the submission envelope structure.
2. **Tokens**: token IDs in-range for the model vocabulary; sequence
   length within the model's max context; deterministic tokenization.
3. **Prompt**: miner's declared prompt + chat-template application
   re-derives the tokenized prompt.
4. **Proof**: sketch re-verification per §3.2.
5. **Termination**: completion ended at EOS or at `max_new_tokens`,
   not mid-stream.
6. **Environment**: env-specific task evaluation (permissionless env
   registry dispatch; first-class entries: SAT, arithmetic).
7. **Reward**: computed against the declared env's reward function
   (in sandbox).
8. **Logprob**: per-position logprob replay; fails if `≥ 51%` of
   positions are drift < `LOGPROB_DRIFT_THRESHOLD = 0.15` log-prob
   units.
9. **Distribution**: median importance ratio between the miner-declared
   logprobs and the validator-replayed logprobs must lie within
   `[0.85, 1.15]`.

Each stage emits to `validator/metrics.py` counters:
`reliquary_verifier_stage_total{stage, result}` +
`reliquary_verifier_rejections_total{stage, reason}` +
`reliquary_verifier_soft_flags_total{stage, reason}`. The Epic 5
dashboards render these in real time.

## 5. Mesh consensus

### 5.1 Stake-weighted median

Per-completion verdicts from multiple validators are aggregated via
`stake_weighted_median` over the validators' `acceptance_score` field.
Stake is capped at `MESH_STAKE_CAP_FRACTION = 0.10` of the total mesh
stake before aggregation so no single validator — even one with
overwhelming stake — can dominate a single consensus decision. Quorum
requires the cumulative stake of participating validators to cross
`MESH_MIN_QUORUM_STAKE_FRACTION = 0.50` of the capped total.

### 5.2 Outlier detection

Each validator's full verdict score vector is compared against the
stake-weighted median vector in euclidean distance; validators whose
distance exceeds `MESH_OUTLIER_THRESHOLD = 0.25` are flagged as
outliers for the completion. Per-validator disagreement rate is the
fraction of completions a validator was an outlier on in the window.

### 5.3 Gating

Validators whose disagreement rate in a window exceeds
`MESH_OUTLIER_RATE_GATE = 0.05` (5%) are gated: dropped from the median
computation for `COPYCAT_GATE_DURATION_WINDOWS = 12` subsequent
windows. Gating is automatic and reversible — a gated validator that
returns to in-distribution behavior resumes contributing after the
cooldown.

The Tier 2 Epic 1 v2 mesh-live audit empirically validated this path:
four validators (three honest Blackwell hosts + one simulated malicious)
produced 256 verdicts across one window; the malicious validator's
disagreement rate was 1.0 and it was cleanly gated without affecting
consensus.

## 6. Copycat detection

Directional copycat detection (Tier 1 Epic 3) distinguishes the
first-submitter of a near-duplicate from the copier. The first
submission within a rolling time window claims the canonical position;
subsequent submissions with sha256-equal completion or high-similarity
index-based duplication are flagged as copies and lose their position.

An *ambiguity window* (2 seconds, `COPYCAT_AMBIGUITY_WINDOW_SECONDS`)
absorbs clock skew between submission timestamps. When two submissions
fall inside this window, neither is flagged — operator-grade clock
drift cannot slash a legitimate first-submitter.

Hysteresis on the gate: once flagged for copycat above
`COPYCAT_WINDOW_THRESHOLD = 0.05`, a miner stays gated for
`COPYCAT_GATE_DURATION_WINDOWS = 12` windows to prevent a miner from
oscillating in and out of penalty.

## 7. Artifact storage and commitment

### 7.1 Content-addressable pipeline

Every artifact a validator signs (`VerdictArtifact`) is published
through the signed-verdict storage pipeline (Tier 2 Epic 1 residual):

```
verdicts/<netuid>/<window_id>/<validator_hotkey>/<completion_id>.json
```

The envelope is `{payload_json, signature, signer_id}`; `payload_json`
is the canonical-JSON bytes of the verdict (sort_keys, compact
separators, ensure_ascii). Signatures use HMAC-SHA256 (v1); a later
version can migrate to Ed25519 per-validator keys without changing
the envelope shape. Pluggable `StorageBackend` Protocol supports
local filesystem, R2/S3, or GCS.

### 7.2 Delta checkpoints

Policy checkpoints are published as **delta bundles** via Tier 2 Epic 2:
the weight diff between two consecutive checkpoints, quantized per-
tensor to int8 with a per-tensor scale, with every shard carrying a
sha256 payload hash and the bundle carrying a Merkle root over the
sorted-by-tensor-name payload hashes. Full snapshots land every
`DEFAULT_FULL_SNAPSHOT_CADENCE_WINDOWS = 64` windows (configurable).

`fetch_bundle` reverifies every shard sha256 + recomputes the Merkle
root before returning a `DeltaBundle`. Fail-before-mutate: any shard
or Merkle failure raises before any reconstructed state exists.

Shard-parallel download (Tier 2 Epic 2 residual) pulls up to 8 shards
concurrently via `ThreadPoolExecutor`.

### 7.3 Onchain Merkle commit

Every window, the Forge validator commits the delta checkpoint's
`merkle_root_hex` onchain under the namespaced key
`reliquary_checkpoint_<subnet>`. Every window, the Ledger validator
commits the stake-weighted median verdict Merkle root under
`reliquary_mesh_verdicts_<subnet>`. An external auditor can reconstruct
the entire subnet's history by:

1. Reading the commitments at a block height.
2. Reading the matching R2 blobs.
3. Reverifying every sha256 + Merkle root + signature.

Cf. `reliquary_inference/chain/merkle_commit.py`.

## 8. Permissionless environment registry

Tier 3 Epic 2 opens the task space: any author can submit an
`EnvSpec` (`reliquary/envs/spec.py`) whose code (task_generator +
reward_function + evaluator) is hash-committed, along with a
sample_task + sample_solution + expected_reward. Validators re-run
the sample in the sandbox and reject the submission if the observed
reward doesn't match expected_reward within `REWARD_TOLERANCE = 1e-6`.
Quorum of stake-weighted validators must accept before the env
transitions CANDIDATE → SHADOW, then after a shadow period with no
security incidents → ACTIVE.

Sandbox (`reliquary/envs/sandbox.py`) uses defense-in-depth:
RestrictedPython compile + AST audit + whitelist builtins +
subprocess isolation + rlimit caps + JSON-only return. An attacker
escape requires compromising all five layers.

First-class registry entries dogfood the pipeline: `sat_classic`
(3-SAT) and `arithmetic_basic` (add 12+5=17) both flow through the
validator's pre-quorum validation path.

## 9. Distributed training (Forge)

Tier 2 Epic 3 ships:

- **FSDP2 backend**: model sharding via `torch.distributed.fsdp.fully_shard`
  wrapped by `reliquary/training/distributed.py`.
- **DiLoCo outer loop**: snapshot → inner steps → all-reduce mean delta
  → outer-momentum apply. Paper defaults: `inner_steps=30`, `outer_lr=0.4`,
  `outer_momentum=0.95`.
- **GRPO inner loop**: `reliquary/training/grpo.py` — group-relative
  advantage computation against validator-produced verdict rewards;
  KL penalty against the reference policy.

### 9.1 DAPO zone filter

A rollout group is `(miner_id, task_id)` — M samples of one miner on one
prompt. Let `r₁, …, r_M ∈ [0,1]` be the correctness rewards for the M
rollouts (boxed-answer exact match on MATH, so r ∈ {0, 1} in the binary
case). Define the group's within-group reward standard deviation

$$\sigma = \sqrt{\tfrac{1}{M} \sum_{i=1}^{M} (r_i - \mu)^2}$$

where μ = mean(r). The zone filter accepts groups with σ ≥ σ_min and
rejects the rest. For binary rewards σ_min = 0.43 corresponds to
Bernoulli k ∈ [2, 6] out of 8 — the canonical DAPO / GRPO-OOA zone
where within-group contrast is informative. A bootstrap threshold
σ_min = 0.33 (k ∈ [1, 7]) admits weaker signal while the base model
is still learning.

Groups outside the zone carry no usable GRPO gradient (the normalized
advantage `a_i = (r_i − μ)/σ` collapses to zero or to numerical noise
when σ is tiny), so dropping them upstream saves Forge compute without
any information loss.

### 9.2 GRPO update

Forge's inner loop runs one optimizer step per B_BATCH = 8 in-zone
groups. Within each group the advantage is

$$a_i = \frac{r_i - \mu}{\sigma + \varepsilon}$$

The PPO-clipped surrogate is

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_i\left[\min\left(\rho_i \cdot a_i,\; \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \cdot a_i\right)\right]$$

with $\rho_i = \exp(\log\pi_\text{new}(a_i|s_i) - \log\pi_\text{old}(a_i|s_i))$.
`π_old` log-probabilities come from the miner's GRAIL commit payload so
Forge saves one forward pass per rollout. `π_new` is the current
Forge policy.

A KL penalty against a frozen reference policy π_ref is added via
Schulman's k3 unbiased estimator:

$$\mathrm{KL}(\pi_\text{new} \| \pi_\text{ref}) \approx \exp(\log\pi_\text{ref} - \log\pi_\text{new}) - 1 - (\log\pi_\text{ref} - \log\pi_\text{new})$$

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \beta \cdot \mathrm{KL}$$

Defaults: clip ε = 0.2, KL β = 0.04, learning rate 5e-7 (AdamW),
gradient clip at 1.0. Only a small subset of target tensors (attention
q/k/v projections in the first four layers) require gradient — the
rest of the 3B parameter surface is frozen for every training cycle
so the published delta stays compact (~20 MB).

### 9.3 Per-prompt cooldown

Once a prompt's rollout group enters a training batch, its
`dataset_index` is parked in a per-validator `CooldownMap` for 50
windows. The next task batch skips those indices via the
`window_context["cooldown_indices"]` hook. The cooldown map persists
atomically to local disk and mirrors to R2 so a validator rebuild
doesn't reset the curriculum-diversity guard.

### 9.4 Reparameterization-trick sanity guard

Before any delta applies to the cached model, every target tensor is
checked for:

- **Finiteness**: NaN or ∞ in any shard → reject.
- **Projection magnitude floor**: `mean(|w|) ≥ PROJ_MIN_MEAN_ABS`
  (default 1e-4) → catches the RMSNorm × Linear rescaling exploit that
  leaves one tensor scaled to ~0.
- **Per-layer scale-ratio bound**: within any `model.layers.N.*` prefix,
  `max(mean|w|) / min(mean|w|) ≤ LAYER_SCALE_RATIO_MAX` (default 1e5)
  → catches the paired half where the other tensor blew up.

Our attack surface is narrower than checkpoint-upload subnets because
Ledger only consumes HMAC-signed deltas from a trusted Forge trainer
(`BridgeVerifier` rejects commitments not on the policy-authority
allowlist). The reparam guard is defense-in-depth against compromised
trainer auth or malformed delta bundles that pass the signature +
smoke-hash gates.

### 9.5 Training metrics

`reliquary/training/metrics.py` emits Prometheus
(`reliquary_training_loss_last`, `reliquary_training_kl_last`,
`reliquary_training_grad_norm_last`, …) and optionally to W&B via the
`WandbSink` adapter. OpenTelemetry tracing wraps each step via
`traced_step` so operators can stitch a full distributed-training span
in Jaeger/Tempo.

## 10. Observability

Every load-bearing path in the validator + trainer emits Prometheus
counters + OTEL spans. Tier 2 Epic 5 shipped four Grafana dashboards
auto-provisioned via docker-compose:

- `reliquary-validator-mesh`: per-validator disagreement, gated
  cumulative, missing cumulative, acceptance rate.
- `reliquary-verifier-pipeline`: stage throughput, rejection rate per
  stage, top rejection reasons, soft flags.
- `reliquary-miner-scoreboard`: top-25 by acceptance rate, accept/reject
  timeseries, top rejection reasons across miners, per-miner score
  heatmap.
- `reliquary-training` (Forge): loss + rolling mean, advantage
  variance, KL, gradient norm, step time, totals.

All metrics + dashboards are versioned with the subnet code. OTEL
SDK install is optional; when absent the facade degrades to a silent
no-op so dev environments don't need to install full tracing deps.

## 11. Security properties

The following properties are pinned by the test suite and re-verified
at every CI run:

1. **Proof forgery infeasible**: a miner cannot submit a valid sketch
   for tokens they did not actually run through the policy model —
   they'd have to predict `r` in advance (PRF-bound to the window's
   randomness not known until submission).
2. **Signature forgery infeasible**: HMAC-SHA256 bind `s_vals` to the
   miner's secret; validators run `hmac.compare_digest` in constant
   time against tampered signatures.
3. **Cross-validator spoofing blocked**: a validator cannot sign a
   verdict as another validator — signatures bind to `signer_id`
   recorded in the envelope.
4. **Mesh-level minority cannot rewrite consensus**: stake cap + 50%
   quorum. Tier 2 Epic 1 v2 audit: a malicious 10%-stake validator
   was cleanly gated with no impact on consensus across 256 verdicts.
5. **Sandbox escape requires 5-layer bypass**: RestrictedPython +
   AST audit + builtins whitelist + subprocess + rlimit.
6. **Storage path traversal blocked**: both verdict + checkpoint
   backends resolve keys under `Path.root` + reject `..` + null-byte +
   absolute paths + symlinks escaping the root.
7. **Pickle unmarshal surface closed**: every payload crossing a
   process boundary is JSON-serializable only; no `torch.load`, no
   pickle.

## 12. Versioning + upgrade path

Protocol constants + canonicalization + signature primitives live in
the shared `reliquary-protocol` PyPI-shaped package (Tier 3 Epic 4).
Both runtimes pin an exact version. A protocol bump requires:

1. Bump `reliquary_protocol.VERSION`.
2. Land matching `pyproject.toml` pin bumps on both runtimes.
3. 24h testnet bake where both runtimes run the new protocol version.
4. Mainnet rollout at a pre-announced block height.
5. Validators commit upgrade intent onchain before the cutover.

Any runtime still on the old version after the cutover is considered
slash-eligible (policy to be wired into the Tier 4 Epic 1 governance
charter).

## 13. Empirical validation

- **Tier 1 Epic 6 baseline**: 1000 honest + 1500 adversarial trials on
  RTX PRO 6000 Blackwell with HIDDEN_DIM=256. Honest FP rate = 0.0000;
  all three adversarial classes at FN rate varying with the tamper
  magnitude, caught by the combined 9-stage pipeline even when the
  sketch alone cannot distinguish.
- **Tier 2 Epic 6 cross-GPU**: 4 hosts (3× RTX PRO 6000 Blackwell +
  1× H100) produce bit-exact sketch digest across 90 samples.
- **Tier 2 Epic 1 mesh-live v2**: 4 validators (3 honest Blackwell +
  1 simulated malicious) across 256 verdicts; 64/64 completions
  accepted; mesh-M disagreement 1.0 → gated.
- **Tier 3 Epic 1 distillation live**: asymmetric teacher (Qwen3-8B
  via vLLM on dev-server Blackwell) + student (Qwen3-4B-Instruct-2507
  via transformers on staging1 Blackwell); all 4 math answers correct;
  all 4 provenance bindings resolve.

## 14. Open questions

- **Sketch tolerance lower bound**: `PROOF_SKETCH_TOLERANCE_BASE = 6000`
  has substantial headroom in empirical cross-GPU runs; can it be
  tightened further without losing robustness to long-sequence
  attention drift?
- **Delta checkpoint compression ceiling**: int8 quantization yields
  ~50% of fp16; FP8 once stable across CUDA 13+ hardware could halve
  again.
- **WASM sandbox alternative**: RestrictedPython is a known-good
  baseline; wasmtime-py would give stronger process-level isolation
  but at the cost of forcing env authors to write compile-to-WASM
  code. Trade-off is worth revisiting if we see audit-worthy issues
  in the RestrictedPython path.
- **Signature migration to Ed25519**: HMAC-SHA256 is secure but
  symmetric; asymmetric signatures would let validators publish
  verdicts without pre-sharing secrets with the mesh. Deferred
  until the multi-sig coldkey flow stabilizes.

## References

- Source: `reliquary-inference` (Ledger runtime), `reliquary` (Forge
  runtime), `reliquary-protocol` (shared).
- Tier PRDs: `private/reliquary-plan/{00_STRATEGY,01_TIER1_PRD,
  02_TIER2_PRD,03_TIER3_PRD,04_TIER4_PRD}.md`.
- Clean-room spec docs: `private/reliquary-plan/notes/spec-*.md`.
- Mesh-live audits: `reliquary-inference/docs/audit/mesh_live/`.
- Cross-GPU audit: `reliquary-inference/docs/audit/cross_gpu/`.
- Empirical audit harness: `reliquary_inference.audit_harness`.

---

**Draft status**: this paper is a living document, bumped alongside the
shared protocol version. Reviewers should pin the commit hash they
reviewed against; future revisions may clarify or tighten claims but
will not silently weaken security properties.

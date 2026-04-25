# Reliquary: Proof-Carrying Inference, Permissionless at Every Layer

Reliquary ships a decentralized AI subnet where every inference comes
with a proof — a cryptographic fingerprint that lets any validator
confirm, in milliseconds, that a miner actually computed what they
claim to have computed. No re-execution. No trust assumption. No
gatekeeper.

## What's different

Most Bittensor-class subnets today accept miner outputs on faith —
validators re-run a sample and trust the rest. That model breaks down
once miner inputs become high-stakes: real training data, real
inference latency-bound applications, real economic rewards tied to
output correctness.

Reliquary's proof-carrying inference changes the trust model:

- **Every completion carries a sketch**: a compact hidden-state
  fingerprint of the miner's forward pass at the declared proof
  layer, bound by HMAC to the miner's identity.
- **Validators verify in milliseconds**: one forward pass at sampled
  positions, not the full decode. Sub-millisecond per position on
  commodity GPUs.
- **Cross-hardware determinism**: the same sketch verifies bit-
  identically on A100, H100, Hopper, and Blackwell — empirically
  validated across 4 hosts at 90 samples.
- **Content-addressable end-to-end**: every verdict, every checkpoint,
  every env spec is identified by sha256 over its canonical bytes.
  External auditors can reconstruct the subnet's state from any
  block-height snapshot without talking to a single validator.

## What's on the subnet

**Ledger** — the inference runtime. Miners generate completions
from the authoritative policy model; validators verify through a
nine-stage pipeline (schema / tokens / prompt / proof / termination /
environment / reward / logprob / distribution) with explicit stake-
weighted median consensus across a mesh of validators.

**Forge** — the training runtime. Policy-authority
committee publishes candidate checkpoints; miners propose training
improvements; validators verify via shared proof primitives and the
same artifact-canonicalization rules. Distributed training uses FSDP2
sharding + DiLoCo outer-loop aggregation.

Both runtimes share a single `reliquary-protocol` package, version-
pinned to the same bytes, so artifacts hash identically wherever they're
produced.

## Permissionless at every layer

- **Miners**: any hotkey can submit proofs. No allowlist. Accept or
  reject is determined by protocol, not governance.
- **Validators**: any hotkey with stake can participate in the mesh.
  Stake is capped at 10% of total per-completion so no single large
  validator can dominate consensus.
- **Envs**: any hotkey can submit a new task environment via the
  `EnvSpec` registry. Authors stake TAO as spam prevention + skin-
  in-game; validators re-run the author's sample in an isolated
  sandbox and reject before quorum if the reward doesn't match the
  author's declared expected value. Approved envs run in shadow mode
  before promotion to full reward flow.
- **Protocol upgrades**: coordinated via a shared package version
  bump with a 24h testnet bake + pre-announced mainnet block height.
  No validator can force an upgrade unilaterally.

## What's empirically true

- **Cross-GPU bit-exactness**: 4 hosts (3× Blackwell + 1× H100),
  bit-identical sketch digests across 90 samples. One commitment,
  four independent verifications, zero drift.
- **Mesh consensus works**: 4-validator live audit with 3 honest
  Blackwells + 1 simulated malicious validator across 256 verdicts
  — malicious validator cleanly gated with 1.0 disagreement rate, no
  effect on consensus.
- **Distillation lane is live**: asymmetric Qwen3-8B teacher (dev
  server) + Qwen3-4B-Instruct-2507 student (staging fleet), all
  math answers correct, all provenance bindings resolve.
- **Audit harness ready for 100K**: 100K honest + 10K adversarial
  trial runner with resume / progress / checkpointing for unattended
  multi-day runs on the Blackwell fleet.

## Launch model

Qwen3-4B as the first policy under Ledger. Qwen2.5-7B or the next
Qwen reasoner under Forge. Model choice is a subnet-governance
parameter, not a protocol parameter — future upgrades are voted
onchain without touching the verifier.

## What you do if you want to participate

**As a miner**: clone the `reliquary-inference` repo, install the
`gpu` extras, point a hotkey at the subnet. The quickstart CLI
handles registration + policy download + proof submission. GPU
requirements: one commodity-grade GPU (RTX 4090 / A100 / Blackwell)
per miner hotkey.

**As a validator**: stake on the subnet, run the nine-stage verifier
(same repo, different CLI entry). GPU requirements: one Blackwell or
H100 per validator hotkey; 256 GB RAM recommended for verifier
pipeline with cache-warmed metagraph.

**As an env author**: write your `task_generator` / `reward_function` /
`evaluator` as pure-Python source strings that pass the sandbox audit.
Submit via `reliquary-env submit`. Stake TAO as commitment; your env
runs in shadow mode before promotion.

## Roadmap

- **Tier 1** ✅ — proof protocol + nine-stage verifier + chain
  hardening + bootstrap infrastructure (shipped).
- **Tier 2** 🟢 — validator mesh + distributed training + delta
  checkpoints + observability + empirical audit baseline + multi-
  sig ownership (shipped on code; 100K campaign in progress).
- **Tier 3** 🟡 — permissionless env registry + closed-loop
  runtime + shared protocol package + Forge runtime activation (schema
  + sandbox shipped; onchain submission + Forge runtime launch pending).
- **Tier 4** ⏳ — governance charter + protocol paper review +
  validator recruitment + public comms + tokenomics (this doc
  is Epic 4; paper is Epic 2).

## Links

- Source: [reliquary-ledger](https://github.com/reliquadotai/reliquary-ledger),
  [reliquary-forge](https://github.com/reliquadotai/reliquary-forge),
  [reliquary-protocol](https://github.com/reliquadotai/reliquary-protocol).
- Protocol paper: `docs/paper/reliquary_protocol_paper.md`.
- License: MIT (see each repo's LICENSE).
- Status dashboards + mesh audits: `docs/audit/`.

---

*Last updated 2026-04-18. This document is versioned with the protocol
package; expect revisions.*

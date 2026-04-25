# Reliquary — Anti-Centralization Commitments

Reliquary claims to be a permissionless, decentralized protocol. This
document pins what that means operationally, what we commit to, and
what we explicitly DO NOT control. It is an input to the Tier 4 Epic
1 governance charter.

## Summary

No single human, no team, no committee can:
- Force a protocol upgrade.
- Gate who runs a miner.
- Gate who runs a validator.
- Gate who submits a task environment.
- Overwrite consensus.
- Seize staked TAO.
- Slash a validator outside the mechanical protocol rules.

Everything that happens on-subnet happens because the protocol allows
it. Governance actions happen because a quorum of stake-weighted
signers commit them onchain. Every hash a validator accepts is
auditable from a block-height snapshot + public R2 blobs. Every
protocol parameter is pinned by test in a public repository.

## What we commit to

### 1. Permissionless participation

**Miner onboarding**: any Bittensor hotkey with a registered stake
can submit proofs. There is no allowlist, no KYC, no application
process. Acceptance is determined entirely by the nine-stage verifier
running on validators — accept / reject is a function of the submitted
bytes, not the submitter's identity.

**Validator onboarding**: any hotkey with the minimum stake (set via
Bittensor's built-in subnet parameters, not Reliquary governance) can
run the verifier and participate in mesh consensus. Their verdicts
carry weight proportional to stake, capped at 10% of the mesh total
per-completion so no single whale can dominate.

**Env onboarding**: any hotkey can submit a new `EnvSpec` to the
permissionless registry (Tier 3 Epic 2). No prior relationship with
maintainers. Validators verify the spec + sample in the sandbox and
vote quorum-wise for admission. An env the validators accept becomes
canonical without any human review beyond the validator mesh.

### 2. Multi-sig coldkey from day zero

The subnet coldkey is a multi-sig (initially 3-of-5; signers listed
in the governance charter on registration). No single signer can:
- Submit weights.
- Change subnet parameters.
- Accept subnet ownership.
- Transfer delegated stake.

All of the above require the threshold number of independent
signatures via btcli. The multi-sig drill is rehearsed on testnet
(see `reliquary_inference/chain/multisig.py` + its test suite) before
any mainnet action.

### 3. Content-addressable audit trail

Every artifact the subnet accepts is:
- Identified by sha256 over its canonical JSON form.
- Stored at a predictable R2 path under the artifact id.
- Committed onchain under a namespaced `reliquary_<kind>_<subnet>` key.

External auditors reconstruct subnet state by:
1. Reading the onchain commitments at a block height.
2. Reading the matching R2 blobs.
3. Reverifying every sha256 + signature + Merkle root.

The protocol does NOT rely on any trusted validator to mirror artifacts.

### 4. Protocol upgrade coordination

Upgrades require:
1. `reliquary_protocol.VERSION` bump in the shared package.
2. Matching `pyproject.toml` pin bumps on BOTH runtimes in a single PR
   each.
3. 24h testnet bake where both runtimes run the new protocol version.
4. Mainnet rollout at a pre-announced block height.
5. Validators commit upgrade intent onchain BEFORE the cutover.

Any validator still running the old protocol version after the cutover
is slash-eligible under the governance charter. The mechanical check
is automatic: canonical bytes that don't match across runtimes produce
orphan verdicts the other side rejects.

### 5. Opt-out always possible

No registration lock-in. A miner, validator, or env author can stop
participating at any time with no protocol penalty beyond losing
future emission share. Stake is not confiscated; unstaking follows
Bittensor's built-in unstake cooldown, not Reliquary rules.

### 6. No privileged validators

All validators run the same verifier code. There is no "genesis
validator" with bypass rights. The protocol does not distinguish a
validator based on when they registered, their geographic location,
their hardware vendor, or their relationship with maintainers.

If the founding maintainers' own validator hotkeys misbehave, they
are gated by the same disagreement-rate threshold as any other. We
explicitly commit to not add exemptions or heuristics that favor
known-good operators.

### 7. Open protocol paper + open audit

The protocol paper (`docs/paper/reliquary_protocol_paper.md`) is a
public document. The audit harness, empirical reports, cross-GPU
determinism data, and mesh-live consensus runs are all published
under `docs/audit/` with raw inputs + outputs so any reader can
reproduce the numbers on their own hardware. All audit findings —
critical, major, minor — are published with the reviewer's identity
(if they consent) and the commit hash they reviewed. No findings are
suppressed.

## What we explicitly do NOT commit to

### Not a financial product

Reliquary is a protocol, not a registered investment or financial
product. Staking TAO on a Reliquary subnet is a Bittensor-layer
action under Bittensor's rules, not a Reliquary contract. No
maintainer has fiduciary duty to stakers. Participants operate at
their own risk.

### Not a data processor

Reliquary does not store user data beyond what miners voluntarily
include in a completion. Maintainers do not have access to any
private keys, wallet secrets, or off-subnet data. Anyone running a
miner or validator is solely responsible for the data they submit.

### Not a KYC platform

No identity is verified at the protocol layer. Bittensor hotkeys are
pseudonymous; we do not (and cannot, and commit not to) resolve them
to real-world identities.

### Not a content moderator

Reliquary does not filter, censor, or moderate miner outputs beyond
the mechanical protocol rules (proof validity, environment-specific
reward, distribution sanity). If a miner generates output that a
jurisdiction considers problematic, that is the miner's responsibility
to comply with applicable law — maintainers are not moderators.

### Not a cryptocurrency exchange

Reliquary has no token. All economic activity is denominated in TAO
under Bittensor's subnet emission curve. Maintainers do not operate
an exchange, do not list Reliquary on any DEX, and do not maintain
any liquidity pool.

### No special validator relationships

The maintainers' own validator hotkeys compete under the same rules
as any other validator. Maintainers commit to not:
- Run more validators than the stake cap + quorum math supports.
- Operate side-channel communication with other validators to
  coordinate consensus (side-channel = anything outside the onchain
  + mesh-audit surface).
- Accept payment or favors for tilting consensus.

Violation of these commitments is a reason for the community to
fork Reliquary under a different name and let the original hotkeys
be gated.

## How to verify these commitments

- Every protocol parameter is pinned by a test in the public repos.
  Check `tests/test_*_constants.py` + `tests/test_shared_protocol_alignment.py`.
- Every signer in the multi-sig is named onchain at registration.
- Every protocol commitment in this document is either:
  - enforced by a test (we point to the test),
  - a policy statement (we say "we commit to"),
  - a roadmap item (we say "we plan to").

If you find a commitment that's not backed by a test or onchain rule,
open a GH issue tagged `governance-gap` — we treat those as P0 bugs.

## Timeline of commitments

- **Now (2026-04-18)**: this document; multi-sig drill tested;
  protocol paper draft; content-addressable artifact trail working;
  permissionless env registry schema + sandbox shipped.
- **After mainnet activation**: onchain governance charter published;
  upgrade-intent commitment mechanism wired; validator recruitment
  plan public.
- **After 100K audit campaign**: findings published;
  mitigations merged; protocol version bumped.
- **Forge activation**: both runtimes running at production cadence,
  shared protocol package at a stable version, end-to-end audit
  artifacts published.

---

*This document is the Tier 4 Epic 7 commitment artifact. It is
versioned with the protocol and updated as anti-centralization
invariants are tightened. Any weakening of commitments requires an
explicit changelog entry + a protocol version bump.*

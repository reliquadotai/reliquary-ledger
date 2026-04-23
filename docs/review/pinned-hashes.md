# Pinned commit hashes

The artifact under review is the following four-repo configuration.
The reviewer **pins these exact SHAs in the report**; a report
against any other set is out-of-spec.

| Repo | SHA | Latest change |
|---|---|---|
| [`reliquary-protocol`](https://github.com/0xgrizz/reliquary-protocol) | `8e03d28` | `feat(eval): EvalBundle artifact + deterministic holdout derivation` |
| [`reliquary-inference`](https://github.com/0xgrizz/reliquary-inference) | `c827d5b` | `feat(dataset): GSM8K task source + MixedTasksSource for pool sustainability` |
| [`reliquary`](https://github.com/0xgrizz/reliquary) | `18957d0` | `feat(eval): rolling discovery index for EvalBundles` |
| [`reliquary-web`](https://github.com/0xgrizz/reliquary-web) | `57349f2` | `feat(dashboard): EvalPanel — MATH holdout accuracy chart + per-level breakdown` |

Dated **2026-04-23** (start of the 28-day continuous-operation track
record toward mainnet cutover).

## How to verify

```bash
# Clone + pin
git clone https://github.com/0xgrizz/reliquary-protocol.git
cd reliquary-protocol && git checkout 8e03d28 && cd ..

git clone https://github.com/0xgrizz/reliquary-inference.git
cd reliquary-inference && git checkout c827d5b && cd ..

git clone https://github.com/0xgrizz/reliquary.git
cd reliquary && git checkout 18957d0 && cd ..

git clone https://github.com/0xgrizz/reliquary-web.git
cd reliquary-web && git checkout 57349f2 && cd ..

# Confirm clean state
(cd reliquary-protocol && git status --short && git rev-parse HEAD)
(cd reliquary-inference && git status --short && git rev-parse HEAD)
(cd reliquary && git status --short && git rev-parse HEAD)
(cd reliquary-web && git status --short && git rev-parse HEAD)
```

Each `git rev-parse HEAD` must output the SHA above.

## Why these specific commits

- **`reliquary-protocol@8e03d28`** — first commit with the full
  four-artifact bridge surface (`RolloutBundle`, `CheckpointAttestation`,
  `PolicyCommitment`, `EvalBundle`). Version pinned at `0.3.0`.
- **`reliquary-inference@c827d5b`** — Ledger runtime with MATH
  holdout guard, GSM8K + MixedTasksSource, and the full 9-stage
  verifier pipeline. Consumes `reliquary-protocol>=0.3.0`.
- **`reliquary@18957d0`** — Forge runtime with live GRPO trainer,
  closed-loop bridge publisher, eval harness + rolling discovery
  index.
- **`reliquary-web@57349f2`** — public dashboard consuming the
  above. Useful for the reviewer to visually confirm state during
  testing but not on the critical-path surface.

## What's NOT covered by the pin

- **The HMAC policy authority secret** — lives in operator secret
  storage, not in any repo. The reviewer verifies the verification
  logic against a test secret, not against the production one.
- **The on-chain subnet identity commit history** — verifiable via
  `subtensor.get_commitments(netuid=462)`; the reviewer re-derives
  the commitment chain independently.
- **The live R2 artifact corpus** — the reviewer fetches artifacts
  from the public bucket at
  `https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev/` (read-only,
  unauthenticated).
- **GPU hardware** — the Tier 2 Epic 6 cross-GPU determinism result
  (`docs/audit/cross_gpu/`) is reproducible on any modern CUDA GPU
  with `flash-attn2` + `bf16`; results against different hardware are
  welcome as supplementary appendix material.

## Updating the pin

If the maintainers ship additional changes between package delivery
and review completion, the **original pin stays canonical**. The
reviewer may include an appendix addressing the post-pin changes but
is not obligated to.

If a critical finding lands during review, the maintainers may
deliver a patch-pin bump (e.g. `8e03d28 → 8e03d28+fix1`). That bump
is documented in this file's git history; only the text of this
file is authoritative at review-close time.

---

*Previous review pins archived at the bottom of this file below the
separator as they land.*

---

<!-- No prior review pins yet — this is the first external review. -->

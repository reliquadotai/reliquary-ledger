# Reliquary Overview

## Problem

Bittensor can reward useful model outputs, but many inference subnet designs leave a gap between what miners claim they generated and what validators can cheaply verify. That gap grows when the subnet also wants deterministic tasks, uniqueness incentives, and public auditability.

Reliquary narrows that gap by turning **accepted proof-carrying completions** into the commodity instead of raw token volume.

## Operating Model

Each accepted completion is:

- tied to a deterministic task batch
- bound to a declared task source
- signed by the miner identity
- replayable by validators with the same hidden-state extraction path
- filtered for duplicates and copycats
- traceable through manifests and public score summaries

```mermaid
flowchart LR
    T["Deterministic Task Batch"] --> M["Miner"]
    M --> C["Completion + Proof Commitments"]
    C --> S["Signature Binding"]
    S --> B["Completion Bundle"]
    B --> V["Validator Replay"]
    V --> U["Unique Accepted Work"]
    U --> W["Set Weights"]
```

## Miner Path

Miners win by submitting completions that are:

- valid for the active task source
- consistent with the proof commitments
- unique against competing submissions
- early enough to beat copycats on deterministic datasets

The scoring surface is intentionally narrow: unique accepted work gets paid, not raw token count.

## Validator Path

Validators do not trust miner output at face value. They:

1. rebuild or load the exact task batch for the window
2. verify the miner signature and binding fields
3. replay the shared forward path
4. challenge-check the hidden-state sketch commitments
5. reject duplicate nonces, duplicate digests, and dataset copycats
6. publish weights only for accepted unique work

The validator surface stays strict and machine-readable through `verdict`, `scorecard`, and `window_manifest` artifacts.

## Chain Role

Bittensor provides:

- miner and validator identity
- metagraph membership
- weight publication
- incentive coordination

The chain is used for **weights and references**, not as a bulk artifact store. Artifacts stay in object storage or local registry backends, and the public audit surface summarizes what happened per window.

```mermaid
flowchart TD
    B["Task Batch"] --> M["Miner Bundle"]
    M --> V["Validator"]
    V --> S["Scorecard"]
    S --> C["Bittensor Weight Publish"]
    V --> WM["Window Manifest"]
    WM --> A["Audit Index"]
    WM --> O["Private Artifact Storage"]
    A --> P["Public Audit Bucket"]
```

## Live Runtime

The current repository includes:

- local proof-complete demo paths
- live Bittensor `test` reads and writes on **netuid 462**
- single-GPU HF miner with M-batch generate (5-7× faster than serial)
  on a real RTX 6000 Blackwell staging box
- 4-validator mesh on testnet with batched proof verify
- R2-backed artifacts with retry + exponential backoff
- public audit index rebuilt every 10 minutes
- Prometheus `/metrics` + JSON `/healthz` + `/status` + static `/dashboard`
- **live task source**: `math` (Hendrycks MATH via `qwedsacf/competition_math`);
  legacy `reasoning_tasks` + `dataset_prompts` retained as low-resource
  fallbacks
- DAPO zone filter (σ ≥ 0.33 bootstrap threshold) + 50-window per-prompt
  cooldown for curriculum diversity
- closed-loop bridge to Forge: `CheckpointAttestation + PolicyCommitment`
  consumed by `policy_consumer` at `effective_at_ledger_window`
- reparam-trick sanity guard on delta apply (finite / magnitude /
  layer-scale ratio checks)

See [status.md](status.md) for the current live snapshot.

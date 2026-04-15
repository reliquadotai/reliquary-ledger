# Architecture

Reliquary is an inference-only subnet runtime with two pluggable task sources:

- `dataset_prompts`
- `reasoning_tasks`

## Window Flow

1. A chain adapter resolves the current window and public randomness.
2. A task source builds a deterministic `task_batch`.
3. Miners generate proof-carrying `completion` artifacts.
4. Validators replay the forward path, verify commitments, and run soft checks.
5. A `scorecard` artifact computes final miner weights.
6. A `window_manifest` captures all storage refs and the chain publish result.

## Verification Layers

- Hard checks:
  - proof version
  - signature
  - prompt or task binding
  - nonce uniqueness
  - hidden-state sketch commitment replay
- Soft checks:
  - duplicate task submission
  - cross-miner dataset copycat
  - duplicate completion digest
  - reasoning contamination tag overlap

## Task Sources

`dataset_prompts` emphasizes deterministic prompt selection and copycat filtering.

`reasoning_tasks` provides a deterministic arithmetic task family with exact-answer validation and contamination tags.

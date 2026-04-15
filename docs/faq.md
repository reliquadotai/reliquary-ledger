# FAQ

## Why keep the runtime inference-only?

Because it keeps the subnet legible. An inference runtime with deterministic tasks, replayable proofs, and explicit manifests is easier to audit and operate than a larger stack that mixes training, packing, and policy promotion into the first public surface.

## Why do proofs matter here?

They reduce the trust gap between what miners claim and what validators can replay. Reliquary does not just score completions; it scores completions that stay bound to deterministic tasks, signature bindings, and hidden-state sketch checks.

## Why support both `reasoning_tasks` and `dataset_prompts`?

They cover two complementary markets:

- `reasoning_tasks` gives a fully deterministic task family with exact-answer validation
- `dataset_prompts` gives a dataset-index-bound prompt market with explicit copycat pressure

Each window activates one task source at a time so validator logic stays source-specific and easy to inspect.

## Why use Bittensor instead of a normal API marketplace?

Bittensor is the coordination and incentive layer. It provides subnet membership, identity, and weight publication. Reliquary keeps large artifacts off-chain and uses the chain for what it is good at: coordination and incentives.

## Why are artifacts stored off-chain?

Artifacts are larger, more numerous, and more inspectable than what belongs directly on-chain. Reliquary uses object storage and manifests for the artifact layer, then publishes weights and compact references through the chain.

## Why is the repository named `reliquary-inference` if the product name is Reliquary?

The repository name identifies the inference runtime package and deployment surface. The product-facing name is Reliquary.

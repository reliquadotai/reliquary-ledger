# Public Release Checklist

Use this checklist before treating the repo as a public-facing subnet surface.

## Repository

- `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, issue templates, and PR template are present
- CPU CI passes on `main`
- overview, FAQ, status, and deployment docs are current
- no operator-only secrets or host-specific details remain in public docs

## Technical Validation

- local `pytest` passes
- local `py_compile` passes
- readonly testnet smoke passes
- real-model readonly smoke passes
- live miner/validator cycle succeeds
- audit publish succeeds

## Storage Posture

- main artifact storage is configured intentionally
- public audit target is configured intentionally
- raw artifact URLs are exposed only if explicitly desired
- public audit URLs resolve correctly

## Operator Readiness

- dedicated websocket endpoint decision is documented
- runtime status command works
- rollback steps are documented
- release notes clearly explain testnet scope and known limitations

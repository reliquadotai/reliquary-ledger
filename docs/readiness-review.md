# Readiness Review

Review date: 2026-04-15

This review covers both repositories with the inference runtime as P0 and the broader platform as P1.

## Blocker

- None after this pass.

## Must-Fix Before Public Release

- Operator-private staging details were present in public docs.
  - Fixed by replacing host-specific handoff notes with sanitized public status and deployment guidance.
- Public audit and raw artifact storage were too easy to read as one public surface.
  - Fixed by adding separate audit-bucket configuration, default raw-artifact URL suppression, and explicit docs for private artifacts plus public audit.
- Public copy still carried comparative and positioning language.
  - Fixed by rewriting the public docs around Reliquary as a self-contained product and removing lineage/comparison wording.
- `reliquary-forge` still exposed tone debt in README and blueprint docs.
  - Fixed by rewriting the README and the lightweight blueprint in neutral product language.

## Post-Release Hardening

- Pin a dedicated Bittensor websocket endpoint for long-running public staging.
- Split the live public audit onto a dedicated public bucket or domain in production, not just at the configuration layer.
- Expand public status automation so the status page can be refreshed from CI or a scheduled job.

## Nice-To-Have

- richer public audit views with charts and miner history
- automated screenshot refresh for the audit preview asset
- additional operator playbooks for multi-node staging

## Repo Outcome

- `reliquary-ledger`: inference runtime and live subnet surface for Reliquary
- `reliquary-forge`: broader platform for verified inference, distillation, and RL

# Review Findings

This document captures the current technical review across `reliquary-inference` and the broader `reliquary` platform.

## Blocker

- None identified in the current testnet staging path.

## Must-Fix Before Wider Operator Rollout

- Monitoring is private-first and localhost-bound by design; operator docs must keep SSH tunneling as the default access path.
- The broader `reliquary` platform metrics depend on the FastAPI control plane being present on the monitored host; operators should treat that target as optional, not required for the live inference subnet.

## Post-Rollout Hardening

- Add alert delivery routing once the on-node Prometheus/Grafana stack has been observed for a sustained period.
- Consider a dedicated public audit bucket or custom domain if stricter separation between raw artifacts and public audit pages becomes an operational requirement.
- Add retention and backup policy for Prometheus data if the node moves from staging to a longer-lived operational role.

## Nice-To-Have

- Extend the inference exporter with per-task-source counters over longer rolling windows.
- Add richer learner and benchmark panels if the broader `reliquary` control plane becomes part of routine staging operations.
- Add a small operator landing page that links runtime status, Grafana, Prometheus, and the public audit index from one localhost-only view.

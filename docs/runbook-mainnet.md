# Reliquary Ledger — Mainnet Runbook

Incident response for Reliquary Ledger on Bittensor finney. Use alongside `docs/deployment.md` (environment setup) and `docs/protocol.md` (what the subnet does).

## Health states

The `/health` endpoint (see `reliquary_inference.shared.health`) reports one of:

| State | Meaning | Operator action |
|---|---|---|
| `ok` | All component checks passing | none |
| `degraded` | Non-fatal stale component (e.g. chain resync in progress) | observe; page only if state persists > 10 min |
| `unhealthy` | Fatal component failure (model not loaded, chain > 3 min stale) | page immediately |

Health state is rolled up from individual checks: `model`, `chain`, `last_window`, `proof_worker`.

## Scenario: dead websocket endpoint

Symptom: `chain` check age climbs; `set_weights_with_retry` returns `success=False` with `ConnectionError` in `last_error`.

1. Check cloudflare-fronted endpoint reachability: `curl -I https://entrypoint-finney.opentensor.ai:443` — expect 426.
2. Fail over to an archive node. Edit `/etc/reliquary-ledger/sn81.env` and set `RELIQUARY_INFERENCE_CHAIN_ENDPOINT` to a known-good endpoint (mainnet archive-finney preferred; document fallbacks as they are published by the team).
3. Restart: `systemctl --user restart reliquary-ledger-validator-mainnet`.
4. Confirm `chain` check returns to `ok` within 120 seconds.

## Scenario: proof verification spike

Symptom: `reliquary_verifier_stage_total{stage="proof",result="rejected"}` rises sharply; miner complaints.

1. Correlate with model version. A mid-window policy switch can produce transient proof mismatches until miners catch up. Check `subtensor.commit` for a recent `reliquary_policy_*` entry.
2. If no policy change: inspect the latest rejected completion via `reliquary-inference audit show --completion-id <id>`. Look at `proof_summary.sketch_diff` vs `sketch_tolerance`.
3. Cross-GPU drift: confirm validator and miner share the same CUDA/cuDNN/torch versions. Hardware drift is the #1 root cause.
4. Escalate only if drift exceeds documented per-hardware tolerances (Tier 2 Epic 6 empirical audit values); otherwise widen window tolerance via protocol upgrade (governance action).

## Scenario: weight-set failure

Symptom: validator's window closes but weights don't land onchain. `reliquary_chain_set_weights_total{result="failed"}` increments.

1. Inspect validator log for `WeightSubmissionResult(success=False)`. Look at `last_error`.
2. Common causes:
   - `no_matching_hotkeys` — metagraph doesn't see expected miner hotkeys; check hotkey registration.
   - `txpool full` — rate-limited; retry already applied; widen `RetryPolicy.max_attempts` to 5 temporarily.
   - `substrate.InvalidTransaction` — inspect the rejection reason string (usually under `last_error`).
3. Manual last-resort: `btcli subnet metagraph --netuid $RELIQUARY_NETUID --network finney` to confirm validator is registered and active; `btcli stake add` if stake has dropped below threshold.

## Scenario: wallet compromise

Symptom: unexpected transactions from validator hotkey; stake movement not initiated by operators.

1. Stop both services immediately:
   `systemctl --user stop reliquary-ledger-validator-mainnet reliquary-ledger-miner-mainnet`
2. Rotate hotkey:
   - `btcli wallet new_hotkey --wallet.name reliquary-ledger --wallet.hotkey validator-next`
   - `btcli stake add` to the new hotkey; `btcli stake remove` from compromised one.
   - Update `HOTKEY_NAME` in env file; restart services.
3. File incident postmortem within 7 days.
4. If coldkey compromise is suspected: follow Bittensor's standard subnet-owner key-recovery procedure (out of runbook scope).

## Scenario: health endpoint reports UNHEALTHY after restart

Model not loaded or chain never connected. Expected during first 90 seconds of startup; `unhealthy` > 3 minutes means a genuine failure.

1. `journalctl --user -u reliquary-ledger-validator-mainnet -n 200` — find the failing subsystem.
2. Model loading failures: check GPU availability (`nvidia-smi`), CUDA version, `RELIQUARY_INFERENCE_MODEL_REF` spelling, HuggingFace cache integrity.
3. Chain connect failures: see "dead websocket endpoint" above.

## Deployment verification checklist

After any configuration change:

- [ ] `ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh ./deploy/mainnet/sn81.env` exits 0.
- [ ] `/health` returns `"overall": "ok"` within 180s.
- [ ] First set_weights call lands onchain (visible in metagraph within 1 window).
- [ ] Prometheus `reliquary_verifier_stage_total` counter advancing.

## Scenario: public audit mirror is stale

Symptom: `https://pub-…r2.dev/audit/index.json` reports a `generated_at`
timestamp older than ~30 min, OR `audit/index.json` is missing entirely
for the live netuid. External viewers see no recent windows.

This does NOT necessarily mean the validator is down — it usually means
the audit-index publisher service has stopped, or the public R2 base URL
has drifted from what the audit index is configured to write.

```bash
# 1. Confirm the validator + miner are actually mining live windows.
#    `latest_window_mined` should be within ~12 min of the current chain window.
curl -s http://127.0.0.1:9108/status | jq '{
  latest_window_mined,
  latest_importable_window,
  import_lag_windows,
  netuid,
  network
}'

# 2. Confirm the audit-index timer fired recently. If LAST is > 15 min ago,
#    the rebuild has stalled — restart the timer.
systemctl --user list-timers reliquary-audit-index.timer

# 3. If the timer is healthy but the public bucket is still stale, the
#    publish path is probably mis-configured. Compare:
grep -E 'RELIQUARY_INFERENCE_(AUDIT_BUCKET|PUBLIC_AUDIT_BASE_URL|AUDIT_PREFIX)' \
  /save/state/reliquary-inference.env 2>/dev/null
#    against the URL you're querying. The PUBLIC_AUDIT_BASE_URL must match
#    the public.r2.dev hostname for the bucket the AUDIT_BUCKET writes to.

# 4. Force a one-shot rebuild + publish so the mirror catches up immediately.
python3 -m reliquary_inference.cli build-audit-index --publish

# 5. Verify externally — should print a recent ISO timestamp + the live netuid.
curl -s "${RELIQUARY_INFERENCE_PUBLIC_AUDIT_BASE_URL}/audit/index.json" \
  | jq '{generated_at, netuid, window_count, latest: .windows[0].window_id}'

# 6. Re-arm the timer if it had drifted.
systemctl --user restart reliquary-audit-index.timer
```

If step 1 shows `latest_window_mined` is ALSO stale (> 30 min behind chain),
the validator/miner is actually down — switch to "validator stopped"
playbook below before worrying about the mirror.

## Scenario: validator stopped publishing verdicts

Symptom: `latest_window_mined` from `/status` is multiple windows behind
the live chain head; verdicts haven't shown up in R2 for the recent
windows; mesh consensus hasn't advanced.

```bash
# 1. Are services up?
systemctl --user status reliquary-ledger-validator-mainnet \
                         reliquary-ledger-miner-mainnet 2>&1 \
  | grep -E 'Active|Loaded'

# 2. Recent journal — look for an unhandled exception or a chain-RPC error.
journalctl --user -u reliquary-ledger-validator-mainnet --since '1 hour ago' \
  --no-pager | tail -60

# 3. Is the chain endpoint reachable?
btcli subnet metagraph --netuid "$RELIQUARY_NETUID" --network finney 2>&1 | head -10

# 4. If the service exited cleanly (no stack trace) but isn't running:
systemctl --user restart reliquary-ledger-validator-mainnet \
                          reliquary-ledger-miner-mainnet

# 5. If the service is restart-looping with a chain-connect error:
#    fall through to the "dead websocket endpoint" playbook above.

# 6. Once recovered, confirm `latest_window_mined` advances on the next
#    `/status` poll (~12 min later). Also re-publish the audit index per
#    the stale-mirror playbook above.
```

Important: don't restart blindly. If the service died from a known
correctness invariant (e.g. reparam guard rejected every delta for a
window, or a sketch-tolerance assertion fired), restarting will just
re-trigger the same exit. Read the journal first.

## Escalation

- **Severity 1 (outage, >10 min unhealthy):** page oncall rotation.
- **Severity 2 (degraded, >30 min):** Slack #reliquary-ops, no page.
- **Severity 3 (observability anomaly):** log issue, review next business day.

## Useful commands

```bash
# Status
journalctl --user -u reliquary-ledger-validator-mainnet -f
curl -s http://127.0.0.1:9180/health | jq .

# Onchain cross-checks
btcli subnet metagraph --netuid $RELIQUARY_NETUID --network finney
btcli wallet overview --wallet.name reliquary-ledger --wallet.path /home/OPERATOR/.bittensor/wallets

# Force a window re-verify (dry run — does not mutate chain)
python3 -m reliquary_inference.cli replay-window --window 12345 --dry-run
```

## Change log

Operators: append dated entries after any non-trivial change.

- YYYY-MM-DD — operator name — change summary.

# Runbook: emergency rollback to testnet

If something breaks within the first 72 hours after the SN81 cutover and
we need to retreat to testnet 462 to debug, follow this procedure.
Practice it on a staging box **before** cutover-day so it's muscle memory.

## When to invoke this

Roll back if **any** of:

- A novel proof-verification failure pattern emerges (>1% hard-fail rate
  on validators running mainline code, where testnet bake-in had 0%).
- Mesh consensus disagreement rate spikes > 0.10 across multiple
  validators, suggesting a deterministic-execution divergence we didn't
  catch in cross-GPU audit.
- A signed `PolicyCommitment` apply on miners corrupts the model
  irreversibly and `RELIQUARY_INFERENCE_RESUME_FROM` recovery fails.
- Wallet compromise (see `docs/runbook-mainnet.md` §4 first; that's a
  separate procedure).
- Subtensor / Bittensor mainnet itself is degraded (per OpenTensor
  status page) and we cannot publish weights for > 30 min.

Do **not** invoke for:

- Single-window proof-verification spikes that resolve within one
  weight-submission interval (360 blocks / ~72 min). That's normal noise.
- Single mesh-validator outage. Other validators continue with quorum.
- Slow R2 reads (transient Cloudflare issue).

## Procedure (validator side, 4-node mesh)

Goal: **all 4 mesh validators stop mainnet, switch env, restart on testnet
within 10 minutes**. Done in parallel across the 4 nodes via the
shared on-call channel.

### Step 1 (T+0 m) — declare incident

- Post in operator channel: `INCIDENT: rolling back SN81 -> testnet 462`.
- Page tech lead. No further changes to `main` until rollback complete.

### Step 2 (T+1 m) — stop mainnet processes on all 4 nodes

```bash
# bare-metal
sudo systemctl stop reliquary-ledger-validator-mainnet
sudo systemctl stop reliquary-ledger-miner-mainnet  # if running

# docker
docker compose down
```

Verify with `systemctl status` / `docker ps`. The mainnet containers
must be **fully stopped** before step 3 — a half-stopped validator can
still publish weights and corrupt the rollback state.

### Step 3 (T+3 m) — point env at testnet 462

```bash
# back up the mainnet env file in case forensics needs it
sudo cp /etc/reliquary-ledger/sn81.env /etc/reliquary-ledger/sn81.env.incident-$(date +%Y%m%dT%H%M)

# install the testnet template
sudo cp /opt/reliquary-ledger/env.testnet.example /etc/reliquary-ledger/sn462.env
# (if your operator config keeps wallet/R2 creds out of the env file,
#  copy them from sn81.env.incident-* into sn462.env now)
sudo chmod 600 /etc/reliquary-ledger/sn462.env
```

Critical fields that **must** match testnet 462:
- `RELIQUARY_INFERENCE_NETWORK=test`
- `RELIQUARY_INFERENCE_NETUID=462`
- `RELIQUARY_INFERENCE_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443`

### Step 4 (T+5 m) — start testnet processes

```bash
# bare-metal
sudo ln -sf /etc/reliquary-ledger/sn462.env /etc/reliquary-ledger/active.env
sudo systemctl start reliquary-ledger-validator-testnet

# docker
cp env.testnet.example .env  # then edit with the wallet path
docker compose up -d validator
```

### Step 5 (T+8 m) — verify health

```bash
curl -fsS http://localhost:9180/health | jq .overall
# expect: "ok"

curl -fsS http://localhost:9108/metrics | grep -E '^(reliquary_chain_window_id|reliquary_health_)'
# expect: window_id increasing, health_chain_state=1, etc.
```

Cross-check the public R2 audit index — verdicts from the rolled-back
validator should appear within 1 window (~6 min on testnet) under the
testnet 462 prefix.

### Step 6 (T+10 m) — declare rollback complete

- Post in operator channel: `INCIDENT: rollback complete; on testnet 462`.
- Snapshot the mainnet incident state (env file backup, last 1 hour of
  logs, last 4 verdicts) into `private/reliquary-plan/audit/incidents/`.
- Schedule postmortem within 7 days.

## What's preserved across the rollback

- **Wallet** (coldkey + hotkey files) — unchanged on disk.
- **Validator state** under `/var/lib/reliquary-ledger` — preserved;
  testnet boot reads it back.
- **R2 buckets** — separate buckets for mainnet vs testnet (per the env
  template); no overlap, no clobbering.
- **HF Hub repos** — separate repos per environment.

## What's lost / forfeited

- Recent mainnet windows' completions for which the validator was the
  only mesh node still publishing — they are **orphaned**, not part of
  any consensus, and must be regenerated post-recovery if needed.
- Any partial weight-set that was in flight when systemctl stopped the
  process. Weights default to last-good values on chain; the rollback
  doesn't write null weights.
- Watchtower-pulled image upgrades after T-1 h are reverted to whatever
  was running locally — verify image tag with `docker images`.

## Postmortem (within 7 days)

The postmortem template lives at `docs/postmortem-template.md`. Required
fields:

1. Incident summary (2–3 sentences).
2. Timeline (T-N min before cutover, T+N min during incident).
3. Root cause (5-whys; do not stop at "the proof verifier failed").
4. What worked (the rollback procedure, the audit index, the on-call channel, etc.).
5. What didn't work.
6. Action items with owners + dates.
7. Update to this runbook + the cutover checklist.

If this runbook itself fails (e.g. step 3 didn't preserve creds), that
is the postmortem's most-important action item. Update before the next
cutover attempt.

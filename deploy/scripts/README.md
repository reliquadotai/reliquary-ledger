# Reliquary Ledger operator deploy scripts

## Shared HMAC signing secret — mandatory

Every miner + every validator on a Reliquary subnet MUST use the SAME value
for `RELIQUARY_INFERENCE_SIGNING_SECRET` when `RELIQUARY_INFERENCE_SIGNATURE_SCHEME=local_hmac`.

Miners sign each completion's commit-binding with HMAC-SHA256 keyed by this
secret. Validators verify using the same key. If they differ, every verdict
is hard-rejected with `hard_fail_reason="invalid_signature"` and the
validator publishes `weights={<miner>: 0.0}` regardless of how correct the
completions are. On-chain `set_weights` is then skipped entirely because no
weight > 0 remains.

Generation + distribution pattern:

```bash
# On an authoritative host (e.g. the subnet owner's laptop)
openssl rand -hex 32 > /tmp/reliquary-signing-secret
chmod 600 /tmp/reliquary-signing-secret

# Distribute to every node before starting services:
scp /tmp/reliquary-signing-secret root@devserver:/save/secrets/
scp /tmp/reliquary-signing-secret root@staging1:/opt/reliquary-secrets/
# ...etc

# On each node, the deploy scripts + bootstrap.sh read this file and
# inject it into the systemd EnvironmentFile as
# RELIQUARY_INFERENCE_SIGNING_SECRET=<value>.
```

This is a subnet-level shared secret — rotate by stopping all services,
regenerating, redistributing, restarting. Miners signed with the OLD secret
will be rejected by validators using the NEW secret (and vice versa) during
the rotation window.

A future revision will move to per-hotkey substrate signatures, eliminating
the shared secret entirely. Until then, treat this as a subnet-wide op.

## Scripts

- `deploy-staging-validator.sh <ssh_host> <hotkey_name>` — deploy one mesh
  validator to a remote staging host. Reads wallet + R2 token + signing
  secret from operator-local paths; scp's only the hotkey (no coldkey
  seed) to the target. Installs + enables `reliquary-staging-validator.service`.

## Referenced by

- `/deploy/bootstrap-persistent.sh` — Reliquary Ledger persistent-volume
  bootstrap used on Targon-style ephemeral containers with a 200GB /save
  mount. Re-runnable on every container restart.
- `/deploy/systemd/reliquary-staging-validator.service` — unit file
  installed on staging hosts by the deploy script.

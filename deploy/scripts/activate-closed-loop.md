# Activating the continuous closed-loop flywheel

The bridge code (Forge → R2 → Ledger) ships flag-gated OFF by default.
When off, both miner and validator operate on fixed base model weights
and `set_weights` carries whatever the verifier scored. When on, both
services poll for fresh `PolicyCommitment`s and hot-swap their local
model weights in-place at window boundaries.

Turning this on is operator-controlled because:
- A mismatch between miner and validator (one flag-on, one flag-off)
  causes proofs to hard-fail and weights drop to 0.
- The policy_authority signing key must be distributed before enabling.
- A malformed delta from a misconfigured Forge can block an entire mesh
  until rolled back.

## Prerequisites (one-time)

1. **Forge policy_authority keypair.** The same HMAC key that Forge uses
   to sign `CheckpointAttestation` + `PolicyCommitment`. On the Forge
   host, this lives at `/save/secrets/signing-secret` (or
   `/opt/reliquary-secrets/signing-secret` on stagings). Distribute the
   SAME secret to every Ledger node so the validator can verify.

2. **`reliquary` (Forge) package importable on every Ledger node.** The
   bundle_aware_delta_loader imports `reliquary.training.checkpoint_storage.fetch_bundle`.
   If missing, the hook self-disables with a yellow warning — no harm,
   but the hot-swap never runs.

3. **Both miner + validator on the same build** — the in-place delta
   apply relies on identical tensor naming. Deploy the same commit
   hash to every node before flipping.

4. **Stable baseline.** Confirm `weight_publication_success=1.0` and
   mesh is publishing clean verdicts before the flip. Rolling back is
   clean; activating on an already-broken subnet tangles failure modes.

## Activation — 5 env vars, then restart

On every Ledger node (devserver + every staging validator):

```
RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED=true
RELIQUARY_INFERENCE_POLICY_AUTHORITY_HOTKEY=<policy_authority_hotkey_string>
RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET=<contents of /save/secrets/signing-secret>
RELIQUARY_INFERENCE_TRAINING_NETUID=<same as INFERENCE_NETUID under the single-subnet model>
```

Then:

```
systemctl restart inference-miner inference-validator reliquary-staging-validator
```

The run_miner + run_validator loops now:
1. Fetch latest commitments from R2 (`commitments/<netuid>/policy/*.json`)
2. Verify signature chain (HMAC against policy_authority)
3. Apply delta in-place to the live model (miner: engine.model;
   validator: every bundle in shared.modeling._BUNDLE_CACHE)
4. Continue mining / verifying with the updated weights

## Verification

A successful flip shows:

```
journalctl -u inference-miner --since "2 min ago" | grep policy_consumer
  policy_consumer applied run_id=forge-... at ledger_window=... merkle=...

journalctl -u inference-validator --since "2 min ago" | grep policy_consumer
  validator policy_consumer applied run_id=... at ledger_window=...

journalctl -u inference-validator --since "2 min ago" | grep "published weights"
  published weights for window N: {'5Ceudda79...': 1.0}
```

Weights stay at 1.0 if miner and validator applied the same delta at
the same window boundary. If only the miner applied, you'll see
`hard_fail_reason=proof_failed` (sketch mismatch) in the verdicts.

## Kill switch

To roll back:

```
RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED=false
systemctl restart inference-miner inference-validator reliquary-staging-validator
```

After restart the services are on fresh base Qwen2.5-3B weights (cache
was mutated in memory, restarts reload from HF cache). All applied
deltas are erased. Next windows back to baseline.

## Known limitations

- Deltas accumulate in the in-memory cache but are lost on restart. For
  production, the miner/validator should call `register_delta` into a
  persistent store at boot OR the applier should extend to re-fetch
  and re-apply the latest N commitments during bundle load.
- `reliquary.training.checkpoint_storage.fetch_bundle` requires the
  `reliquary` package. On Ledger-only deployments, install with
  `pip install -e /path/to/reliquary-forge` alongside `reliquary-ledger`.
- No automatic revert on bad-delta: if Forge publishes a destructive
  delta, every node applies it. Monitor `weight_publication_success`
  after each hot-swap and roll back the flag if the signal drops.

Future work: signed-revocation commitments (a Forge-emitted "undo"
that rewinds to a prior run_id) + operator-side sanity checks before
applying.

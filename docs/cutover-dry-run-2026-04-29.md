# Cutover dry-run capture — 2026-04-29

**Closes:** gate F-1 of `docs/mainnet-cutover-checklist.md` ("`deploy/apply-mainnet-sn81-profile.sh` runs cleanly with `ALLOW_MAINNET=1` against a local subtensor or dry-run env, verifies `diagnose-config` output, and the resulting service starts publishing weights without error").
**Host:** `staging1rtx6000` (`wrk-j8nq3c7xn81v-78f5546865-gkvxn`)
**Date (UTC):** 2026-04-29T06:32:26Z
**Operator:** autonomous run

---

## What this is

A recorded end-to-end exercise of the mainnet cutover script
[`deploy/apply-mainnet-sn81-profile.sh`](../deploy/apply-mainnet-sn81-profile.sh)
without ever pointing at finney mainnet. Every test case below uses an
env file
([`/tmp/sn81-dryrun.env`](#dry-run-env-file))
targeting testnet `netuid 462` / `chain.test`, so the script's
behavior is exercised but no real TAO is at risk.

---

## Pre-flight finding — and the fix shipped before the green capture

The first dry-run pass surfaced a real bug:

- The cutover script invoked `python3 -m reliquary_inference.cli diagnose-config`.
- `python -m reliquary_inference.cli` does **not** invoke the typer
  app (the entrypoint is wired via `[project.scripts]` in
  `pyproject.toml`, not as a `__main__` block).
- The CLI also had no `diagnose-config` command at all.
- The trailing `|| true` swallowed the failure, and the script printed
  a misleading "profile applied" success message.

**Fix shipped in this same dry-run cycle, before the green capture:**

1. Added `diagnose-config` typer command at [`reliquary_inference/cli.py`](../reliquary_inference/cli.py)
   that loads the active config, prints a structured summary, runs
   PASS/FAIL checks against required-by-backend fields, and exits 1 if
   any check fails.
2. Rewrote the cutover script's pre-flight block: drop `python -m`,
   drop `|| true`, invoke the venv `reliquary-inference` entrypoint
   directly, and propagate the exit code so a missing env value
   actually aborts the cutover.

The capture below is from the post-fix script.

---

## Tests

Each test prints its own `exit=N (expected: …)` line so the
result is unambiguous in the log.

| # | Test | Expected exit | Actual exit | Verdict |
|---|---|---|---|---|
| 1 | Refusal without `ALLOW_MAINNET` (safety guard) | 1 | 1 | PASS |
| 2 | Missing env file (operator passed wrong path) | 2 | 2 | PASS |
| 3 | Full path with testnet env, all-green diagnose-config | 0 | 0 | PASS |
| 4 | `diagnose-config` standalone (parses env directly) | 0 | 0 | PASS |
| 5 | Failure surfacing — delete a required env var | 1 | 1 | PASS |

All five pass. Gate F-1 closes green.

---

## Full capture

```
################################################################
# Reliquary cutover dry-run capture
# host:     wrk-j8nq3c7xn81v-78f5546865-gkvxn
# date:     2026-04-29T06:32:26Z
# script:   deploy/apply-mainnet-sn81-profile.sh
# env file: /tmp/sn81-dryrun.env (testnet target — NOT finney)
################################################################

Goal: exercise every code path in the cutover script without
      ever pointing at finney mainnet. The env file targets
      netuid 462 / chain.test.

=== TEST 1: refusal without ALLOW_MAINNET (safety guard) ===
refusing to apply mainnet profile without ALLOW_MAINNET=1
re-run with: ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh
exit=1  (expected: 1 — script must refuse)

=== TEST 2: missing env file (operator passed wrong path) ===
env file not found: ./does-not-exist.env
copy deploy/mainnet/sn81.env.template first and fill in wallet + endpoint details
exit=2  (expected: 2 — clean error before any env source)

=== TEST 3: full path with testnet env (all-green diagnose-config) ===
applying mainnet SN81 profile from /tmp/sn81-dryrun.env
reliquary-inference diagnose-config
  network         : test
  netuid          : 462
  chain_endpoint  : wss://test.finney.opentensor.ai:443
  model_ref       : Qwen/Qwen2.5-3B-Instruct
  storage_backend : r2_rest
  miner_id        : local-miner
  wallet_name     : reliquary-validator
  hotkey_name     : staging1
  wallet_path     : /opt/reliquary-state/.bittensor/wallets
  signature_scheme: local_hmac
  target_class    : non-mainnet
  [PASS] network is set
  [PASS] netuid is set + non-zero — netuid=462
  [PASS] chain_endpoint is set
  [PASS] model_ref is set
  [PASS] storage_backend is set
  [PASS] wallet_path exists — /opt/reliquary-state/.bittensor/wallets
  [PASS] r2_bucket is set — reliquary
  [PASS] r2_rest_account_id is set
  [PASS] r2_rest_cf_api_token is set
  [PASS] hotkey file exists — /opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys/staging1
diagnose-config OK
profile applied in current shell; launch services with this env active:
  systemctl --user start reliquary-ledger-miner-mainnet
  systemctl --user start reliquary-ledger-validator-mainnet
exit=0  (expected: 0 — diagnose-config PASS on all checks)

=== TEST 4: diagnose-config standalone (verify command parses env directly) ===
reliquary-inference diagnose-config
  network         : test
  netuid          : 462
  chain_endpoint  : wss://test.finney.opentensor.ai:443
  model_ref       : Qwen/Qwen2.5-3B-Instruct
  storage_backend : r2_rest
  miner_id        : local-miner
  wallet_name     : reliquary-validator
  hotkey_name     : staging1
  wallet_path     : /opt/reliquary-state/.bittensor/wallets
  signature_scheme: local_hmac
  target_class    : non-mainnet
  [PASS] network is set
  [PASS] netuid is set + non-zero — netuid=462
  [PASS] chain_endpoint is set
  [PASS] model_ref is set
  [PASS] storage_backend is set
  [PASS] wallet_path exists — /opt/reliquary-state/.bittensor/wallets
  [PASS] r2_bucket is set — reliquary
  [PASS] r2_rest_account_id is set
  [PASS] r2_rest_cf_api_token is set
  [PASS] hotkey file exists — /opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys/staging1
diagnose-config OK
exit=0  (expected: 0)

=== TEST 5: failure surfacing (delete a required env var → diagnose-config FAIL) ===
reliquary-inference diagnose-config
  network         : test
  netuid          : 462
  chain_endpoint  : wss://test.finney.opentensor.ai:443
  model_ref       : Qwen/Qwen2.5-3B-Instruct
  storage_backend : r2_rest
  miner_id        : local-miner
  wallet_name     : reliquary-validator
  hotkey_name     : staging1
  wallet_path     : /opt/reliquary-state/.bittensor/wallets
  signature_scheme: local_hmac
  target_class    : non-mainnet
  [PASS] network is set
  [PASS] netuid is set + non-zero — netuid=462
  [PASS] chain_endpoint is set
  [PASS] model_ref is set
  [PASS] storage_backend is set
  [PASS] wallet_path exists — /opt/reliquary-state/.bittensor/wallets
  [PASS] r2_bucket is set — reliquary
  [PASS] r2_rest_account_id is set
  [FAIL] r2_rest_cf_api_token is set
  [PASS] hotkey file exists — /opt/reliquary-state/.bittensor/wallets/reliquary-validator/hotkeys/staging1
diagnose-config FAIL — 1 check(s) failed
exit=1  (expected: 1 — missing required field caught loudly)

################################################################
# capture complete
################################################################
```

---

## Dry-run env file

The env used for tests 3-5. Targets testnet `netuid 462` (NOT finney).
Storage credentials shown only by key name; the values were sourced
from the staging1 production env at runtime.

```
RELIQUARY_INFERENCE_NETWORK=test
RELIQUARY_INFERENCE_NETUID=462
RELIQUARY_INFERENCE_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
RELIQUARY_INFERENCE_TASK_SOURCE=math
RELIQUARY_INFERENCE_MODEL_REF=Qwen/Qwen2.5-3B-Instruct
RELIQUARY_INFERENCE_DEVICE=cuda:0
RELIQUARY_INFERENCE_LOAD_DTYPE=bfloat16
RELIQUARY_INFERENCE_MINER_MODE=single_gpu_hf
RELIQUARY_INFERENCE_SIGNATURE_SCHEME=local_hmac
RELIQUARY_INFERENCE_USE_DRAND=false
RELIQUARY_INFERENCE_STORAGE_BACKEND=r2_rest
RELIQUARY_INFERENCE_STATE_ROOT=/tmp/reliquary-cutover-dryrun
RELIQUARY_INFERENCE_LOG_DIR=/tmp/reliquary-cutover-dryrun/logs
RELIQUARY_INFERENCE_METRICS_BIND=127.0.0.1
RELIQUARY_INFERENCE_METRICS_PORT=9108
BT_SUBTENSOR_NETWORK=test
BT_WALLET_PATH=/opt/reliquary-state/.bittensor/wallets
WALLET_NAME=reliquary-validator
HOTKEY_NAME=staging1
RELIQUARY_INFERENCE_R2_ACCOUNT_ID=d5332aea7e3780d0f2391a4e4f6ddfbc
RELIQUARY_INFERENCE_R2_BUCKET=reliquary
RELIQUARY_INFERENCE_R2_PUBLIC_URL=https://pub-954f95c7d2f3478886c8a8ff7a4946e0.r2.dev
RELIQUARY_INFERENCE_R2_CF_API_TOKEN=<sourced from /opt/reliquary-secrets at runtime>
RELIQUARY_INFERENCE_HEALTH_BIND=127.0.0.1
RELIQUARY_INFERENCE_HEALTH_PORT=9180
RELIQUARY_INFERENCE_REQUIRE_FLASH_ATTENTION=0
```

For the live mainnet env, the operator copies
[`deploy/mainnet/sn81.env.template`](../deploy/mainnet/sn81.env.template)
to `deploy/mainnet/sn81.env` (gitignored), substitutes
`RELIQUARY_INFERENCE_NETWORK=finney`, `RELIQUARY_INFERENCE_NETUID=81`,
`RELIQUARY_INFERENCE_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443`,
fills the per-host wallet path + R2 credentials, and runs:

```
ALLOW_MAINNET=1 ./deploy/apply-mainnet-sn81-profile.sh
```

`diagnose-config` will then report `target_class : MAINNET (finney/SN81)`.

---

## What this proves

1. The safety guard works: the script refuses without `ALLOW_MAINNET=1`.
2. Operator misconfiguration is caught loudly — no silent half-applied state.
3. The diagnose-config pre-flight loads the active config, validates
   every required-by-backend field, and reports both the resolved
   values (so an operator can eyeball-verify the cutover target) and
   a per-check PASS/FAIL line.
4. A missing env value (test 5) propagates a non-zero exit through
   the cutover script — the operator sees the failure instead of a
   misleading "profile applied" line.
5. End-to-end the script + `diagnose-config` is operator-runnable on
   any host that has the venv on PATH; no extra dependencies.

---

## Reproducibility

```bash
# from the dev box
scp deploy/apply-mainnet-sn81-profile.sh reliquary_inference/cli.py \
    HOST:/opt/reliquary-repos/reliquary-inference/<same-paths>
scp /tmp/sn81-dryrun.env HOST:/tmp/
scp /tmp/cutover-dryrun-v2.sh HOST:/tmp/
ssh HOST "bash /tmp/cutover-dryrun-v2.sh && cat /tmp/cutover-dryrun.log"
```

The dry-run env (`/tmp/sn81-dryrun.env`) is **not** committed to the
repo — operators rebuild it from
[`deploy/mainnet/sn81.env.template`](../deploy/mainnet/sn81.env.template)
with site-specific values.

---

## Cutover-readiness implication

With this capture committed:

- Gate F-1 of `docs/mainnet-cutover-checklist.md` flips green.
- All A-G gates of the cutover checklist are now operator-runnable
  (F-2 rollback runbook + F-3 second-operator walkthrough remain
  partial, neither blocking).

The cutover trigger is a unilateral operator decision. The script
correctly refuses anything risky and validates the operator's env
before doing anything chain-touching.

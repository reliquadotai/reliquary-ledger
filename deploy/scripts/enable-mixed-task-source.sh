#!/usr/bin/env bash
# Switch the Ledger task source from `math` to `mixed` (MATH + GSM8K).
#
# Run on EVERY validator + miner host in the mesh, in close succession.
# A coherent flip is required: validators sample tasks from the source
# named in the env var, and they MUST agree per window. A staggered
# rollout produces a window where some validators draw MATH tasks and
# others draw mixed batches → prompt_hash mismatch → mass-rejection.
#
# Recommended cadence: snapshot the current chain window, run this on
# every host within 30 s, restart all services within 60 s. Both
# operations should land before the next window boundary (6 min).
#
# Pre-flight: confirm reliquary-ledger >= c827d5b (the commit that
# landed GSM8K + MixedTasksSource).
#
# Rollback: re-run with --revert. Same coordination requirement applies.

set -euo pipefail

ENV_FILE="${RELIQUARY_LEDGER_ENV:-/save/state/reliquary-inference.env}"
MIX_DEFAULT="${RELIQUARY_INFERENCE_TASK_MIX:-math:2,gsm8k:1}"

cmd="${1:-enable}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "ERROR: required path missing: $1" >&2
    exit 1
  fi
}

require_path "$ENV_FILE"

# Confirm the package version supports mixed sources. The
# MixedTasksSource landed alongside the GSM8K env. If this import fails,
# you're on an older reliquary-ledger and should pull first.
VENV="${RELIQUARY_LEDGER_VENV:-/opt/reliquary-venv}"
if ! "$VENV/bin/python" -c \
    "from reliquary_inference.dataset.task_sources import MixedTasksSource" \
    2>/dev/null; then
  echo "ERROR: MixedTasksSource not available — pull a newer reliquary-ledger." >&2
  echo "  fix: cd to the reliquary-ledger checkout and: git pull && $VENV/bin/pip install -e ." >&2
  exit 1
fi

case "$cmd" in
  enable)
    echo "=== current task-source config ==="
    grep -E '^RELIQUARY_INFERENCE_TASK_(SOURCE|MIX)' "$ENV_FILE" || true

    # Replace or append RELIQUARY_INFERENCE_TASK_SOURCE.
    if grep -q '^RELIQUARY_INFERENCE_TASK_SOURCE=' "$ENV_FILE"; then
      sed -i.bak 's|^RELIQUARY_INFERENCE_TASK_SOURCE=.*|RELIQUARY_INFERENCE_TASK_SOURCE=mixed|' "$ENV_FILE"
    else
      printf 'RELIQUARY_INFERENCE_TASK_SOURCE=mixed\n' >> "$ENV_FILE"
    fi
    if grep -q '^RELIQUARY_INFERENCE_TASK_MIX=' "$ENV_FILE"; then
      sed -i.bak "s|^RELIQUARY_INFERENCE_TASK_MIX=.*|RELIQUARY_INFERENCE_TASK_MIX=$MIX_DEFAULT|" "$ENV_FILE"
    else
      printf 'RELIQUARY_INFERENCE_TASK_MIX=%s\n' "$MIX_DEFAULT" >> "$ENV_FILE"
    fi

    echo
    echo "=== updated config ==="
    grep -E '^RELIQUARY_INFERENCE_TASK_(SOURCE|MIX)' "$ENV_FILE"

    echo
    echo "=== restarting services ==="
    # Restart whichever role this host runs. Both restart calls are
    # idempotent — non-existent units no-op with a warning we ignore.
    systemctl --user restart reliquary-ledger-validator-mainnet 2>&1 \
      | head -3 || true
    systemctl --user restart reliquary-ledger-miner-mainnet 2>&1 \
      | head -3 || true

    echo
    echo "Done. Confirm coherent flip across the mesh:"
    echo "  - Run this script on EVERY validator + miner within 30s"
    echo "  - Wait one full window (~6 min)"
    echo "  - Inspect /status — task_source field should now read 'mixed'"
    ;;

  --revert)
    echo "=== reverting to math-only ==="
    if grep -q '^RELIQUARY_INFERENCE_TASK_SOURCE=' "$ENV_FILE"; then
      sed -i.bak 's|^RELIQUARY_INFERENCE_TASK_SOURCE=.*|RELIQUARY_INFERENCE_TASK_SOURCE=math|' "$ENV_FILE"
    else
      printf 'RELIQUARY_INFERENCE_TASK_SOURCE=math\n' >> "$ENV_FILE"
    fi
    grep -E '^RELIQUARY_INFERENCE_TASK_(SOURCE|MIX)' "$ENV_FILE"

    systemctl --user restart reliquary-ledger-validator-mainnet 2>&1 \
      | head -3 || true
    systemctl --user restart reliquary-ledger-miner-mainnet 2>&1 \
      | head -3 || true
    echo "=== reverted ==="
    ;;

  *)
    echo "Usage: $0 [enable|--revert]" >&2
    echo
    echo "Optional env vars:"
    echo "  RELIQUARY_LEDGER_ENV   path to env file (default /save/state/reliquary-inference.env)"
    echo "  RELIQUARY_LEDGER_VENV  path to venv (default /opt/reliquary-venv)"
    echo "  RELIQUARY_INFERENCE_TASK_MIX  weights, e.g. 'math:2,gsm8k:1'"
    exit 2
    ;;
esac

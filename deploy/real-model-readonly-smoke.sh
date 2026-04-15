#!/usr/bin/env bash
set -euo pipefail

MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
STATE_ROOT="${STATE_ROOT:-/srv/reliquary-inference/state}"
MODEL_REF="${MODEL_REF:-Qwen/Qwen2.5-1.5B-Instruct}"
TASK_SOURCE="${TASK_SOURCE:-reasoning_tasks}"
TASK_COUNT="${TASK_COUNT:-4}"
MINER_ID="${MINER_ID:-real-model-smoke}"
DEVICE="${DEVICE:-cuda}"
LOAD_DTYPE="${LOAD_DTYPE:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

export RELIQUARY_INFERENCE_NETWORK="${RELIQUARY_INFERENCE_NETWORK:-test}"
export BT_SUBTENSOR_NETWORK="${BT_SUBTENSOR_NETWORK:-test}"
export RELIQUARY_INFERENCE_STORAGE_BACKEND=local
export RELIQUARY_INFERENCE_MODEL_REF="${MODEL_REF}"
export RELIQUARY_INFERENCE_TASK_SOURCE="${TASK_SOURCE}"
export RELIQUARY_INFERENCE_TASK_COUNT="${TASK_COUNT}"
export RELIQUARY_INFERENCE_SAMPLES_PER_TASK="${RELIQUARY_INFERENCE_SAMPLES_PER_TASK:-1}"
export RELIQUARY_INFERENCE_DEVICE="${DEVICE}"
export RELIQUARY_INFERENCE_LOAD_DTYPE="${LOAD_DTYPE}"
export RELIQUARY_INFERENCE_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
export RELIQUARY_INFERENCE_ARTIFACT_DIR="${STATE_ROOT}/real-model-smoke/artifacts"
export RELIQUARY_INFERENCE_EXPORT_DIR="${STATE_ROOT}/real-model-smoke/exports"
export RELIQUARY_INFERENCE_LOG_DIR="${STATE_ROOT}/real-model-smoke/logs"

mkdir -p \
  "${RELIQUARY_INFERENCE_ARTIFACT_DIR}" \
  "${RELIQUARY_INFERENCE_EXPORT_DIR}" \
  "${RELIQUARY_INFERENCE_LOG_DIR}"

WINDOW_ID="$("${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - <<'PY'
from reliquary_inference.chain.adapter import BittensorChainAdapter, LocalChainAdapter
from reliquary_inference.config import load_config

cfg = load_config()
if str(cfg["network"]) in {"local", "mock"}:
    chain = LocalChainAdapter()
else:
    chain = BittensorChainAdapter(
        network=str(cfg["network"]),
        chain_endpoint=str(cfg.get("chain_endpoint", "")),
        netuid=int(cfg["netuid"]),
        wallet_name=str(cfg["wallet_name"]),
        hotkey_name=str(cfg["hotkey_name"]),
        wallet_path=str(cfg["wallet_path"]),
        use_drand=bool(cfg["use_drand"]),
    )
print(chain.get_window_context(cfg=cfg).window_id)
PY
)"

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" \
  reliquary-inference publish-tasks --source "${TASK_SOURCE}" --count "${TASK_COUNT}" --window "${WINDOW_ID}"
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" \
  reliquary-inference mine-window --source "${TASK_SOURCE}" --miner-id "${MINER_ID}" --window "${WINDOW_ID}"
"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" \
  reliquary-inference validate-window --source "${TASK_SOURCE}" --window "${WINDOW_ID}"

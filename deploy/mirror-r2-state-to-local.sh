#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/srv/reliquary-inference/current}"
ENV_FILE="${ENV_FILE:-/srv/reliquary-inference/state/reliquary-inference.env}"
MINIFORGE_ROOT="${MINIFORGE_ROOT:-/opt/miniforge}"
APP_ENV="${APP_ENV:-reliquary-inference}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi

cd "${REPO_ROOT}"

"${MINIFORGE_ROOT}/bin/conda" run --no-capture-output -n "${APP_ENV}" python - <<'PY'
from __future__ import annotations

from pathlib import Path

from reliquary_inference.config import load_config
from reliquary_inference.storage.registry import R2ObjectStore

cfg = load_config()
if str(cfg["storage_backend"]) != "r2":
    raise SystemExit("storage backend is not r2; nothing to mirror")

store = R2ObjectStore(
    bucket=str(cfg["r2_bucket"]),
    endpoint_url=str(cfg["r2_endpoint_url"]),
    access_key_id=str(cfg["r2_access_key_id"]),
    secret_access_key=str(cfg["r2_secret_access_key"]),
)

artifact_root = Path(str(cfg["artifact_dir"]))
export_root = Path(str(cfg["export_dir"]))
artifact_root.mkdir(parents=True, exist_ok=True)
export_root.mkdir(parents=True, exist_ok=True)

prefixes = [
    "task_batches",
    "completions",
    "verdicts",
    "scorecards",
    "window_manifests",
    "run_manifests",
    "completion_bundles",
    "verdict_bundles",
    "audit",
]

count = 0
for prefix in prefixes:
    for ref in store.list_prefix(prefix):
        key = str(ref["key"])
        target_root = export_root if key.startswith("audit/") else artifact_root
        target = target_root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(store.get_bytes(key))
        count += 1

print(f"mirrored {count} objects from r2 into local state")
PY

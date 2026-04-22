from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..utils.json_io import sha256_json

ARTIFACT_DIRECTORIES = {
    "task_batch": "task_batches",
    "completion": "completions",
    "verdict": "verdicts",
    "scorecard": "scorecards",
    "window_manifest": "window_manifests",
    "run_manifest": "run_manifests",
}


def artifact_directory_name(artifact_type: str) -> str:
    return ARTIFACT_DIRECTORIES.get(artifact_type, f"{artifact_type}s")


def make_artifact(
    *,
    artifact_type: str,
    producer_id: str,
    producer_role: str,
    window_id: int,
    payload: dict[str, Any],
    parent_ids: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    artifact = {
        "artifact_type": artifact_type,
        "producer_id": producer_id,
        "producer_role": producer_role,
        "window_id": window_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "parent_ids": list(parent_ids or []),
        "payload": payload,
    }
    artifact["artifact_id"] = sha256_json(artifact)
    return artifact

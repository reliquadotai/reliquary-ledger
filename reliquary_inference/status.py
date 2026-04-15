from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils.json_io import read_json


def bucket_mode(cfg: dict[str, Any]) -> str:
    has_public_audit = bool(str(cfg.get("public_audit_base_url", "")).strip())
    has_dedicated_audit_target = bool(str(cfg.get("audit_bucket", "")).strip())
    exposes_public_artifacts = bool(cfg.get("expose_public_artifact_urls", False))
    if has_dedicated_audit_target and has_public_audit:
        return "private_artifacts_public_audit"
    if has_public_audit and exposes_public_artifacts:
        return "public_artifacts_public_audit"
    if has_public_audit:
        return "private_artifacts_public_audit_links"
    if str(cfg["storage_backend"]) == "r2":
        return "private_r2"
    return "local"


def chain_endpoint_mode(cfg: dict[str, Any]) -> str:
    endpoint = str(cfg.get("chain_endpoint", "")).strip()
    if str(cfg["network"]) in {"local", "mock"}:
        return "local"
    if endpoint:
        return "dedicated"
    return "network_default"


def audit_index_path(cfg: dict[str, Any], registry) -> Path:
    export_root = cfg.get("export_dir", getattr(registry, "export_root", "./exports"))
    return Path(str(export_root)) / "audit" / "index.json"


def read_audit_index(cfg: dict[str, Any], registry) -> dict[str, Any] | None:
    path = audit_index_path(cfg, registry)
    if not path.exists():
        return None
    return read_json(path)


def status_summary(cfg: dict[str, Any], registry) -> dict[str, Any]:
    completions = registry.list_artifacts("completion")
    latest_completion = max(
        completions,
        key=lambda item: (int(item["window_id"]), str(item["created_at"])),
        default=None,
    )
    latest_manifest = None
    latest_publish = None
    latest_importable_window = None
    audit_payload = read_audit_index(cfg, registry)
    if audit_payload and audit_payload.get("windows"):
        latest_window = audit_payload["windows"][0]
        latest_manifest = {
            "window_id": latest_window["window_id"],
            "payload": {
                "chain_publish_result": latest_window.get("chain_publish_result"),
            },
        }
        latest_publish = latest_window.get("chain_publish_result")
        latest_importable_window = int(latest_window["window_id"])
    if latest_manifest is None:
        finalized_manifests = [
            manifest
            for manifest in registry.list_artifacts("window_manifest")
            if manifest["payload"].get("chain_publish_result") is not None
        ]
        latest_completion = max(
            completions,
            key=lambda item: (int(item["window_id"]), str(item["created_at"])),
            default=None,
        )
        latest_manifest = max(
            finalized_manifests,
            key=lambda item: (int(item["window_id"]), str(item["created_at"])),
            default=None,
        )
        latest_publish = latest_manifest["payload"]["chain_publish_result"] if latest_manifest else None
        latest_importable_window = int(latest_manifest["window_id"]) if latest_manifest else None
    latest_window_mined = int(latest_completion["window_id"]) if latest_completion else None
    import_lag_windows = None
    if latest_window_mined is not None and latest_importable_window is not None:
        import_lag_windows = max(latest_window_mined - latest_importable_window, 0)
    return {
        "network": cfg["network"],
        "netuid": int(cfg["netuid"]),
        "model_ref": cfg["model_ref"],
        "task_source": cfg["task_source"],
        "storage_backend": cfg["storage_backend"],
        "bucket_mode": bucket_mode(cfg),
        "chain_endpoint_mode": chain_endpoint_mode(cfg),
        "chain_endpoint": cfg.get("chain_endpoint", ""),
        "artifact_bucket": cfg.get("r2_bucket", "") if str(cfg["storage_backend"]) == "r2" else "",
        "audit_bucket": cfg.get("audit_bucket", ""),
        "public_audit_base_url": cfg.get("public_audit_base_url", ""),
        "latest_window_mined": latest_window_mined,
        "latest_importable_window": latest_importable_window,
        "import_lag_windows": import_lag_windows,
        "latest_weight_publication": {
            "window_id": int(latest_manifest["window_id"]) if latest_manifest else None,
            "success": latest_publish.get("success") if isinstance(latest_publish, dict) else None,
            "uids": latest_publish.get("uids") if isinstance(latest_publish, dict) else None,
        },
    }

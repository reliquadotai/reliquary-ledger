from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol.artifacts import artifact_directory_name
from .storage.registry import FilesystemObjectStore, R2ObjectStore
from .utils.json_io import write_json


def build_audit_index(
    *,
    cfg: dict[str, Any],
    registry,
    limit: int = 25,
    publish: bool = False,
) -> dict[str, Any]:
    window_manifests = _latest_window_manifests(registry=registry, limit=limit)
    # Fetch ONLY the scorecards referenced by the selected manifests —
    # not the full list_artifacts scan which is O(N) over all scorecards
    # (700+ under heavy R2 throttle = ~10 min). With limit=25 we do at
    # most 25 gets instead. Fall back to list_artifacts for backends
    # that don't support get_artifact (tests, local fs).
    scorecards: dict[str, dict[str, Any]] = {}
    for manifest in window_manifests:
        sc_id = manifest["payload"].get("scorecard_id")
        if not sc_id:
            continue
        try:
            scorecards[sc_id] = registry.get_artifact("scorecard", sc_id)
        except Exception:
            # Skip missing / unreachable scorecards rather than failing
            # the whole index — partial audit is better than none.
            continue
    entries = [
        _window_audit_entry(
            cfg=cfg,
            manifest=manifest,
            scorecard=scorecards.get(manifest["payload"]["scorecard_id"]),
        )
        for manifest in window_manifests
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "network": cfg["network"],
        "netuid": cfg["netuid"],
        "model_ref": cfg["model_ref"],
        "task_source": cfg["task_source"],
        "window_count": len(entries),
        "windows": entries,
    }
    export_dir = registry.run_dir("audit")
    json_path = export_dir / "index.json"
    html_path = export_dir / "index.html"
    write_json(json_path, payload)
    html_path.write_text(_render_html(payload), encoding="utf-8")

    published: dict[str, Any] = {}
    if publish:
        audit_prefix = str(cfg["audit_prefix"]).strip("/") or "audit"
        json_key = f"{audit_prefix}/index.json"
        html_key = f"{audit_prefix}/index.html"
        audit_store = _audit_store(cfg=cfg, registry=registry)
        audit_store.put_bytes(json_key, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        audit_store.put_bytes(html_key, _render_html(payload).encode("utf-8"))
        public_base_url = str(cfg.get("public_audit_base_url", "")).rstrip("/")
        if public_base_url:
            published = {
                "json_url": f"{public_base_url}/{json_key}",
                "html_url": f"{public_base_url}/{html_key}",
            }
        else:
            published = {"json_key": json_key, "html_key": html_key}

    return {
        "generated_at": generated_at,
        "window_count": len(entries),
        "json_path": str(json_path),
        "html_path": str(html_path),
        "published": published,
        "payload": payload,
    }


def _latest_window_manifests(*, registry, limit: int) -> list[dict[str, Any]]:
    manifests = registry.list_artifacts("window_manifest")
    by_window: dict[int, dict[str, Any]] = {}
    for manifest in manifests:
        window_id = int(manifest["window_id"])
        current = by_window.get(window_id)
        if current is None:
            by_window[window_id] = manifest
            continue
        if _manifest_rank(manifest) > _manifest_rank(current):
            by_window[window_id] = manifest
    ordered = sorted(by_window.values(), key=lambda item: int(item["window_id"]), reverse=True)
    return ordered[:limit]


def _manifest_rank(manifest: dict[str, Any]) -> tuple[int, str]:
    publish_result = manifest["payload"].get("chain_publish_result")
    finalized = 1 if publish_result is not None else 0
    return finalized, str(manifest["created_at"])


def _window_audit_entry(
    *,
    cfg: dict[str, Any],
    manifest: dict[str, Any],
    scorecard: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = manifest["payload"]
    score_payload = scorecard["payload"] if scorecard else {}
    verification_totals = score_payload.get("verification_totals", {})
    miner_totals = score_payload.get("miner_totals", {})
    public_base_url = str(cfg.get("public_audit_base_url", "")).rstrip("/")
    expose_artifact_urls = bool(cfg.get("expose_public_artifact_urls", False))
    entry = {
        "window_id": int(manifest["window_id"]),
        "created_at": manifest["created_at"],
        "task_source": payload["task_source"],
        "artifact_ids": {
            "window_manifest": manifest["artifact_id"],
            "task_batch": payload["task_batch_id"],
            "scorecard": payload["scorecard_id"],
        },
        "verification_totals": verification_totals,
        "window_metrics": score_payload.get("window_metrics", {}),
        "weights": score_payload.get("weights", {}),
        "miner_totals": miner_totals,
        "chain_publish_result": payload.get("chain_publish_result"),
        "refs": {
            "task_batch": _artifact_public_ref(
                public_base_url=public_base_url,
                artifact_type="task_batch",
                artifact_id=payload["task_batch_id"],
                expose_url=expose_artifact_urls,
            ),
            "scorecard": _artifact_public_ref(
                public_base_url=public_base_url,
                artifact_type="scorecard",
                artifact_id=payload["scorecard_id"],
                expose_url=expose_artifact_urls,
            ),
            "window_manifest": _artifact_public_ref(
                public_base_url=public_base_url,
                artifact_type="window_manifest",
                artifact_id=manifest["artifact_id"],
                expose_url=expose_artifact_urls,
            ),
            "completion_bundles": [
                _bundle_public_ref(public_base_url=public_base_url, ref=ref, expose_url=expose_artifact_urls)
                for ref in payload.get("completion_bundle_refs", [])
            ],
            "verdict_bundle": _bundle_public_ref(
                public_base_url=public_base_url,
                ref=payload.get("verdict_bundle_ref"),
                expose_url=expose_artifact_urls,
            ),
        },
    }
    return entry


def _artifact_public_ref(*, public_base_url: str, artifact_type: str, artifact_id: str, expose_url: bool) -> dict[str, Any]:
    key = f"{artifact_directory_name(artifact_type)}/{artifact_id}.json"
    ref: dict[str, Any] = {"key": key}
    if public_base_url and expose_url:
        ref["url"] = f"{public_base_url}/{key}"
    return ref


def _bundle_public_ref(*, public_base_url: str, ref: dict[str, Any] | None, expose_url: bool) -> dict[str, Any] | None:
    if ref is None:
        return None
    bundle_ref = {key: value for key, value in ref.items() if key in {"key", "path", "miner_id", "validator_id", "uploaded_at", "backend"}}
    key = ref.get("key")
    if key and public_base_url and expose_url:
        bundle_ref["url"] = f"{public_base_url}/{key}"
    return bundle_ref


def _audit_store(*, cfg: dict[str, Any], registry):
    audit_bucket = str(cfg.get("audit_bucket", "")).strip()
    audit_endpoint_url = str(cfg.get("audit_endpoint_url", "")).strip()
    audit_access_key_id = str(cfg.get("audit_access_key_id", "")).strip()
    audit_secret_access_key = str(cfg.get("audit_secret_access_key", "")).strip()
    if audit_bucket and audit_endpoint_url and audit_access_key_id and audit_secret_access_key:
        return R2ObjectStore(
            bucket=audit_bucket,
            endpoint_url=audit_endpoint_url,
            access_key_id=audit_access_key_id,
            secret_access_key=audit_secret_access_key,
        )
    if hasattr(registry, "put_blob"):
        return _RegistryBlobStore(registry)
    export_root = str(getattr(registry, "export_root", "./exports"))
    return FilesystemObjectStore(export_root)


class _RegistryBlobStore:
    def __init__(self, registry) -> None:
        self.registry = registry

    def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
        ref = self.registry.put_blob(key=key, data=data)
        return ref if isinstance(ref, dict) else {"path": str(ref)}


def _render_html(payload: dict[str, Any]) -> str:
    rows = []
    for entry in payload["windows"]:
        weights = ", ".join(
            f"{html.escape(str(miner))}: {weight:.4f}" for miner, weight in entry["weights"].items()
        ) or "none"
        totals = entry["verification_totals"]
        window_metrics = entry.get("window_metrics", {})
        manifest_link = _html_link(entry["refs"]["window_manifest"])
        scorecard_link = _html_link(entry["refs"]["scorecard"])
        task_batch_link = _html_link(entry["refs"]["task_batch"])
        rows.append(
            "<tr>"
            f"<td>{entry['window_id']}</td>"
            f"<td>{html.escape(entry['created_at'])}</td>"
            f"<td>{html.escape(entry['task_source'])}</td>"
            f"<td>{totals.get('accepted', 0)} / {totals.get('submitted', 0)}</td>"
            f"<td>{totals.get('hard_failed', 0)} / {totals.get('soft_failed', 0)}"
            f"<br><small>format {window_metrics.get('reasoning_format_ok_total', 0)}</small></td>"
            f"<td>{html.escape(weights)}</td>"
            f"<td>{manifest_link}<br>{scorecard_link}<br>{task_batch_link}</td>"
            "</tr>"
        )
    table_rows = "\n".join(rows) if rows else "<tr><td colspan='7'>No window manifests available.</td></tr>"
    generated_at = html.escape(str(payload["generated_at"]))
    model_ref = html.escape(str(payload["model_ref"]))
    network = html.escape(str(payload["network"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reliquary Audit Index</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; background: #0b1020; color: #eef2ff; }}
    h1, h2 {{ margin: 0 0 0.75rem; }}
    .meta {{ color: #c7d2fe; margin-bottom: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; background: #121a33; }}
    th, td {{ border: 1px solid #253055; padding: 0.65rem; vertical-align: top; text-align: left; }}
    th {{ background: #1b2550; }}
    a {{ color: #8bd3ff; }}
    code {{ font-family: ui-monospace, monospace; }}
  </style>
</head>
<body>
  <h1>Reliquary Audit Index</h1>
  <div class="meta">
    Generated at <code>{generated_at}</code><br>
    Network <code>{network}</code> | Model <code>{model_ref}</code> | Windows indexed <code>{payload['window_count']}</code>
  </div>
  <table>
    <thead>
      <tr>
        <th>Window</th>
        <th>Created</th>
        <th>Task Source</th>
        <th>Accepted / Submitted</th>
        <th>Hard / Soft Fails</th>
        <th>Weights</th>
        <th>Artifacts</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""


def _html_link(ref: dict[str, Any] | None) -> str:
    if not ref:
        return "-"
    label = html.escape(Path(str(ref.get("key", "-"))).name)
    url = ref.get("url")
    if url:
        return f'<a href="{html.escape(str(url))}">{label}</a>'
    return label

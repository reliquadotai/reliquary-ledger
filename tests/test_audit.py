from __future__ import annotations

from reliquary_inference.audit import build_audit_index
from reliquary_inference.protocol.artifacts import make_artifact
from reliquary_inference.storage.registry import LocalRegistry
from reliquary_inference.utils.json_io import read_json


def test_build_audit_index_prefers_finalized_manifest_and_renders_public_links(tmp_path) -> None:
    registry = LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))
    cfg = {
        "network": "test",
        "netuid": 1,
        "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "task_source": "reasoning_tasks",
        "public_audit_base_url": "https://pub.example.r2.dev",
        "expose_public_artifact_urls": True,
        "audit_prefix": "audit",
    }
    task_batch = make_artifact(
        artifact_type="task_batch",
        producer_id="validator",
        producer_role="task_source",
        window_id=123,
        payload={"tasks": []},
    )
    scorecard = make_artifact(
        artifact_type="scorecard",
        producer_id="validator",
        producer_role="validator",
        window_id=123,
        parent_ids=[task_batch["artifact_id"]],
        payload={
            "window_id": 123,
            "task_source": "reasoning_tasks",
            "weights": {"miner-1": 1.0},
            "miner_totals": {"miner-1": {"submitted": 2, "accepted": 2}},
            "verification_totals": {"submitted": 2, "accepted": 2, "hard_failed": 0, "soft_failed": 0},
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
        },
    )
    draft_manifest = make_artifact(
        artifact_type="window_manifest",
        producer_id="validator",
        producer_role="validator",
        window_id=123,
        parent_ids=[task_batch["artifact_id"], scorecard["artifact_id"]],
        payload={
            "task_batch_id": task_batch["artifact_id"],
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
            "scorecard_id": scorecard["artifact_id"],
            "task_source": "reasoning_tasks",
            "window_randomness_ref": {"block_hash": "0x01", "public_randomness": "abcd"},
            "chain_publish_result": None,
        },
    )
    finalized_manifest = make_artifact(
        artifact_type="window_manifest",
        producer_id="validator",
        producer_role="validator",
        window_id=123,
        parent_ids=draft_manifest["parent_ids"] + [draft_manifest["artifact_id"]],
        payload={
            **draft_manifest["payload"],
            "chain_publish_result": {"success": True, "uids": [7], "weights": [1.0]},
        },
    )
    registry.put_artifact(task_batch)
    registry.put_artifact(scorecard)
    registry.put_artifact(draft_manifest)
    registry.put_artifact(finalized_manifest)

    result = build_audit_index(cfg=cfg, registry=registry, limit=10, publish=False)

    payload = result["payload"]
    assert payload["window_count"] == 1
    assert payload["windows"][0]["chain_publish_result"]["success"] is True
    assert payload["windows"][0]["refs"]["window_manifest"]["url"].endswith(
        f"/window_manifests/{finalized_manifest['artifact_id']}.json"
    )
    assert "Reliquary Audit Index" in (tmp_path / "exports" / "audit" / "index.html").read_text(encoding="utf-8")
    assert read_json(tmp_path / "exports" / "audit" / "index.json")["windows"][0]["window_id"] == 123


def test_build_audit_index_hides_raw_artifact_urls_by_default(tmp_path) -> None:
    registry = LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))
    cfg = {
        "network": "test",
        "netuid": 1,
        "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "task_source": "reasoning_tasks",
        "public_audit_base_url": "https://pub.example.r2.dev",
        "audit_prefix": "audit",
    }
    task_batch = make_artifact(
        artifact_type="task_batch",
        producer_id="validator",
        producer_role="task_source",
        window_id=55,
        payload={"tasks": []},
    )
    scorecard = make_artifact(
        artifact_type="scorecard",
        producer_id="validator",
        producer_role="validator",
        window_id=55,
        parent_ids=[task_batch["artifact_id"]],
        payload={
            "window_id": 55,
            "task_source": "reasoning_tasks",
            "weights": {},
            "miner_totals": {},
            "verification_totals": {"submitted": 0, "accepted": 0, "hard_failed": 0, "soft_failed": 0},
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
        },
    )
    manifest = make_artifact(
        artifact_type="window_manifest",
        producer_id="validator",
        producer_role="validator",
        window_id=55,
        parent_ids=[task_batch["artifact_id"], scorecard["artifact_id"]],
        payload={
            "task_batch_id": task_batch["artifact_id"],
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
            "scorecard_id": scorecard["artifact_id"],
            "task_source": "reasoning_tasks",
            "window_randomness_ref": {"block_hash": "0x02", "public_randomness": "efgh"},
            "chain_publish_result": {"success": True, "uids": [1], "weights": [1.0]},
        },
    )
    registry.put_artifact(task_batch)
    registry.put_artifact(scorecard)
    registry.put_artifact(manifest)

    result = build_audit_index(cfg=cfg, registry=registry, limit=5, publish=False)

    assert "url" not in result["payload"]["windows"][0]["refs"]["window_manifest"]

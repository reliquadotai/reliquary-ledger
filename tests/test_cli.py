from reliquary_inference.chain.adapter import BittensorChainAdapter, LocalChainAdapter
from reliquary_inference.cli import _bucket_mode, _chain, _status_summary
from reliquary_inference.protocol.artifacts import make_artifact
from reliquary_inference.storage.registry import LocalRegistry


def test_chain_uses_local_adapter_for_local_network() -> None:
    cfg = {
        "network": "local",
        "netuid": 1,
        "wallet_name": "wallet",
        "hotkey_name": "validator",
        "wallet_path": "~/.bittensor/wallets",
        "use_drand": False,
    }
    adapter = _chain(cfg)
    assert isinstance(adapter, LocalChainAdapter)


def test_chain_uses_bittensor_adapter_for_test_network() -> None:
    cfg = {
        "network": "test",
        "netuid": 1,
        "wallet_name": "wallet",
        "hotkey_name": "validator",
        "wallet_path": "~/.bittensor/wallets",
        "use_drand": False,
    }
    adapter = _chain(cfg)
    assert isinstance(adapter, BittensorChainAdapter)


def test_status_summary_reports_latest_mined_and_published_windows(tmp_path) -> None:
    registry = LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))
    completion = make_artifact(
        artifact_type="completion",
        producer_id="miner-1",
        producer_role="miner",
        window_id=10,
        payload={"task_id": "task-1"},
    )
    manifest = make_artifact(
        artifact_type="window_manifest",
        producer_id="validator",
        producer_role="validator",
        window_id=9,
        payload={
            "task_batch_id": "task-batch-1",
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
            "scorecard_id": "scorecard-1",
            "task_source": "reasoning_tasks",
            "window_randomness_ref": {"block_hash": "0x01", "public_randomness": "abcd"},
            "chain_publish_result": {"success": True, "uids": [103], "weights": [1.0]},
        },
    )
    registry.put_artifact(completion)
    registry.put_artifact(manifest)
    cfg = {
        "network": "test",
        "netuid": 1,
        "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "task_source": "reasoning_tasks",
        "storage_backend": "r2",
        "audit_bucket": "reliquary-audit",
        "public_audit_base_url": "https://audit.example.com",
        "expose_public_artifact_urls": False,
    }

    summary = _status_summary(cfg, registry)

    assert summary["latest_window_mined"] == 10
    assert summary["latest_weight_publication"]["window_id"] == 9
    assert summary["bucket_mode"] == "private_artifacts_public_audit"
    assert summary["chain_endpoint_mode"] == "network_default"
    assert summary["artifact_bucket"] == ""
    assert summary["audit_bucket"] == "reliquary-audit"


def test_bucket_mode_marks_public_artifacts_when_urls_are_exposed() -> None:
    cfg = {
        "storage_backend": "r2",
        "audit_bucket": "",
        "public_audit_base_url": "https://pub.example.com",
        "expose_public_artifact_urls": True,
    }
    assert _bucket_mode(cfg) == "public_artifacts_public_audit"


def test_status_summary_hides_artifact_bucket_when_storage_is_local(tmp_path) -> None:
    registry = LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))
    cfg = {
        "network": "test",
        "netuid": 1,
        "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "task_source": "reasoning_tasks",
        "storage_backend": "local",
        "r2_bucket": "reliquary",
        "audit_bucket": "reliquary",
        "public_audit_base_url": "https://audit.example.com",
        "expose_public_artifact_urls": False,
    }

    summary = _status_summary(cfg, registry)

    assert summary["artifact_bucket"] == ""
    assert summary["audit_bucket"] == "reliquary"

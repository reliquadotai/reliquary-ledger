import json
from pathlib import Path
from types import SimpleNamespace

from reliquary_inference.audit import build_audit_index
from reliquary_inference.metrics import collect_metrics_snapshot, render_metrics
from reliquary_inference.protocol.artifacts import make_artifact
from reliquary_inference.storage.registry import LocalRegistry


class _FakeChain:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def get_current_block(self) -> int:
        if self.fail:
            raise RuntimeError("chain unavailable")
        return 123456

    def get_window_context(self, *, cfg, window_id=None):
        if self.fail:
            raise RuntimeError("chain unavailable")
        return SimpleNamespace(as_dict=lambda: {"window_id": 6911280})

    def get_metagraph(self):
        if self.fail:
            raise RuntimeError("chain unavailable")
        return SimpleNamespace(hotkeys=["5miner", "5validator"], uids=[103, 104])

    def describe_hotkeys(self, hotkeys):
        if self.fail:
            raise RuntimeError("chain unavailable")
        return {
            role: {
                "hotkey": hotkey,
                "registered": hotkey in {"5miner", "5validator"},
                "uid": 103 if hotkey == "5miner" else 104,
            }
            for role, hotkey in hotkeys.items()
        }


def _cfg(tmp_path: Path) -> dict:
    return {
        "network": "test",
        "netuid": 1,
        "model_ref": "Qwen/Qwen2.5-1.5B-Instruct",
        "task_source": "reasoning_tasks",
        "storage_backend": "local",
        "audit_bucket": "public-audit",
        "public_audit_base_url": "https://audit.example.com",
        "expose_public_artifact_urls": False,
        "artifact_dir": str(tmp_path / "artifacts"),
        "export_dir": str(tmp_path / "exports"),
        "metrics_window_count": 5,
        "wallet_public_file": str(tmp_path / "wallet-public.json"),
        "miner_ss58": "",
        "validator_ss58": "",
        "chain_endpoint": "wss://endpoint.example",
    }


def test_collect_metrics_snapshot_uses_audit_and_chain_state(tmp_path: Path) -> None:
    registry = LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))
    cfg = _cfg(tmp_path)
    Path(cfg["wallet_public_file"]).write_text(
        json.dumps(
            {
                "miner_hotkey_ss58": "5miner",
                "validator_hotkey_ss58": "5validator",
            }
        ),
        encoding="utf-8",
    )
    scorecard = make_artifact(
        artifact_type="scorecard",
        producer_id="validator",
        producer_role="validator",
        window_id=6911280,
        payload={
            "verification_totals": {
                "submitted": 4,
                "accepted": 3,
                "hard_failed": 1,
                "soft_failed": 0,
            },
            "window_metrics": {
                "reasoning_eval_count": 4,
                "reasoning_correct_total": 3.0,
                "reasoning_format_ok_total": 4.0,
                "reasoning_policy_compliance_total": 3.5,
            },
            "weights": {"5miner": 1.0},
            "miner_totals": {"5miner": {"accepted": 3}},
        },
    )
    manifest = make_artifact(
        artifact_type="window_manifest",
        producer_id="validator",
        producer_role="validator",
        window_id=6911280,
        payload={
            "task_batch_id": "task-batch-1",
            "completion_bundle_refs": [],
            "verdict_bundle_ref": None,
            "scorecard_id": scorecard["artifact_id"],
            "task_source": "reasoning_tasks",
            "window_randomness_ref": {"block_hash": "0x01", "public_randomness": "abcd"},
            "chain_publish_result": {"success": True, "uids": [103], "weights": [1.0]},
        },
    )
    registry.put_artifact(scorecard)
    registry.put_artifact(manifest)
    build_audit_index(cfg=cfg, registry=registry, limit=25, publish=False)

    snapshot, _, _ = collect_metrics_snapshot(
        cfg=cfg,
        registry=registry,
        chain=_FakeChain(),
        now=1_700_000_000.0,
    )

    assert snapshot["rolling_submitted_total"] == 4.0
    assert snapshot["rolling_accepted_total"] == 3.0
    assert snapshot["rolling_acceptance_rate"] == 0.75
    assert snapshot["rolling_reasoning_correct_rate"] == 0.75
    assert snapshot["latest_importable_window"] == 6911280.0
    assert snapshot["publish_success_total"] == 1.0
    assert snapshot["chain"]["hotkeys"]["miner"]["uid"] == 103

    rendered = render_metrics(snapshot)
    assert "reliquary_rolling_acceptance_rate 0.75" in rendered
    assert "reliquary_rolling_reasoning_format_rate 1.0" in rendered
    assert 'reliquary_task_source_accepted_total{task_source="reasoning_tasks"} 3.0' in rendered
    assert 'reliquary_hotkey_registered{hotkey="5miner",role="miner"} 1.0' in rendered


def test_collect_metrics_snapshot_preserves_last_good_chain_state(tmp_path: Path) -> None:
    registry = LocalRegistry(str(tmp_path / "artifacts"), str(tmp_path / "exports"))
    cfg = _cfg(tmp_path)
    previous_state = {
        "chain_scrape_success": 1.0,
        "metagraph_size": 2.0,
        "subnet_visible": 1.0,
        "current_block": 10.0,
        "chain_window_id": 20.0,
        "hotkeys": {"miner": {"hotkey": "5miner", "registered": True, "uid": 103}},
    }

    snapshot, _, last_success = collect_metrics_snapshot(
        cfg=cfg,
        registry=registry,
        chain=_FakeChain(fail=True),
        previous_chain_state=previous_state,
        last_successful_chain_scrape_at=100.0,
        now=160.0,
    )

    assert snapshot["chain"]["chain_scrape_success"] == 0.0
    assert snapshot["chain"]["metagraph_size"] == 2.0
    assert snapshot["chain_scrape_age_seconds"] == 60.0
    assert last_success == 100.0

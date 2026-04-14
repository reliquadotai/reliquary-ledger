from pathlib import Path

from reliquary_inference.protocol.artifacts import make_artifact
from reliquary_inference.storage.registry import FilesystemObjectStore, LocalRegistry, ObjectRegistry


def test_local_and_object_registry_round_trip(tmp_path: Path) -> None:
    artifact = make_artifact(
        artifact_type="task_batch",
        producer_id="task-source",
        producer_role="task_source",
        window_id=0,
        payload={"task_source": "reasoning_tasks", "tasks": []},
    )
    local = LocalRegistry(str(tmp_path / "local-artifacts"), str(tmp_path / "local-exports"))
    object_registry = ObjectRegistry(FilesystemObjectStore(str(tmp_path / "object-store")), str(tmp_path / "object-exports"))

    local.put_artifact(artifact)
    object_registry.put_artifact(artifact)

    assert local.get_artifact("task_batch", artifact["artifact_id"])["artifact_id"] == artifact["artifact_id"]
    assert object_registry.get_artifact("task_batch", artifact["artifact_id"])["artifact_id"] == artifact["artifact_id"]


def test_object_registry_completion_bundle_refs_keep_clean_miner_ids(tmp_path: Path) -> None:
    registry = ObjectRegistry(
        FilesystemObjectStore(str(tmp_path / "object-store")),
        str(tmp_path / "object-exports"),
    )
    ref = registry.write_completion_bundle(window_id=7, miner_id="miner-a", completions=[])
    bundles = registry.list_completion_bundles(window_id=7)

    assert ref["miner_id"] == "miner-a"
    assert len(bundles) == 1
    assert bundles[0]["miner_id"] == "miner-a"

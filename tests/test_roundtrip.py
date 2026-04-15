from pathlib import Path

from reliquary_inference.chain.adapter import LocalChainAdapter
from reliquary_inference.dataset.task_sources import build_task_source
from reliquary_inference.miner.engine import MiningEngine
from reliquary_inference.protocol.artifacts import make_artifact
from reliquary_inference.storage.registry import LocalRegistry
from reliquary_inference.validator.service import finalize_window_manifest, validate_window


def _cfg(tmp_path: Path, *, task_source: str) -> dict:
    return {
        "artifact_dir": str(tmp_path / "artifacts"),
        "export_dir": str(tmp_path / "exports"),
        "model_ref": "toy://local-inference-v1",
        "task_source": task_source,
        "task_count": 3,
        "samples_per_task": 1,
        "max_new_tokens": 64,
        "device": "cpu",
        "miner_mode": "single_gpu_hf",
        "miner_id": "miner-a",
        "validator_id": "validator-a",
        "signature_scheme": "local_hmac",
        "signing_secret": "local-secret",
        "local_signer_id": "miner-a",
        "dataset_name": "missing-dataset",
        "dataset_split": "train",
    }


def _task_batch(cfg: dict, window_context: dict) -> dict:
    source = build_task_source(window_context["task_source"])
    payload = source.build_window_batch(window_context, count=int(cfg["task_count"]))
    return make_artifact(
        artifact_type="task_batch",
        producer_id=str(cfg["validator_id"]),
        producer_role="task_source",
        window_id=int(window_context["window_id"]),
        payload=payload,
    )


def test_local_roundtrip_for_both_task_sources(tmp_path: Path) -> None:
    chain = LocalChainAdapter()
    for window_id, source_id in enumerate(("dataset_prompts", "reasoning_tasks")):
        cfg = _cfg(tmp_path, task_source=source_id)
        registry = LocalRegistry(cfg["artifact_dir"], cfg["export_dir"])
        window_context = chain.get_window_context(cfg=cfg, window_id=window_id).as_dict()
        task_batch = _task_batch(cfg, window_context)
        registry.put_artifact(task_batch)

        completion_refs = []
        for miner_id in ("miner-a", "miner-b"):
            engine = MiningEngine(cfg={**cfg, "miner_id": miner_id})
            completions = []
            for task in task_batch["payload"]["tasks"]:
                completion = engine.generate_completion(
                    task=task,
                    window_context=window_context,
                    registry=registry,
                    miner_id=miner_id,
                    sample_index=0,
                )
                assert isinstance(completion["payload"]["completion_token_ids"], list)
                assert isinstance(completion["payload"]["completion_logprobs"], list)
                assert completion["payload"]["old_sum_logprob"] == sum(
                    completion["payload"]["completion_logprobs"]
                )
                completions.append(completion)
            bundle_ref = registry.write_completion_bundle(window_id=window_id, miner_id=miner_id, completions=completions)
            completion_refs.append(bundle_ref)
            for completion in completions:
                completion["payload"]["upload_ref"] = bundle_ref
                registry.put_artifact(completion)

        verdicts, scorecard, window_manifest = validate_window(
            cfg=cfg,
            registry=registry,
            window_context=window_context,
            task_batch_artifact=task_batch,
        )
        assert verdicts
        assert scorecard["payload"]["verification_totals"]["submitted"] > 0
        assert scorecard["payload"]["weights"]
        if source_id == "reasoning_tasks":
            assert "window_metrics" in scorecard["payload"]
            assert scorecard["payload"]["window_metrics"]["reasoning_eval_count"] > 0
            assert any("final_answer" in verdict["payload"] for verdict in verdicts)

        publish_result = chain.publish_weights(window_id=window_id, weights=scorecard["payload"]["weights"])
        finalized = finalize_window_manifest(
            cfg=cfg,
            registry=registry,
            window_manifest=window_manifest,
            publish_result=publish_result,
        )
        assert finalized["payload"]["chain_publish_result"]["event_type"] == "weights"

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from ..protocol.artifacts import make_artifact
from ..utils.json_io import write_json
from .cooldown import CooldownMap, DEFAULT_COOLDOWN_WINDOWS, default_cooldown_path
from .batched_verify import (
    compute_cached_hidden_states,
    group_completions_for_batched_forward,
)
from .copycat import detect_index_copycats
from .verifier import verify_completion
from .weights import compute_weights
from .zone_filter import filter_groups, zone_summary


def validate_window(
    *,
    cfg: dict[str, Any],
    registry,
    window_context: dict[str, Any],
    task_batch_artifact: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    completion_bundle_refs = registry.list_completion_bundles(window_id=int(window_context["window_id"]))
    all_completions: list[dict[str, Any]] = []
    # GRPO requires M rollouts per task from the same miner — the duplicate
    # check must key on (task_id, sample_index) rather than task_id alone,
    # otherwise 7/8 rollouts get rejected as "duplicates" the moment
    # SAMPLES_PER_TASK > 1.
    seen_rollout_keys_by_miner: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for ref in completion_bundle_refs:
        completions = registry.read_completion_bundle(ref)
        for completion in completions:
            completion["payload"]["upload_ref"] = ref
        all_completions.extend(completions)
    duplicate_digest_losers = _duplicate_completion_losers(all_completions)
    dataset_copycats = _dataset_copycats(all_completions)
    verdicts: list[dict[str, Any]] = []
    seen_nonces: set[tuple[str, int]] = set()
    miner_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"submitted": 0, "accepted": 0, "valid": 0, "hard_failed": 0, "soft_failed": 0, "unique": 0})
    verification_totals = {
        "submitted": len(all_completions),
        "accepted": 0,
        "hard_failed": 0,
        "soft_failed": 0,
        "copycat_rejections": 0,
        "digest_rejections": 0,
    }
    window_metrics = {
        "submitted_count": len(all_completions),
        "accepted_count": 0,
        "training_eligible_count": 0,
        "hard_failed_count": 0,
        "soft_failed_count": 0,
        "copycat_rejections": 0,
        "digest_rejections": 0,
        "reasoning_eval_count": 0,
        "reasoning_correct_total": 0.0,
        "reasoning_format_ok_total": 0.0,
        "reasoning_policy_compliance_total": 0.0,
        "reasoning_difficulty_total": 0.0,
        "reasoning_final_answer_count": 0,
        "per_task_source": {},
    }
    # Batched proof verification: one forward pass per (miner, window)
    # group instead of per-completion. Mirrors the miner's batched
    # generate — essential for the validator to keep up with M=8 GRPO
    # groups. Feature-flag gated via cfg["batched_verify"] (default True).
    # Gracefully empty on any failure → per-completion ProofStage falls
    # back to its own forward.
    cached_hidden_states: dict[str, Any] = {}
    if bool(cfg.get("batched_verify", True)) and all_completions:
        try:
            from ..shared.modeling import load_model_bundle
            bundle = load_model_bundle(
                str(task_batch_artifact["payload"]["model_ref"]),
                device=str(cfg["device"]),
                dtype_name=str(cfg.get("load_dtype", "auto")),
                require_flash_attention=bool(cfg.get("require_flash_attention", False)),
            )
            from ..protocol.constants import LAYER_INDEX
            for batch in group_completions_for_batched_forward(
                all_completions,
                max_batch_size=int(cfg.get("batched_verify_max_size", 8)),
            ):
                cached_hidden_states.update(
                    compute_cached_hidden_states(
                        completions=batch,
                        model=bundle["model"],
                        tokenizer=bundle["tokenizer"],
                        layer_index=LAYER_INDEX,
                    )
                )
        except Exception:
            cached_hidden_states = {}

    for completion in all_completions:
        miner_id = completion["producer_id"]
        payload = completion["payload"]
        miner_totals[miner_id]["submitted"] += 1
        rollout_key = (payload["task_id"], int(payload.get("sample_index", 0)))
        duplicate_task = rollout_key in seen_rollout_keys_by_miner[miner_id]
        seen_rollout_keys_by_miner[miner_id].add(rollout_key)
        report = verify_completion(
            cfg=cfg,
            completion=completion,
            task_batch=task_batch_artifact,
            seen_nonces=seen_nonces,
            cached_hidden_states=cached_hidden_states,
        )
        if duplicate_task:
            report["accepted"] = False
            report["soft_fail_reason"] = "duplicate_task_submission"
        if completion["artifact_id"] in duplicate_digest_losers:
            report["accepted"] = False
            report["soft_fail_reason"] = "duplicate_completion_digest"
            report["copycat_status"] = "digest_duplicate"
        if completion["artifact_id"] in dataset_copycats:
            report["accepted"] = False
            report["soft_fail_reason"] = "dataset_copycat"
            report["copycat_status"] = "dataset_copycat"
        if report["accepted"]:
            miner_totals[miner_id]["accepted"] += 1
            miner_totals[miner_id]["valid"] += 1
            miner_totals[miner_id]["unique"] += 1
            verification_totals["accepted"] += 1
            window_metrics["accepted_count"] += 1
            window_metrics["training_eligible_count"] += 1
        elif report["hard_fail_reason"] is not None:
            miner_totals[miner_id]["hard_failed"] += 1
            verification_totals["hard_failed"] += 1
            window_metrics["hard_failed_count"] += 1
        else:
            miner_totals[miner_id]["soft_failed"] += 1
            verification_totals["soft_failed"] += 1
            window_metrics["soft_failed_count"] += 1
            if report["soft_fail_reason"] == "dataset_copycat":
                verification_totals["copycat_rejections"] += 1
                window_metrics["copycat_rejections"] += 1
            if report["soft_fail_reason"] == "duplicate_completion_digest":
                verification_totals["digest_rejections"] += 1
                window_metrics["digest_rejections"] += 1
        task = report.get("task_binding_summary", {}).get("task", {})
        task_source = str(report["task_source"])
        task_source_totals = window_metrics["per_task_source"].setdefault(
            task_source,
            {
                "submitted": 0,
                "accepted": 0,
                "correct_total": 0.0,
                "format_ok_total": 0.0,
                "policy_compliance_total": 0.0,
            },
        )
        task_source_totals["submitted"] += 1
        if report["accepted"]:
            task_source_totals["accepted"] += 1
        semantic_evaluation = report.get("semantic_evaluation", {})
        # The MATH task source's evaluate_math_trace returns the same
        # correctness_or_judge / format_ok / policy_compliance / final_answer
        # shape as reasoning_tasks, so the aggregate counters apply uniformly.
        if task_source in {"reasoning_tasks", "math"}:
            window_metrics["reasoning_eval_count"] += 1
            window_metrics["reasoning_correct_total"] += float(semantic_evaluation.get("correctness_or_judge", 0.0))
            window_metrics["reasoning_format_ok_total"] += 1.0 if semantic_evaluation.get("format_ok") else 0.0
            window_metrics["reasoning_policy_compliance_total"] += float(semantic_evaluation.get("policy_compliance", 0.0))
            window_metrics["reasoning_difficulty_total"] += float(task.get("difficulty", 0.0))
            if semantic_evaluation.get("final_answer") is not None:
                window_metrics["reasoning_final_answer_count"] += 1
            task_source_totals["correct_total"] += float(semantic_evaluation.get("correctness_or_judge", 0.0))
            task_source_totals["format_ok_total"] += 1.0 if semantic_evaluation.get("format_ok") else 0.0
            task_source_totals["policy_compliance_total"] += float(semantic_evaluation.get("policy_compliance", 0.0))
        contamination_tags = set(payload.get("contamination_tags", []))
        forbidden_overlap_tags = set(task.get("contamination_policy", {}).get("forbidden_overlap_tags", []))
        contamination_overlap = sorted(contamination_tags & forbidden_overlap_tags)
        verdicts.append(
            make_artifact(
                artifact_type="verdict",
                producer_id=str(cfg["validator_id"]),
                producer_role="validator",
                window_id=int(window_context["window_id"]),
                parent_ids=[completion["artifact_id"], task_batch_artifact["artifact_id"]],
                payload={
                    "completion_id": completion["artifact_id"],
                    # miner_id is the producer of the completion being
                    # judged here — NOT the validator that produced this
                    # verdict (that's ``producer_id`` on the artifact
                    # envelope). Zone filter groups by (miner_id, task_id);
                    # without this field it falls back to the verdict's
                    # producer_id = validator, which never matches the
                    # miner's hotkey in completion_bundles → in-zone keys
                    # never join back to rollout data → GRPO trainer
                    # silently assembles zero groups.
                    "miner_id": miner_id,
                    "sample_index": int(payload.get("sample_index", 0)),
                    "accepted": report["accepted"],
                    "hard_fail_reason": report["hard_fail_reason"],
                    "soft_fail_reason": report["soft_fail_reason"],
                    "proof_summary": {
                        "checked_positions": report["checked_positions"],
                        "passed_positions": report["passed_positions"],
                        "signature_status": report["signature_status"],
                    },
                    "task_binding_summary": report["task_binding_summary"],
                    "copycat_summary": {"status": report["copycat_status"]},
                    "task_source": report["task_source"],
                    "task_id": payload["task_id"],
                    "task_index": payload.get("task_index"),
                    "difficulty": float(task.get("difficulty", 0.0)),
                    "task_tags": list(task.get("tags", [])),
                    "contamination_tags": list(payload.get("contamination_tags", [])),
                    "contamination_overlap_tags": contamination_overlap,
                    "final_answer": semantic_evaluation.get("final_answer"),
                    "correctness": float(semantic_evaluation.get("correctness_or_judge", 0.0)),
                    "policy_compliance": float(semantic_evaluation.get("policy_compliance", 0.0)),
                    "format_ok": bool(semantic_evaluation.get("format_ok", False)),
                    "format_reason": semantic_evaluation.get("format_reason"),
                    "semantic_evaluation": semantic_evaluation,
                },
            )
        )
    verdict_bundle_ref = registry.write_verdict_bundle(
        window_id=int(window_context["window_id"]),
        validator_id=str(cfg["validator_id"]),
        verdicts=verdicts,
    )
    for verdict in verdicts:
        registry.put_artifact(verdict)

    # DAPO zone filter: for binary MATH rewards, group rollouts by
    # (miner_id, task_id) and keep only groups whose reward-std passes
    # σ ≥ 0.43 (bootstrap: 0.33). Out-of-zone groups carry no GRPO
    # gradient signal and would add pure noise to the trainer.
    bootstrap = bool(cfg.get("zone_filter_bootstrap", False))
    group_verdicts = filter_groups(verdicts, bootstrap=bootstrap)
    zone_info = zone_summary(group_verdicts, bootstrap=bootstrap)
    window_metrics["zone_filter"] = zone_info

    # DAPO cooldown: prompts that just contributed an in-zone group are
    # parked in the per-prompt cooldown map so the task builder skips
    # them for the next COOLDOWN_WINDOWS windows (forces curriculum rotation).
    _persist_cooldown_updates(
        cfg=cfg,
        window_id=int(window_context["window_id"]),
        task_batch_artifact=task_batch_artifact,
        group_verdicts=group_verdicts,
    )

    scorecard = score_window(
        cfg=cfg,
        registry=registry,
        window_context=window_context,
        task_batch_artifact=task_batch_artifact,
        verdicts=verdicts,
        completion_bundle_refs=completion_bundle_refs,
        verdict_bundle_ref=verdict_bundle_ref,
        miner_totals=miner_totals,
        verification_totals=verification_totals,
        window_metrics=window_metrics,
    )
    registry.put_artifact(scorecard)
    window_manifest = make_artifact(
        artifact_type="window_manifest",
        producer_id=str(cfg["validator_id"]),
        producer_role="validator",
        window_id=int(window_context["window_id"]),
        parent_ids=[task_batch_artifact["artifact_id"], scorecard["artifact_id"]],
        payload={
            "task_batch_id": task_batch_artifact["artifact_id"],
            "completion_bundle_refs": completion_bundle_refs,
            "verdict_bundle_ref": verdict_bundle_ref,
            "scorecard_id": scorecard["artifact_id"],
            "task_source": window_context["task_source"],
            "window_randomness_ref": {
                "block_hash": window_context["block_hash"],
                "public_randomness": window_context["public_randomness"],
            },
            "chain_publish_result": None,
        },
    )
    registry.put_artifact(window_manifest)
    return verdicts, scorecard, window_manifest


def score_window(
    *,
    cfg: dict[str, Any],
    registry,
    window_context: dict[str, Any],
    task_batch_artifact: dict[str, Any],
    verdicts: list[dict[str, Any]],
    completion_bundle_refs: list[dict[str, Any]],
    verdict_bundle_ref: dict[str, Any],
    miner_totals: dict[str, dict[str, int]],
    verification_totals: dict[str, int],
    window_metrics: dict[str, Any],
) -> dict[str, Any]:
    weights = compute_weights(miner_totals)
    return make_artifact(
        artifact_type="scorecard",
        producer_id=str(cfg["validator_id"]),
        producer_role="validator",
        window_id=int(window_context["window_id"]),
        parent_ids=[task_batch_artifact["artifact_id"]] + [verdict["artifact_id"] for verdict in verdicts],
        payload={
            "window_id": int(window_context["window_id"]),
            "task_source": window_context["task_source"],
            "weights": weights,
            "miner_totals": miner_totals,
            "verification_totals": verification_totals,
            "window_metrics": window_metrics,
            "completion_bundle_refs": completion_bundle_refs,
            "verdict_bundle_ref": verdict_bundle_ref,
        },
    )


def _persist_cooldown_updates(
    *,
    cfg: dict[str, Any],
    window_id: int,
    task_batch_artifact: dict[str, Any],
    group_verdicts,
) -> None:
    """Park in-zone dataset_indices in the cooldown map so the next task
    batch skips them.

    We look up each in-zone group's ``task_id`` in the task batch to
    recover its ``dataset_index``; that integer is the key the task
    source uses when sampling. Unknown task_ids (shouldn't happen) are
    silently skipped.

    Cooldown file lives at ``{state_dir}/cooldown.json``. On first
    boot the file doesn't exist and ``load()`` is a no-op, so this is
    safe to call unconditionally.
    """
    state_dir = cfg.get("state_dir") or cfg.get("local_root")
    if not state_dir:
        return
    horizon = int(cfg.get("cooldown_windows", DEFAULT_COOLDOWN_WINDOWS))
    if horizon <= 0:
        return
    task_by_id = {
        task["task_id"]: task
        for task in task_batch_artifact["payload"].get("tasks", [])
    }
    in_zone_indices: set[int] = set()
    for group in group_verdicts.values():
        if not group.in_zone:
            continue
        task = task_by_id.get(group.task_id)
        if not task:
            continue
        idx = task.get("dataset_index")
        if idx is None:
            continue
        in_zone_indices.add(int(idx))
    if not in_zone_indices:
        return
    cooldown = CooldownMap(cooldown_windows=horizon)
    path = default_cooldown_path(state_dir)
    # Prefer the local file as authoritative; if missing, try R2 restore
    # before starting fresh.  Avoids an empty map right after a node
    # rebuild re-sampling prompts that were supposed to be parked.
    cooldown.load(path)
    backend = _cooldown_r2_backend(cfg)
    r2_key = str(cfg.get("cooldown_r2_key", "") or "")
    if len(cooldown) == 0 and backend is not None and r2_key:
        cooldown.load_r2(backend, r2_key=r2_key)
    cooldown.record_batched_many(in_zone_indices, window=window_id)
    cooldown.prune(current_window=window_id)
    cooldown.save(path)
    if backend is not None and r2_key:
        cooldown.save_r2(backend, r2_key=r2_key)


def _cooldown_r2_backend(cfg: dict[str, Any]):
    """Construct an R2 backend from cfg for cooldown backup. Returns
    None if any required piece is missing — caller then skips R2 I/O
    entirely (local file still works)."""
    account_id = str(cfg.get("r2_rest_account_id", "") or "")
    bucket = str(cfg.get("r2_rest_bucket", "") or "")
    token = str(cfg.get("r2_rest_cf_api_token", "") or "")
    if not (account_id and bucket and token):
        return None
    try:
        from reliquary_protocol.storage import R2ObjectBackend
        return R2ObjectBackend(
            account_id=account_id,
            bucket=bucket,
            cf_api_token=token,
            public_url=str(cfg.get("r2_rest_public_url", "") or "") or None,
        )
    except Exception:
        return None


def finalize_window_manifest(
    *,
    cfg: dict[str, Any],
    registry,
    window_manifest: dict[str, Any],
    publish_result: dict[str, Any],
) -> dict[str, Any]:
    finalized = make_artifact(
        artifact_type="window_manifest",
        producer_id=str(cfg["validator_id"]),
        producer_role="validator",
        window_id=int(window_manifest["window_id"]),
        parent_ids=window_manifest["parent_ids"] + [window_manifest["artifact_id"]],
        payload={
            **window_manifest["payload"],
            "chain_publish_result": publish_result,
        },
    )
    registry.put_artifact(finalized)
    return finalized


def write_run_manifest(*, registry, run_id: str, window_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = registry.run_dir(run_id)
    run_manifest = make_artifact(
        artifact_type="run_manifest",
        producer_id="local-runner",
        producer_role="operator",
        window_id=window_manifests[-1]["window_id"] if window_manifests else 0,
        parent_ids=[artifact["artifact_id"] for artifact in window_manifests],
        payload={
            "window_manifest_ids": [artifact["artifact_id"] for artifact in window_manifests],
            "window_count": len(window_manifests),
            "task_sources": [artifact["payload"]["task_source"] for artifact in window_manifests],
        },
    )
    registry.put_artifact(run_manifest)
    write_json(run_dir / "run-manifest.json", run_manifest)
    return run_manifest


def _duplicate_completion_losers(completions: list[dict[str, Any]]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for completion in completions:
        grouped[completion["payload"]["completion_digest"]].append(completion)
    losers: set[str] = set()
    for matches in grouped.values():
        if len(matches) <= 1:
            continue
        if len({match["producer_id"] for match in matches}) <= 1:
            continue
        matches = sorted(matches, key=lambda item: item["payload"]["upload_ref"].get("uploaded_at", float("inf")))
        winner_miner = matches[0]["producer_id"]
        for duplicate in matches[1:]:
            if duplicate["producer_id"] != winner_miner:
                losers.add(duplicate["artifact_id"])
    return losers


def _dataset_copycats(completions: list[dict[str, Any]]) -> set[str]:
    submissions: dict[str, dict[str, Any]] = {}
    for completion in completions:
        if completion["payload"]["task_source"] != "dataset_prompts":
            continue
        miner_id = completion["producer_id"]
        submission = submissions.setdefault(
            miner_id,
            {
                "indices": set(),
                "upload_time": completion["payload"]["upload_ref"].get("uploaded_at"),
                "artifact_ids": {},
            },
        )
        dataset_index = completion["payload"].get("task_index")
        if isinstance(dataset_index, int):
            submission["indices"].add(dataset_index)
            submission["artifact_ids"][dataset_index] = completion["artifact_id"]
    rejected = detect_index_copycats(submissions)
    rejected_artifacts: set[str] = set()
    for miner_id, indices in rejected.items():
        artifact_ids = submissions[miner_id]["artifact_ids"]
        for dataset_index in indices:
            rejected_artifacts.add(artifact_ids[dataset_index])
    return rejected_artifacts

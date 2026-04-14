from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Annotated

import typer
from rich.console import Console

from .audit import build_audit_index
from .chain.adapter import BittensorChainAdapter, LocalChainAdapter
from .config import load_config
from .dataset.task_sources import build_task_source
from .miner.engine import MiningEngine
from .protocol.artifacts import make_artifact
from .storage.registry import LocalRegistry, R2Registry
from .utils.json_io import read_json, write_json
from .validator.service import finalize_window_manifest, validate_window, write_run_manifest

app = typer.Typer(no_args_is_help=True)
console = Console()


def _cfg() -> dict:
    return load_config()


def _registry(cfg: dict):
    if cfg["storage_backend"] == "r2":
        return R2Registry(
            artifact_root=str(cfg["artifact_dir"]),
            export_root=str(cfg["export_dir"]),
            bucket=str(cfg["r2_bucket"]),
            endpoint_url=str(cfg["r2_endpoint_url"]),
            access_key_id=str(cfg["r2_access_key_id"]),
            secret_access_key=str(cfg["r2_secret_access_key"]),
        )
    return LocalRegistry(str(cfg["artifact_dir"]), str(cfg["export_dir"]))


def _chain(cfg: dict):
    if str(cfg["network"]) in {"local", "mock"}:
        return LocalChainAdapter()
    return BittensorChainAdapter(
        network=str(cfg["network"]),
        chain_endpoint=str(cfg.get("chain_endpoint", "")),
        netuid=int(cfg["netuid"]),
        wallet_name=str(cfg["wallet_name"]),
        hotkey_name=str(cfg["hotkey_name"]),
        wallet_path=str(cfg["wallet_path"]),
        use_drand=bool(cfg["use_drand"]),
    )


def _task_batch_artifact(cfg: dict, window_context: dict, count: int) -> dict:
    source = build_task_source(window_context["task_source"])
    payload = source.build_window_batch(window_context, count=count)
    return make_artifact(
        artifact_type="task_batch",
        producer_id=str(cfg["validator_id"]),
        producer_role="task_source",
        window_id=int(window_context["window_id"]),
        payload=payload,
    )


def _latest_or_new_task_batch(cfg: dict, registry, window_context: dict, count: int) -> dict:
    task_batches = registry.list_artifacts("task_batch", window_id=int(window_context["window_id"]))
    if task_batches:
        return task_batches[-1]
    artifact = _task_batch_artifact(cfg, window_context, count)
    registry.put_artifact(artifact)
    return artifact


def _mine_single_window(cfg: dict, registry, window_context: dict, miner_id: str) -> int:
    task_batch = _latest_or_new_task_batch(cfg, registry, window_context, int(cfg["task_count"]))
    engine = MiningEngine(cfg=cfg)
    completions = []
    for task in task_batch["payload"]["tasks"]:
        for sample_index in range(int(cfg["samples_per_task"])):
            completion = engine.generate_completion(
                task=task,
                window_context=window_context,
                registry=registry,
                miner_id=miner_id,
                sample_index=sample_index,
            )
            completions.append(completion)
    bundle_ref = registry.write_completion_bundle(
        window_id=int(window_context["window_id"]),
        miner_id=miner_id,
        completions=completions,
    )
    for completion in completions:
        completion["payload"]["upload_ref"] = bundle_ref
        registry.put_artifact(completion)
    return len(completions)


def _validate_single_window(cfg: dict, registry, window_context: dict) -> tuple[list[dict], dict, dict]:
    task_batches = registry.list_artifacts("task_batch", window_id=int(window_context["window_id"]))
    if not task_batches:
        raise typer.BadParameter("No task batch found for the target window.")
    return validate_window(
        cfg=cfg,
        registry=registry,
        window_context=window_context,
        task_batch_artifact=task_batches[-1],
    )


def _validate_and_score_single_window(cfg: dict, registry, chain, window_context: dict) -> tuple[dict, dict]:
    _, scorecard, window_manifest = _validate_single_window(cfg, registry, window_context)
    publish_result = chain.publish_weights(
        window_id=int(window_context["window_id"]),
        weights=scorecard["payload"]["weights"],
    )
    finalized = finalize_window_manifest(
        cfg=cfg,
        registry=registry,
        window_manifest=window_manifest,
        publish_result=publish_result,
    )
    return scorecard, finalized


def _bucket_mode(cfg: dict) -> str:
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


def _chain_endpoint_mode(cfg: dict) -> str:
    endpoint = str(cfg.get("chain_endpoint", "")).strip()
    if str(cfg["network"]) in {"local", "mock"}:
        return "local"
    if endpoint:
        return "dedicated"
    return "network_default"


def _status_summary(cfg: dict, registry) -> dict:
    latest_completion = None
    latest_manifest = None
    latest_publish = None
    export_root = cfg.get("export_dir", getattr(registry, "export_root", "./exports"))
    audit_index_path = Path(str(export_root)) / "audit" / "index.json"
    if audit_index_path.exists():
        payload = read_json(audit_index_path)
        if payload.get("windows"):
            latest_window = payload["windows"][0]
            latest_manifest = {
                "window_id": latest_window["window_id"],
                "payload": {
                    "chain_publish_result": latest_window.get("chain_publish_result"),
                },
            }
            latest_publish = latest_window.get("chain_publish_result")
            latest_completion = {"window_id": latest_window["window_id"]}
    if latest_manifest is None:
        completions = registry.list_artifacts("completion")
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
    return {
        "network": cfg["network"],
        "netuid": int(cfg["netuid"]),
        "model_ref": cfg["model_ref"],
        "task_source": cfg["task_source"],
        "storage_backend": cfg["storage_backend"],
        "bucket_mode": _bucket_mode(cfg),
        "chain_endpoint_mode": _chain_endpoint_mode(cfg),
        "chain_endpoint": cfg.get("chain_endpoint", ""),
        "artifact_bucket": cfg.get("r2_bucket", "") if str(cfg["storage_backend"]) == "r2" else "",
        "audit_bucket": cfg.get("audit_bucket", ""),
        "public_audit_base_url": cfg.get("public_audit_base_url", ""),
        "latest_window_mined": int(latest_completion["window_id"]) if latest_completion else None,
        "latest_weight_publication": {
            "window_id": int(latest_manifest["window_id"]) if latest_manifest else None,
            "success": latest_publish.get("success") if isinstance(latest_publish, dict) else None,
            "uids": latest_publish.get("uids") if isinstance(latest_publish, dict) else None,
        },
    }


@app.command("publish-tasks")
def publish_tasks(
    source: Annotated[str | None, typer.Option("--source")] = None,
    count: Annotated[int | None, typer.Option("--count")] = None,
    window: Annotated[int | None, typer.Option("--window")] = None,
) -> None:
    cfg = _cfg()
    if source:
        cfg["task_source"] = source
    chain = _chain(cfg)
    window_context = chain.get_window_context(cfg=cfg, window_id=window).as_dict()
    registry = _registry(cfg)
    artifact = _task_batch_artifact(cfg, window_context, count or int(cfg["task_count"]))
    registry.put_artifact(artifact)
    console.print(artifact["artifact_id"])


@app.command("mine-window")
def mine_window(
    window: Annotated[int | None, typer.Option("--window")] = None,
    miner_id: Annotated[str | None, typer.Option("--miner-id")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
) -> None:
    cfg = _cfg()
    if source:
        cfg["task_source"] = source
    miner_id = miner_id or str(cfg["miner_id"])
    registry = _registry(cfg)
    chain = _chain(cfg)
    window_context = chain.get_window_context(cfg=cfg, window_id=window).as_dict()
    mined = _mine_single_window(cfg, registry, window_context, miner_id)
    console.print(f"mined {mined} completions")


@app.command("validate-window")
def validate_window_command(
    window: Annotated[int | None, typer.Option("--window")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
) -> None:
    cfg = _cfg()
    if source:
        cfg["task_source"] = source
    registry = _registry(cfg)
    chain = _chain(cfg)
    window_context = chain.get_window_context(cfg=cfg, window_id=window).as_dict()
    verdicts, scorecard, window_manifest = _validate_single_window(cfg, registry, window_context)
    console.print(
        f"validated {len(verdicts)} completions, accepted {scorecard['payload']['verification_totals']['accepted']}"
    )
    console.print(window_manifest["artifact_id"])


@app.command("score-window")
def score_window_command(
    window: Annotated[int | None, typer.Option("--window")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
) -> None:
    cfg = _cfg()
    if source:
        cfg["task_source"] = source
    registry = _registry(cfg)
    chain = _chain(cfg)
    window_context = chain.get_window_context(cfg=cfg, window_id=window).as_dict()
    scorecard = registry.list_artifacts("scorecard", window_id=int(window_context["window_id"]))[-1]
    publish_result = chain.publish_weights(
        window_id=int(window_context["window_id"]),
        weights=scorecard["payload"]["weights"],
    )
    manifests = registry.list_artifacts("window_manifest", window_id=int(window_context["window_id"]))
    if manifests:
        finalize_window_manifest(cfg=cfg, registry=registry, window_manifest=manifests[-1], publish_result=publish_result)
    console.print(publish_result)


@app.command("demo-local")
def demo_local(
    source: Annotated[str, typer.Option("--source")] = "both",
    count: Annotated[int | None, typer.Option("--count")] = None,
) -> None:
    cfg = _cfg()
    cfg["network"] = "local"
    registry = _registry(cfg)
    chain = LocalChainAdapter()
    sources = ["dataset_prompts", "reasoning_tasks"] if source == "both" else [source]
    window_manifests = []
    for offset, source_id in enumerate(sources):
        cfg["task_source"] = source_id
        window_context = chain.get_window_context(cfg=cfg, window_id=offset).as_dict()
        task_batch = _task_batch_artifact(cfg, window_context, count or int(cfg["task_count"]))
        registry.put_artifact(task_batch)
        for miner_id in ("local-miner-a", "local-miner-b"):
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
                completions.append(completion)
            bundle_ref = registry.write_completion_bundle(
                window_id=offset,
                miner_id=miner_id,
                completions=completions,
            )
            for completion in completions:
                completion["payload"]["upload_ref"] = bundle_ref
                registry.put_artifact(completion)
        _, scorecard, window_manifest = validate_window(
            cfg=cfg,
            registry=registry,
            window_context=window_context,
            task_batch_artifact=task_batch,
        )
        publish_result = chain.publish_weights(window_id=offset, weights=scorecard["payload"]["weights"])
        finalized_manifest = finalize_window_manifest(
            cfg=cfg,
            registry=registry,
            window_manifest=window_manifest,
            publish_result=publish_result,
        )
        window_manifests.append(finalized_manifest)
    run_manifest = write_run_manifest(registry=registry, run_id="demo-local", window_manifests=window_manifests)
    run_dir = registry.run_dir("demo-local")
    write_json(run_dir / "chain-events.json", chain.events())
    write_json(run_dir / "run-manifest.json", run_manifest)
    console.print(f"demo complete: {run_dir}")


@app.command("build-audit-index")
def build_audit_index_command(
    limit: Annotated[int, typer.Option("--limit")] = 25,
    publish: Annotated[bool, typer.Option("--publish")] = False,
    public_base_url: Annotated[str | None, typer.Option("--public-base-url")] = None,
) -> None:
    cfg = _cfg()
    if public_base_url:
        cfg["public_audit_base_url"] = public_base_url
    registry = _registry(cfg)
    result = build_audit_index(cfg=cfg, registry=registry, limit=limit, publish=publish)
    console.print(f"audit json: {result['json_path']}")
    console.print(f"audit html: {result['html_path']}")
    if result["published"]:
        console.print(result["published"])


@app.command("status")
def status_command(
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    cfg = _cfg()
    registry = _registry(cfg)
    summary = _status_summary(cfg, registry)
    if as_json:
        console.print_json(json.dumps(summary))
        return
    console.print(f"network: [bold]{summary['network']}[/bold] netuid={summary['netuid']}")
    console.print(f"model: [bold]{summary['model_ref']}[/bold]")
    console.print(f"task source: [bold]{summary['task_source']}[/bold]")
    console.print(f"storage: [bold]{summary['storage_backend']}[/bold] ({summary['bucket_mode']})")
    console.print(
        "chain endpoint: "
        f"[bold]{summary['chain_endpoint_mode']}[/bold] "
        f"{summary['chain_endpoint'] or '(network default)'}"
    )
    console.print(
        "audit target: "
        f"[bold]{summary['public_audit_base_url'] or '(none)'}[/bold] "
        f"artifact_bucket={summary['artifact_bucket'] or '(none)'} "
        f"audit_bucket={summary['audit_bucket'] or '(none)'}"
    )
    console.print(f"latest window mined: [bold]{summary['latest_window_mined']}[/bold]")
    latest_publish = summary["latest_weight_publication"]
    console.print(
        "latest weight publication: "
        f"[bold]{latest_publish['window_id']}[/bold] "
        f"success={latest_publish['success']} uids={latest_publish['uids']}"
    )


@app.command("run-miner")
def run_miner(
    once: Annotated[bool, typer.Option("--once")] = False,
    poll_interval: Annotated[int | None, typer.Option("--poll-interval")] = None,
) -> None:
    cfg = _cfg()
    interval = int(cfg["poll_interval"]) if poll_interval is None else poll_interval
    registry = _registry(cfg)
    chain = _chain(cfg)
    processed: set[tuple[int, str]] = set()
    while True:
        try:
            window_context = chain.get_window_context(cfg=cfg).as_dict()
            key = (int(window_context["window_id"]), str(cfg["miner_id"]))
            if key not in processed:
                mined = _mine_single_window(cfg, registry, window_context, str(cfg["miner_id"]))
                processed.add(key)
                console.print(f"mined {mined} completions for window {window_context['window_id']}")
            if once:
                return
        except Exception as exc:
            if once:
                raise
            console.print(f"[yellow]miner loop error: {exc}[/yellow]")
        time.sleep(interval)


@app.command("run-validator")
def run_validator(
    once: Annotated[bool, typer.Option("--once")] = False,
    poll_interval: Annotated[int | None, typer.Option("--poll-interval")] = None,
) -> None:
    cfg = _cfg()
    interval = int(cfg["poll_interval"]) if poll_interval is None else poll_interval
    registry = _registry(cfg)
    chain = _chain(cfg)
    processed: set[int] = set()
    while True:
        try:
            window_context = chain.get_window_context(cfg=cfg).as_dict()
            window_id = int(window_context["window_id"])
            if window_id not in processed and registry.list_completion_bundles(window_id=window_id):
                scorecard, _ = _validate_and_score_single_window(cfg, registry, chain, window_context)
                processed.add(window_id)
                console.print(f"published weights for window {window_id}: {scorecard['payload']['weights']}")
            if once:
                return
        except Exception as exc:
            if once:
                raise
            console.print(f"[yellow]validator loop error: {exc}[/yellow]")
        time.sleep(interval)

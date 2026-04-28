from __future__ import annotations

import json
import time
from typing import Annotated

import typer
from rich.console import Console

from .audit import build_audit_index
from .chain.adapter import BittensorChainAdapter, LocalChainAdapter
from .config import load_config
from .dataset.task_sources import build_task_source
from .metrics import MetricsCache, serve_metrics
from .miner.engine import MiningEngine
from .protocol.artifacts import make_artifact
from .status import status_summary
from .storage.registry import LocalRegistry, R2Registry, RestR2Registry
from .utils.json_io import write_json
from .validator.cooldown import (
    DEFAULT_COOLDOWN_WINDOWS,
    CooldownMap,
    default_cooldown_path,
)
from .validator.service import (
    finalize_window_manifest,
    validate_window,
    write_run_manifest,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


def _cfg() -> dict:
    return load_config()


def _miner_optimized_enabled() -> bool:
    """Honour ``RELIQUARY_INFERENCE_MINER_OPTIMIZED`` to opt into the
    competitive-miner reference (frontier-σ prompt selection +
    cooldown-aware + local σ gate). Defaults to off so the baseline
    deterministic miner is what runs unless an operator explicitly
    enables it. See ``reliquary_inference/miner/optimized_engine.py``.
    """
    import os as _os

    return _os.environ.get("RELIQUARY_INFERENCE_MINER_OPTIMIZED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _make_mining_engine(cfg: dict):
    """Construct the mining engine, honouring the optimized flag.

    Returns ``OptimizedMiningEngine`` when
    ``RELIQUARY_INFERENCE_MINER_OPTIMIZED`` is set, otherwise the
    baseline ``MiningEngine``. Both share the same constructor
    signature beyond the optimized engine's tuning kwargs (which take
    sensible defaults), so call-sites are interchangeable.
    """
    if _miner_optimized_enabled():
        from .miner.optimized_engine import OptimizedMiningEngine

        console.print(
            "[cyan]miner=optimized (frontier-σ prompt selection + "
            "cooldown-aware + local σ gate)[/cyan]"
        )
        return OptimizedMiningEngine(cfg=cfg)
    return MiningEngine(cfg=cfg)


def _apply_resume_from(cfg: dict, raw_source: str) -> None:
    """Apply a ``--resume-from`` source string to ``cfg`` and log the change.

    Thin CLI-side wrapper around
    ``reliquary_inference.validator.resume.apply_resume_from`` so the bulk
    of the resolve logic is testable without pulling in typer/rich.
    """
    from .validator.resume import apply_resume_from

    apply_resume_from(cfg, raw_source)
    console.print(
        f"[cyan]--resume-from {raw_source} → model_ref={cfg['model_ref']}[/cyan]"
    )


def _start_health_server(cfg: dict):
    """Start the structured /health HTTP server in a daemon thread.

    Returns the ``HealthSignalsHolder`` so the main loop can update
    signals on each iteration, or ``None`` if disabled.

    Phase 3.1 wiring (from the mainnet-readiness audit). The endpoint
    serves /health (structured HealthReport) and /healthz (binary)
    with HTTP status reflecting state (200 OK / 200 DEGRADED /
    503 UNHEALTHY). See ``reliquary_inference.shared.health_server``
    for the handler implementation.
    """
    import os as _os
    import threading as _threading
    import time as _time

    bind = _os.environ.get("RELIQUARY_INFERENCE_HEALTH_BIND", "127.0.0.1")
    port_raw = _os.environ.get("RELIQUARY_INFERENCE_HEALTH_PORT")
    if not port_raw:
        return None
    try:
        port = int(port_raw)
    except ValueError:
        console.print(
            f"[yellow]invalid RELIQUARY_INFERENCE_HEALTH_PORT={port_raw!r}; "
            "skipping health server[/yellow]"
        )
        return None

    from .shared.health import HealthSignals
    from .shared.health_server import HealthSignalsHolder, make_server

    holder = HealthSignalsHolder(
        HealthSignals(started_at=_time.time(), model_loaded=False)
    )
    try:
        server = make_server(bind=bind, port=port, holder=holder)
    except OSError as exc:
        console.print(
            f"[yellow]could not bind health server on {bind}:{port} ({exc}); "
            "continuing without /health[/yellow]"
        )
        return None
    thread = _threading.Thread(
        target=server.serve_forever, daemon=True, name="reliquary-health"
    )
    thread.start()
    console.print(f"[cyan]health server listening on {bind}:{port}[/cyan]")
    return holder


def _update_health(
    holder,
    *,
    started_at: float,
    chain_ok: bool,
    window_verified: bool,
    model_loaded: bool,
    now: float | None = None,
) -> None:
    """Snapshot-update the HealthSignalsHolder from inside the main loop.

    Idempotent / no-op when ``holder`` is None (health server disabled).
    """
    if holder is None:
        return
    import time as _time

    now = _time.time() if now is None else now
    snap, _ = holder.snapshot()
    last_chain = now if chain_ok else snap.last_chain_ok_at
    last_window = now if window_verified else snap.last_window_verified_at
    from .shared.health import HealthSignals

    holder.update(
        HealthSignals(
            started_at=started_at,
            last_chain_ok_at=last_chain,
            last_window_verified_at=last_window,
            last_proof_worker_heartbeat_at=snap.last_proof_worker_heartbeat_at,
            model_loaded=bool(model_loaded) or bool(snap.model_loaded),
        )
    )


class _StorageBackendShim:
    """Adapts the ObjectStore contract (put_bytes/get_bytes/list_prefix)
    used by :class:`reliquary_inference.storage.registry.ObjectRegistry`'s
    backing stores (FilesystemObjectStore / R2ObjectStore / RestR2ObjectStore)
    into the :class:`reliquary_inference.validator.verdict_storage.StorageBackend`
    Protocol (put/get/list/delete) that PolicyConsumer + RolloutBundleFetcher
    expect.

    Two API surfaces for a ~same-shaped object is a quirk of the repo's
    evolution; this shim is the single place we reconcile them.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def put(self, key: str, data: bytes) -> None:
        self._inner.put_bytes(key, data)

    def get(self, key: str):
        # ObjectStore.get_bytes raises FileNotFoundError on missing; the
        # StorageBackend Protocol expects None-on-missing.
        try:
            return self._inner.get_bytes(key)
        except FileNotFoundError:
            return None

    def list(self, prefix: str) -> list[str]:
        return [ref["key"] for ref in self._inner.list_prefix(prefix)]

    def delete(self, key: str) -> None:
        # Not all ObjectStores expose delete; best-effort.
        delete_fn = getattr(self._inner, "delete", None) or getattr(self._inner, "delete_bytes", None)
        if delete_fn is not None:
            try:
                delete_fn(key)
            except FileNotFoundError:
                pass


def _registry(cfg: dict):
    backend = cfg["storage_backend"]
    if backend == "r2":
        return R2Registry(
            artifact_root=str(cfg["artifact_dir"]),
            export_root=str(cfg["export_dir"]),
            bucket=str(cfg["r2_bucket"]),
            endpoint_url=str(cfg["r2_endpoint_url"]),
            access_key_id=str(cfg["r2_access_key_id"]),
            secret_access_key=str(cfg["r2_secret_access_key"]),
        )
    if backend == "r2_rest":
        # CF REST API path — reuses reliquary-protocol's R2ObjectBackend.
        # Authenticates with a single ``cfat_...`` account-level API token
        # rather than an S3-style access-key/secret pair.
        return RestR2Registry(
            artifact_root=str(cfg["artifact_dir"]),
            export_root=str(cfg["export_dir"]),
            account_id=str(cfg["r2_rest_account_id"]),
            bucket=str(cfg["r2_rest_bucket"]),
            cf_api_token=str(cfg["r2_rest_cf_api_token"]),
            public_url=str(cfg["r2_rest_public_url"]) or None,
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


def _load_cooldown_map(cfg: dict) -> CooldownMap:
    """Load the persistent per-prompt cooldown map from the validator state dir.

    Falls back to a fresh empty map if no file exists — that's the correct
    behaviour on first boot. Horizon comes from ``cfg['cooldown_windows']``
    (env: ``RELIQUARY_INFERENCE_COOLDOWN_WINDOWS``) and defaults to 50.
    """
    horizon = int(cfg.get("cooldown_windows", DEFAULT_COOLDOWN_WINDOWS))
    cooldown = CooldownMap(cooldown_windows=horizon)
    state_dir = cfg.get("state_dir") or cfg.get("local_root")
    if state_dir:
        cooldown.load(default_cooldown_path(state_dir))
    return cooldown


def _task_batch_artifact(cfg: dict, window_context: dict, count: int) -> dict:
    # Inject the current per-prompt cooldown set so deterministic task sources
    # (e.g. MathTasksSource) can skip prompts that were recently batched — DAPO
    # curriculum-diversity guard. Sources that don't honour the key are fine;
    # they just ignore it.
    current_window = int(window_context["window_id"])
    cooldown = _load_cooldown_map(cfg)
    enriched = dict(window_context)
    enriched.setdefault(
        "cooldown_indices", sorted(cooldown.current_cooldown_set(current_window))
    )
    source = build_task_source(
        enriched["task_source"],
        max_level=cfg.get("math_max_level"),
        mix=cfg.get("task_mix"),
    )
    payload = source.build_window_batch(enriched, count=count)
    return make_artifact(
        artifact_type="task_batch",
        producer_id=str(cfg["validator_id"]),
        producer_role="task_source",
        window_id=current_window,
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
    engine = _make_mining_engine(cfg)
    samples_per_task = int(cfg["samples_per_task"])
    completions: list[dict] = []
    if _miner_optimized_enabled():
        # Route candidates through the optimized engine's frontier-σ
        # selection. ``select_prompts`` drops cooldown task ids and
        # picks ``samples_per_task * task_count`` prompts most likely
        # to land in the validator's σ ≥ 0.43 zone. We pick at most
        # the original task_batch size so the rollout count stays
        # constant — the optimization is *which* prompts, not how
        # many. ``cooldown_task_ids`` here is best-effort: the
        # validator-side cooldown is the source of truth, but a miner
        # that asked the registry directly would still skip recently
        # batched prompts. With no registry-side cooldown surface
        # exposed yet we pass an empty set; a follow-on can wire the
        # CooldownMap in.
        all_tasks = list(task_batch["payload"]["tasks"])
        try:
            n = max(1, len(all_tasks))
            selected = engine.select_prompts(all_tasks, n=n, cooldown_task_ids=set())
            if selected:
                tasks_iter = selected
            else:
                tasks_iter = all_tasks
        except Exception as _exc:  # pragma: no cover — never block the loop
            console.print(
                f"[yellow]optimized.select_prompts failed ({_exc}); "
                "falling back to all tasks[/yellow]"
            )
            tasks_iter = all_tasks
    else:
        tasks_iter = task_batch["payload"]["tasks"]
    for task in tasks_iter:
        # Batched path: M rollouts share one GPU forward pass. 5-7x faster
        # than serial at M=8 on H100-class GPUs (see MiningEngine.generate_m_completions).
        completions.extend(
            engine.generate_m_completions(
                task=task,
                window_context=window_context,
                registry=registry,
                miner_id=miner_id,
                num_samples=samples_per_task,
            )
        )
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
            engine = _make_mining_engine({**cfg, "miner_id": miner_id})
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
    summary = status_summary(cfg, registry)
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


@app.command("metrics-exporter")
def metrics_exporter_command(
    bind: Annotated[str | None, typer.Option("--bind")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
) -> None:
    cfg = _cfg()
    registry = _registry(cfg)
    chain = _chain(cfg)
    metrics_bind = bind or str(cfg["metrics_bind"])
    metrics_port = port or int(cfg["metrics_port"])
    console.print(f"serving metrics on [bold]{metrics_bind}:{metrics_port}[/bold]")
    serve_metrics(
        bind=metrics_bind,
        port=metrics_port,
        cache=MetricsCache(cfg=cfg, registry=registry, chain=chain),
    )


@app.command("run-miner")
def run_miner(
    once: Annotated[bool, typer.Option("--once")] = False,
    poll_interval: Annotated[int | None, typer.Option("--poll-interval")] = None,
    resume_from: Annotated[
        str | None,
        typer.Option(
            "--resume-from",
            envvar="RELIQUARY_INFERENCE_RESUME_FROM",
            help=(
                "Boot model bundle from sha:<hex> (HF revision of cfg model_ref) "
                "or path:<dir> (local snapshot). Useful for pinning to a known-good "
                "checkpoint or rolling back without changing model_ref. "
                "Parallel-work credit: romain13190/reliquary@1801544."
            ),
        ),
    ] = None,
) -> None:
    cfg = _cfg()
    if resume_from is not None:
        _apply_resume_from(cfg, resume_from)
    interval = int(cfg["poll_interval"]) if poll_interval is None else poll_interval
    registry = _registry(cfg)
    chain = _chain(cfg)
    processed: set[tuple[int, str]] = set()

    # /health HTTP server in daemon thread (Phase 3.1). No-op when
    # RELIQUARY_INFERENCE_HEALTH_PORT is unset.
    health_holder = _start_health_server(cfg)
    miner_started_at = time.time()

    # Policy-consumer integration: off by default. When enabled, the miner
    # polls Forge-side PolicyCommitments each loop iteration and hot-swaps
    # its model weights via ReloadingPolicyApplier at the window boundary.
    # See reliquary_inference.shared.policy_applier for the delta-apply
    # semantics. Operators flip this on only after wiring the matching
    # consumer into run_validator — otherwise miner's proofs will fail
    # verification because the validator's model copy hasn't been updated.
    policy_consumer_hook = _build_miner_policy_consumer_hook(cfg)

    while True:
        chain_ok = False
        window_verified = False
        try:
            window_context = chain.get_window_context(cfg=cfg).as_dict()
            chain_ok = True
            ledger_window = int(window_context["window_id"])

            if policy_consumer_hook is not None:
                policy_consumer_hook(ledger_window=ledger_window)

            key = (ledger_window, str(cfg["miner_id"]))
            if key not in processed:
                mined = _mine_single_window(cfg, registry, window_context, str(cfg["miner_id"]))
                processed.add(key)
                window_verified = True
                console.print(f"mined {mined} completions for window {window_context['window_id']}")
            if once:
                _update_health(
                    health_holder,
                    started_at=miner_started_at,
                    chain_ok=chain_ok,
                    window_verified=window_verified,
                    model_loaded=True,
                )
                return
        except Exception as exc:
            if once:
                raise
            console.print(f"[yellow]miner loop error: {exc}[/yellow]")
        finally:
            _update_health(
                health_holder,
                started_at=miner_started_at,
                chain_ok=chain_ok,
                window_verified=window_verified,
                model_loaded=chain_ok,
            )
        time.sleep(interval)


def _build_miner_policy_consumer_hook(cfg: dict):
    """Build a hot-swap hook that runs the Ledger-side PolicyConsumer once
    per miner loop iteration, or ``None`` if the feature is disabled.

    Gated by ``RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED=true``. Requires
    the shared HMAC signing secret (``RELIQUARY_INFERENCE_POLICY_AUTHORITY_HOTKEY``
    + ``RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET``) and a
    ``RELIQUARY_INFERENCE_TRAINING_NETUID`` (defaults to the inference
    netuid since Forge ships as a v2 update inside the same subnet).

    Each invocation:
      1. Constructs a fresh MiningEngine if we don't have one yet
         (so the applier has a model to mutate).
      2. consumer.poll_once(ledger_window=...) → applied|ready|idle|rejected.
      3. On ``applied``, the delta has already been mutated into the
         engine's model in place; subsequent ``_mine_single_window``
         calls will generate with the updated weights.
      4. On ``rejected``, the error is logged and the current weights
         remain untouched.
    """
    import os

    if os.environ.get("RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    # Lazy imports so disabling the flag doesn't pay any import cost.
    from .miner.engine import MiningEngine
    from .shared.policy_applier import (
        ReloadingPolicyApplier,
        bundle_aware_delta_loader,
    )
    from .shared.policy_consumer import PolicyConsumer, default_smoke_runner
    from .validator.rollout_bundle import make_hmac_verifier

    # Cross-repo import — reliquary (Forge) package optional. If missing,
    # the consumer can't fetch delta bundles, so disable gracefully.
    try:
        from reliquary.training.checkpoint_storage import fetch_bundle as _fetch_bundle
    except ImportError:
        console.print(
            "[yellow]policy_consumer: reliquary.training.checkpoint_storage "
            "not importable — disabling hot-swap hook[/yellow]"
        )
        return None

    authority_hotkey = str(
        os.environ.get("RELIQUARY_INFERENCE_POLICY_AUTHORITY_HOTKEY", "")
    ).strip()
    authority_secret = str(
        os.environ.get("RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET", "")
    ).strip()
    if not authority_hotkey or not authority_secret:
        console.print(
            "[yellow]policy_consumer: authority hotkey / secret env unset — "
            "disabling hot-swap hook[/yellow]"
        )
        return None

    training_netuid = int(
        os.environ.get("RELIQUARY_INFERENCE_TRAINING_NETUID", cfg["netuid"])
    )

    # Shared R2 backend — reuse the one the miner already uses for artifacts.
    registry = _registry(cfg)
    raw_store = getattr(registry, "store", None)
    if raw_store is None:
        console.print(
            "[yellow]policy_consumer: registry has no underlying store — "
            "disabling hot-swap hook[/yellow]"
        )
        return None
    store = _StorageBackendShim(raw_store)

    # Engine is built lazily on first call — the consumer may need to sign
    # and the HMAC check + attestation read are cheap even if we haven't
    # loaded the model yet.
    state = {"engine": None, "consumer": None}

    def _ensure_engine():
        if state["engine"] is None:
            state["engine"] = _make_mining_engine(cfg)
        return state["engine"]

    def _ensure_consumer():
        if state["consumer"] is None:
            verifier = make_hmac_verifier({authority_hotkey: authority_secret})
            loader = bundle_aware_delta_loader(_fetch_bundle, lambda: store)
            engine = _ensure_engine()
            applier = ReloadingPolicyApplier(engine)
            state["consumer"] = PolicyConsumer(
                backend=store,
                verifier=verifier,
                inference_netuid=int(cfg["netuid"]),
                training_netuid=training_netuid,
                delta_loader=loader,
                smoke_runner=default_smoke_runner,
                applier=applier,
            )
        return state["consumer"]

    def _hook(*, ledger_window: int) -> None:
        consumer = _ensure_consumer()
        try:
            outcome = consumer.poll_once(ledger_window=ledger_window)
        except Exception as exc:
            console.print(f"[yellow]policy_consumer error: {exc}[/yellow]")
            return
        if outcome.state == "applied":
            console.print(
                f"policy_consumer applied run_id="
                f"{outcome.attestation.checkpoint_run_id if outcome.attestation else '?'} "
                f"at ledger_window={ledger_window} "
                f"merkle={outcome.delta.merkle_root_hex[:12] if outcome.delta else '?'}"
            )
        elif outcome.state == "rejected":
            console.print(
                f"[yellow]policy_consumer rejected: {outcome.reason}[/yellow]"
            )

    return _hook


@app.command("run-validator")
def run_validator(
    once: Annotated[bool, typer.Option("--once")] = False,
    poll_interval: Annotated[int | None, typer.Option("--poll-interval")] = None,
    resume_from: Annotated[
        str | None,
        typer.Option(
            "--resume-from",
            envvar="RELIQUARY_INFERENCE_RESUME_FROM",
            help=(
                "Boot model bundle from sha:<hex> (HF revision of cfg model_ref) "
                "or path:<dir> (local snapshot). Lets a validator pin to a known-good "
                "checkpoint without changing model_ref. "
                "Parallel-work credit: romain13190/reliquary@1801544."
            ),
        ),
    ] = None,
) -> None:
    cfg = _cfg()
    if resume_from is not None:
        _apply_resume_from(cfg, resume_from)
    interval = int(cfg["poll_interval"]) if poll_interval is None else poll_interval
    registry = _registry(cfg)
    chain = _chain(cfg)
    processed: set[int] = set()

    # /health HTTP server in daemon thread (Phase 3.1). No-op when
    # RELIQUARY_INFERENCE_HEALTH_PORT is unset.
    health_holder = _start_health_server(cfg)
    validator_started_at = time.time()

    # Matches the miner-side hook in run_miner: when the flag is on, every
    # loop iteration polls for new PolicyCommitments + applies the delta
    # to the validator's cached model bundles. Without this, miner weights
    # would drift from validator weights and every verdict would
    # hard-fail proof verification.
    policy_consumer_hook = _build_validator_policy_consumer_hook(cfg)

    # Backfill horizon: how many historical windows the validator will
    # scan each loop looking for unprocessed completion bundles. With
    # 6-min windows this covers the last ~1 hour. Prevents windows
    # mined while the validator was offline or slow from being orphaned
    # forever — the forge GRPO cycle needs scorecards for EVERY mined
    # window where at least 1 rollout is in zone.
    backfill_horizon = int(cfg.get("validator_backfill_horizon_windows", 10))
    while True:
        chain_ok = False
        any_window_verified = False
        try:
            window_context = chain.get_window_context(cfg=cfg).as_dict()
            chain_ok = True
            current_window = int(window_context["window_id"])

            # Early health update: mark chain reachable as soon as the
            # window context comes back. Without this, the backfill loop
            # below can take 5-30 min per outer iteration before the
            # finally block fires — leaving /health "never connected"
            # for the duration even though chain is healthy.
            _update_health(
                health_holder,
                started_at=validator_started_at,
                chain_ok=True,
                window_verified=any_window_verified,
                model_loaded=True,
            )

            if policy_consumer_hook is not None:
                policy_consumer_hook(ledger_window=current_window)

            # Backfill loop: start from the oldest unprocessed window within
            # the horizon and walk forward. Stops at current_window.
            # WINDOW_LENGTH on Bittensor testnet is 30 blocks.
            WINDOW_STRIDE = int(cfg.get("window_stride_blocks", 30))
            oldest = current_window - backfill_horizon * WINDOW_STRIDE
            candidate = max(oldest, 0)
            while candidate <= current_window:
                if candidate not in processed:
                    try:
                        bundles = registry.list_completion_bundles(window_id=candidate)
                    except Exception as exc:
                        console.print(
                            f"[yellow]list_completion_bundles({candidate}) error: {exc}[/yellow]"
                        )
                        bundles = []
                    if bundles:
                        window_ctx = dict(window_context)
                        window_ctx["window_id"] = candidate
                        try:
                            scorecard, _ = _validate_and_score_single_window(
                                cfg, registry, chain, window_ctx,
                            )
                            processed.add(candidate)
                            any_window_verified = True
                            console.print(
                                f"published weights for window {candidate}: "
                                f"{scorecard['payload']['weights']}"
                            )
                            # Mid-loop health update so /health flips to
                            # ok within seconds of each successful window
                            # rather than waiting for the full backfill.
                            _update_health(
                                health_holder,
                                started_at=validator_started_at,
                                chain_ok=True,
                                window_verified=True,
                                model_loaded=True,
                            )
                        except Exception as exc:
                            console.print(
                                f"[yellow]validate window={candidate} error: {exc}[/yellow]"
                            )
                candidate += WINDOW_STRIDE
            if once:
                _update_health(
                    health_holder,
                    started_at=validator_started_at,
                    chain_ok=chain_ok,
                    window_verified=any_window_verified,
                    model_loaded=True,
                )
                return
        except Exception as exc:
            if once:
                raise
            console.print(f"[yellow]validator loop error: {exc}[/yellow]")
        finally:
            _update_health(
                health_holder,
                started_at=validator_started_at,
                chain_ok=chain_ok,
                window_verified=any_window_verified,
                model_loaded=chain_ok,
            )
        time.sleep(interval)


def _build_validator_policy_consumer_hook(cfg: dict):
    """Validator-side counterpart of _build_miner_policy_consumer_hook.

    Uses the same gating env var (``RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED``)
    plus the same POLICY_AUTHORITY_HOTKEY + _SECRET as the miner. Applies
    deltas to every cached model bundle in :mod:`reliquary_inference.shared.modeling`
    so subsequent ``verify_completion`` calls use the updated weights
    without a process restart.

    Both miner and validator should be flag-enabled simultaneously. Turning
    it on for one but not the other causes proof-verification failures
    because the two models diverge.
    """
    import os

    if os.environ.get("RELIQUARY_INFERENCE_POLICY_CONSUMER_ENABLED", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    from .shared import modeling as _modeling  # for apply_delta_to_cached_bundles
    from .shared.policy_consumer import (
        LoadedDelta,
        PolicyConsumer,
        default_smoke_runner,
    )
    from .validator.rollout_bundle import make_hmac_verifier

    try:
        from reliquary.training.checkpoint_storage import fetch_bundle as _fetch_bundle
    except ImportError:
        console.print(
            "[yellow]validator policy_consumer: reliquary.training.checkpoint_storage "
            "not importable — disabling hot-swap hook[/yellow]"
        )
        return None

    authority_hotkey = str(
        os.environ.get("RELIQUARY_INFERENCE_POLICY_AUTHORITY_HOTKEY", "")
    ).strip()
    authority_secret = str(
        os.environ.get("RELIQUARY_INFERENCE_POLICY_AUTHORITY_SECRET", "")
    ).strip()
    if not authority_hotkey or not authority_secret:
        console.print(
            "[yellow]validator policy_consumer: authority hotkey / secret env unset — "
            "disabling hot-swap hook[/yellow]"
        )
        return None

    training_netuid = int(
        os.environ.get("RELIQUARY_INFERENCE_TRAINING_NETUID", cfg["netuid"])
    )

    registry = _registry(cfg)
    raw_store = getattr(registry, "store", None)
    if raw_store is None:
        console.print(
            "[yellow]validator policy_consumer: registry has no underlying store — "
            "disabling hot-swap hook[/yellow]"
        )
        return None
    store = _StorageBackendShim(raw_store)

    # The validator doesn't hold a single engine — the verifier loads bundles
    # lazily and now caches them in modeling._BUNDLE_CACHE. The applier
    # iterates every cached bundle and mutates all of them.
    class _GlobalCacheApplier:
        def __init__(self):
            self.metrics_counters = {"reliquary_policy_applier_apply_total": 0}

        def __call__(self, delta: LoadedDelta) -> None:
            bundle = delta.extra.get("bundle") if delta.extra else None
            if bundle is None:
                raise RuntimeError("validator applier requires delta.extra['bundle']")
            mutated = _modeling.apply_delta_to_cached_bundles(bundle)
            self.metrics_counters["reliquary_policy_applier_apply_total"] += 1
            console.print(
                f"validator policy_consumer: delta applied to {mutated} cached "
                f"model bundle(s) run_id={delta.run_id} window={delta.window_id}"
            )

    # Delta loader: same bundle_aware pattern as miner.
    from .shared.policy_applier import bundle_aware_delta_loader

    verifier = make_hmac_verifier({authority_hotkey: authority_secret})
    loader = bundle_aware_delta_loader(_fetch_bundle, lambda: store)
    applier = _GlobalCacheApplier()

    consumer = PolicyConsumer(
        backend=store,
        verifier=verifier,
        inference_netuid=int(cfg["netuid"]),
        training_netuid=training_netuid,
        delta_loader=loader,
        smoke_runner=default_smoke_runner,
        applier=applier,
    )

    def _hook(*, ledger_window: int) -> None:
        try:
            outcome = consumer.poll_once(ledger_window=ledger_window)
        except Exception as exc:
            console.print(f"[yellow]validator policy_consumer error: {exc}[/yellow]")
            return
        if outcome.state == "applied":
            console.print(
                f"validator policy_consumer applied run_id="
                f"{outcome.attestation.checkpoint_run_id if outcome.attestation else '?'} "
                f"at ledger_window={ledger_window}"
            )
        elif outcome.state == "rejected":
            console.print(
                f"[yellow]validator policy_consumer rejected: {outcome.reason}[/yellow]"
            )

    return _hook

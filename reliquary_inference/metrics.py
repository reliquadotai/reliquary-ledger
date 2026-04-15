from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .status import read_audit_index, status_summary

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _window_totals(audit_payload: dict[str, Any] | None, *, limit: int) -> dict[str, float]:
    windows = list((audit_payload or {}).get("windows", []))[:limit]
    totals = {
        "submitted": 0.0,
        "accepted": 0.0,
        "hard_failed": 0.0,
        "soft_failed": 0.0,
        "publish_success": 0.0,
        "publish_failure": 0.0,
        "reasoning_eval_count": 0.0,
        "reasoning_correct_total": 0.0,
        "reasoning_format_ok_total": 0.0,
        "reasoning_policy_compliance_total": 0.0,
    }
    for window in windows:
        verification_totals = window.get("verification_totals", {})
        window_metrics = window.get("window_metrics", {})
        totals["submitted"] += float(verification_totals.get("submitted", 0))
        totals["accepted"] += float(verification_totals.get("accepted", 0))
        totals["hard_failed"] += float(verification_totals.get("hard_failed", 0))
        totals["soft_failed"] += float(verification_totals.get("soft_failed", 0))
        totals["reasoning_eval_count"] += float(window_metrics.get("reasoning_eval_count", 0))
        totals["reasoning_correct_total"] += float(window_metrics.get("reasoning_correct_total", 0.0))
        totals["reasoning_format_ok_total"] += float(window_metrics.get("reasoning_format_ok_total", 0.0))
        totals["reasoning_policy_compliance_total"] += float(window_metrics.get("reasoning_policy_compliance_total", 0.0))
        publish_result = window.get("chain_publish_result") or {}
        if publish_result.get("success") is True:
            totals["publish_success"] += 1.0
        elif publish_result.get("success") is False:
            totals["publish_failure"] += 1.0
    return totals


def _task_source_totals(audit_payload: dict[str, Any] | None, *, limit: int) -> dict[str, dict[str, float]]:
    windows = list((audit_payload or {}).get("windows", []))[:limit]
    totals: dict[str, dict[str, float]] = {}
    for window in windows:
        task_source = str(window.get("task_source", "unknown"))
        verification_totals = window.get("verification_totals", {})
        bucket = totals.setdefault(task_source, {"submitted": 0.0, "accepted": 0.0})
        bucket["submitted"] += float(verification_totals.get("submitted", 0))
        bucket["accepted"] += float(verification_totals.get("accepted", 0))
    return totals


def _read_wallet_public(cfg: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(str(cfg.get("wallet_public_file", "")).strip())
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _configured_hotkeys(cfg: dict[str, Any]) -> dict[str, str]:
    wallet_payload = _read_wallet_public(cfg) or {}
    miner = str(cfg.get("miner_ss58", "")).strip() or str(wallet_payload.get("miner_hotkey_ss58", "")).strip()
    validator = (
        str(cfg.get("validator_ss58", "")).strip()
        or str(wallet_payload.get("validator_hotkey_ss58", "")).strip()
    )
    hotkeys: dict[str, str] = {}
    if miner:
        hotkeys["miner"] = miner
    if validator:
        hotkeys["validator"] = validator
    return hotkeys


def _chain_state(
    *,
    cfg: dict[str, Any],
    chain,
    previous_state: dict[str, Any] | None,
    last_success_at: float | None,
    now: float,
) -> tuple[dict[str, Any], float | None]:
    default_state = {
        "chain_scrape_success": 0.0,
        "metagraph_size": float((previous_state or {}).get("metagraph_size", 0.0)),
        "subnet_visible": float((previous_state or {}).get("subnet_visible", 0.0)),
        "current_block": float((previous_state or {}).get("current_block", -1.0)),
        "chain_window_id": float((previous_state or {}).get("chain_window_id", -1.0)),
        "hotkeys": dict((previous_state or {}).get("hotkeys", {})),
    }
    if str(cfg.get("network")) in {"local", "mock"}:
        default_state.update(
            {
                "chain_scrape_success": 1.0,
                "subnet_visible": 1.0,
                "current_block": -1.0,
                "chain_window_id": float(int(now)),
            }
        )
        return default_state, now
    try:
        current_block = float(chain.get_current_block())
        context = chain.get_window_context(cfg=cfg).as_dict()
        metagraph = chain.get_metagraph()
        metagraph_size = float(len(getattr(metagraph, "hotkeys", [])))
        hotkey_state = chain.describe_hotkeys(_configured_hotkeys(cfg))
        default_state.update(
            {
                "chain_scrape_success": 1.0,
                "metagraph_size": metagraph_size,
                "subnet_visible": 1.0 if metagraph_size > 0 else 0.0,
                "current_block": current_block,
                "chain_window_id": float(int(context["window_id"])),
                "hotkeys": hotkey_state,
            }
        )
        return default_state, now
    except Exception:
        return default_state, last_success_at


def collect_metrics_snapshot(
    *,
    cfg: dict[str, Any],
    registry,
    chain,
    previous_chain_state: dict[str, Any] | None = None,
    last_successful_chain_scrape_at: float | None = None,
    now: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float | None]:
    now = time.time() if now is None else now
    summary = status_summary(cfg, registry)
    audit_payload = read_audit_index(cfg, registry)
    rolling_totals = _window_totals(audit_payload, limit=int(cfg.get("metrics_window_count", 10)))
    task_source_totals = _task_source_totals(audit_payload, limit=int(cfg.get("metrics_window_count", 10)))
    chain_state, last_success = _chain_state(
        cfg=cfg,
        chain=chain,
        previous_state=previous_chain_state,
        last_success_at=last_successful_chain_scrape_at,
        now=now,
    )
    last_audit_generated_at = _parse_timestamp((audit_payload or {}).get("generated_at"))
    latest_publish = summary.get("latest_weight_publication", {})
    submitted = rolling_totals["submitted"]
    acceptance_rate = (rolling_totals["accepted"] / submitted) if submitted else 0.0
    reasoning_eval_count = rolling_totals["reasoning_eval_count"]
    reasoning_correct_rate = (
        rolling_totals["reasoning_correct_total"] / reasoning_eval_count if reasoning_eval_count else 0.0
    )
    reasoning_format_rate = (
        rolling_totals["reasoning_format_ok_total"] / reasoning_eval_count if reasoning_eval_count else 0.0
    )
    reasoning_policy_compliance = (
        rolling_totals["reasoning_policy_compliance_total"] / reasoning_eval_count if reasoning_eval_count else 0.0
    )
    snapshot = {
        "generated_at": now,
        "runtime": summary,
        "audit_generated_at": last_audit_generated_at,
        "audit_window_count": float((audit_payload or {}).get("window_count", 0)),
        "rolling_windows": float(int(cfg.get("metrics_window_count", 10))),
        "rolling_submitted_total": rolling_totals["submitted"],
        "rolling_accepted_total": rolling_totals["accepted"],
        "rolling_hard_failed_total": rolling_totals["hard_failed"],
        "rolling_soft_failed_total": rolling_totals["soft_failed"],
        "rolling_rejected_total": rolling_totals["hard_failed"] + rolling_totals["soft_failed"],
        "rolling_acceptance_rate": acceptance_rate,
        "rolling_reasoning_eval_count": reasoning_eval_count,
        "rolling_reasoning_correct_rate": reasoning_correct_rate,
        "rolling_reasoning_format_rate": reasoning_format_rate,
        "rolling_reasoning_policy_compliance": reasoning_policy_compliance,
        "task_source_totals": task_source_totals,
        "publish_success_total": rolling_totals["publish_success"],
        "publish_failure_total": rolling_totals["publish_failure"],
        "latest_window_mined": float(summary.get("latest_window_mined") or -1),
        "latest_importable_window": float(summary.get("latest_importable_window") or -1),
        "import_lag_windows": float(summary.get("import_lag_windows") or 0),
        "latest_weight_publication_window": float(latest_publish.get("window_id") or -1),
        "latest_weight_publication_success": 1.0 if latest_publish.get("success") is True else 0.0,
        "latest_weight_publication_uid_count": float(len(latest_publish.get("uids") or [])),
        "chain": chain_state,
        "chain_last_successful_scrape_at": last_success,
        "chain_scrape_age_seconds": (now - last_success) if last_success is not None else math.inf,
        "audit_publish_age_seconds": (now - last_audit_generated_at) if last_audit_generated_at is not None else math.inf,
    }
    return snapshot, chain_state, last_success


def _metric_lines(name: str, value: float, *, help_text: str, metric_type: str = "gauge", labels: dict[str, str] | None = None) -> list[str]:
    label_text = ""
    if labels:
        def _escape(value: str) -> str:
            return value.replace("\\", "\\\\").replace('"', '\\"')

        encoded = ",".join(
            f'{key}="{_escape(str(val))}"'
            for key, val in sorted(labels.items())
        )
        label_text = f"{{{encoded}}}"
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} {metric_type}",
        f"{name}{label_text} {value}",
    ]


def render_metrics(snapshot: dict[str, Any]) -> str:
    runtime = snapshot["runtime"]
    chain = snapshot["chain"]
    lines: list[str] = []
    lines.extend(
        _metric_lines(
            "reliquary_runtime_info",
            1.0,
            help_text="Runtime configuration labels for the active Reliquary node.",
            labels={
                "repo": "reliquary-inference",
                "network": str(runtime["network"]),
                "model_ref": str(runtime["model_ref"]),
                "task_source": str(runtime["task_source"]),
                "storage_backend": str(runtime["storage_backend"]),
                "bucket_mode": str(runtime["bucket_mode"]),
                "chain_endpoint_mode": str(runtime["chain_endpoint_mode"]),
            },
        )
    )
    simple_metrics = {
        "reliquary_latest_window_mined": (
            snapshot["latest_window_mined"],
            "Latest window observed with mined completions.",
        ),
        "reliquary_latest_weight_publication_window": (
            snapshot["latest_weight_publication_window"],
            "Latest window with a finalized weight publication.",
        ),
        "reliquary_latest_importable_window": (
            snapshot["latest_importable_window"],
            "Latest finalized window available for import by downstream training runtimes.",
        ),
        "reliquary_import_lag_windows": (
            snapshot["import_lag_windows"],
            "Difference between the latest mined window and the latest finalized importable window.",
        ),
        "reliquary_latest_weight_publication_success": (
            snapshot["latest_weight_publication_success"],
            "Whether the latest finalized weight publication succeeded.",
        ),
        "reliquary_latest_weight_publication_uid_count": (
            snapshot["latest_weight_publication_uid_count"],
            "UID count in the latest finalized weight publication.",
        ),
        "reliquary_rolling_submitted_total": (
            snapshot["rolling_submitted_total"],
            "Submitted completions across the rolling audit window.",
        ),
        "reliquary_rolling_accepted_total": (
            snapshot["rolling_accepted_total"],
            "Accepted completions across the rolling audit window.",
        ),
        "reliquary_rolling_rejected_total": (
            snapshot["rolling_rejected_total"],
            "Rejected completions across the rolling audit window.",
        ),
        "reliquary_rolling_hard_failed_total": (
            snapshot["rolling_hard_failed_total"],
            "Hard validation failures across the rolling audit window.",
        ),
        "reliquary_rolling_soft_failed_total": (
            snapshot["rolling_soft_failed_total"],
            "Soft validation failures across the rolling audit window.",
        ),
        "reliquary_rolling_acceptance_rate": (
            snapshot["rolling_acceptance_rate"],
            "Acceptance rate across the rolling audit window.",
        ),
        "reliquary_rolling_reasoning_correct_rate": (
            snapshot["rolling_reasoning_correct_rate"],
            "Reasoning correctness rate across recent finalized windows.",
        ),
        "reliquary_rolling_reasoning_format_rate": (
            snapshot["rolling_reasoning_format_rate"],
            "Reasoning final-answer format rate across recent finalized windows.",
        ),
        "reliquary_rolling_reasoning_policy_compliance": (
            snapshot["rolling_reasoning_policy_compliance"],
            "Mean reasoning policy-compliance score across recent finalized windows.",
        ),
        "reliquary_publish_success_total": (
            snapshot["publish_success_total"],
            "Successful publish events across the rolling audit window.",
        ),
        "reliquary_publish_failure_total": (
            snapshot["publish_failure_total"],
            "Failed publish events across the rolling audit window.",
        ),
        "reliquary_audit_window_count": (
            snapshot["audit_window_count"],
            "Window count present in the local audit index.",
        ),
        "reliquary_chain_scrape_success": (
            chain["chain_scrape_success"],
            "Whether the most recent chain scrape succeeded.",
        ),
        "reliquary_chain_scrape_age_seconds": (
            snapshot["chain_scrape_age_seconds"],
            "Seconds since the last successful chain scrape.",
        ),
        "reliquary_chain_current_block": (
            chain["current_block"],
            "Current chain block height observed by the exporter.",
        ),
        "reliquary_chain_window_id": (
            chain["chain_window_id"],
            "Current chain-derived window id observed by the exporter.",
        ),
        "reliquary_metagraph_size": (
            chain["metagraph_size"],
            "Metagraph size observed on the configured subnet.",
        ),
        "reliquary_subnet_visible": (
            chain["subnet_visible"],
            "Whether the configured subnet is visible to the exporter.",
        ),
        "reliquary_audit_publish_age_seconds": (
            snapshot["audit_publish_age_seconds"],
            "Seconds since the audit index was last generated.",
        ),
    }
    for metric_name, (value, help_text) in simple_metrics.items():
        lines.extend(_metric_lines(metric_name, float(value), help_text=help_text))
    for task_source, totals in sorted(snapshot["task_source_totals"].items()):
        lines.extend(
            _metric_lines(
                "reliquary_task_source_submitted_total",
                float(totals["submitted"]),
                help_text="Submitted completions across the rolling audit window by task source.",
                labels={"task_source": task_source},
            )
        )
        lines.extend(
            _metric_lines(
                "reliquary_task_source_accepted_total",
                float(totals["accepted"]),
                help_text="Accepted completions across the rolling audit window by task source.",
                labels={"task_source": task_source},
            )
        )
    if snapshot["audit_generated_at"] is not None:
        lines.extend(
            _metric_lines(
                "reliquary_audit_generated_timestamp_seconds",
                float(snapshot["audit_generated_at"]),
                help_text="Unix timestamp for the latest local audit index generation.",
            )
        )
    if snapshot["chain_last_successful_scrape_at"] is not None:
        lines.extend(
            _metric_lines(
                "reliquary_chain_last_successful_scrape_timestamp_seconds",
                float(snapshot["chain_last_successful_scrape_at"]),
                help_text="Unix timestamp of the last successful chain scrape.",
            )
        )
    for role, hotkey_state in sorted(chain.get("hotkeys", {}).items()):
        lines.extend(
            _metric_lines(
                "reliquary_hotkey_registered",
                1.0 if hotkey_state.get("registered") else 0.0,
                help_text="Whether the configured hotkey is registered on the configured subnet.",
                labels={"role": role, "hotkey": str(hotkey_state.get("hotkey", ""))},
            )
        )
        lines.extend(
            _metric_lines(
                "reliquary_hotkey_uid",
                float(hotkey_state.get("uid", -1)),
                help_text="UID for the configured hotkey on the configured subnet.",
                labels={"role": role, "hotkey": str(hotkey_state.get("hotkey", ""))},
            )
        )
    return "\n".join(lines) + "\n"


@dataclass
class MetricsCache:
    cfg: dict[str, Any]
    registry: Any
    chain: Any
    previous_chain_state: dict[str, Any] | None = None
    last_successful_chain_scrape_at: float | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_generated_at: float = 0.0

    def current(self) -> dict[str, Any]:
        now = time.time()
        refresh_interval = float(self.cfg.get("metrics_refresh_interval", 15))
        if self.snapshot is not None and (now - self.snapshot_generated_at) < refresh_interval:
            return self.snapshot
        snapshot, chain_state, last_success = collect_metrics_snapshot(
            cfg=self.cfg,
            registry=self.registry,
            chain=self.chain,
            previous_chain_state=self.previous_chain_state,
            last_successful_chain_scrape_at=self.last_successful_chain_scrape_at,
            now=now,
        )
        self.previous_chain_state = chain_state
        self.last_successful_chain_scrape_at = last_success
        self.snapshot = snapshot
        self.snapshot_generated_at = now
        return snapshot


def serve_metrics(*, bind: str, port: int, cache: MetricsCache) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in {"/metrics", "/metrics/"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = render_metrics(cache.current()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", PROMETHEUS_CONTENT_TYPE)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((bind, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()

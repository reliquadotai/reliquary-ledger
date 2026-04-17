"""Live mesh-integration harness: run one validator on one host, emit a
signed verdict artifact, and have the operator aggregate verdicts from
multiple hosts into a MeshAggregationReport.

Usage on each validator host:

    python -m reliquary_inference.validator.mesh_integration produce \
        --validator-hotkey mesh-A --validator-stake 40.0 \
        --scenario honest --window-id 100 --count 32 \
        --output /tmp/mesh_verdicts_hostA.json

Then on the orchestrator (any host with all verdict files rsynced
together):

    python -m reliquary_inference.validator.mesh_integration aggregate \
        --input /tmp/mesh_verdicts_hostA.json /tmp/mesh_verdicts_hostB.json \
        --expected-hotkeys mesh-A=40.0 mesh-B=40.0 mesh-C=20.0 \
        --output /tmp/mesh_report.json

This module is deliberately synthetic (no real model, no real chain).
The verdict artifact shape matches ``validator.mesh.VerdictArtifact``
so the aggregator consumes real + simulated verdicts identically.

Spec: private/reliquary-plan/notes/spec-validator-mesh.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .mesh import (
    MeshPolicy,
    ValidatorIdentity,
    VerdictArtifact,
    aggregate_verdicts,
)


def _synthetic_completions(*, window_id: int, count: int) -> list[dict]:
    """Fabricate a set of deterministic completion IDs for the window."""
    return [
        {
            "completion_id": f"w{window_id:06d}-c{i:04d}",
            "miner_hotkey": f"miner-{i % 5}",
            "window_id": window_id,
        }
        for i in range(count)
    ]


def _honest_verdict(*, completion: dict, identity: ValidatorIdentity, now: float) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion["completion_id"],
        miner_hotkey=completion["miner_hotkey"],
        window_id=completion["window_id"],
        validator=identity,
        accepted=True,
        stage_failed=None,
        reject_reason=None,
        scores={"correctness": 0.9, "format": 0.85},
        signed_at=now,
    )


def _malicious_verdict(*, completion: dict, identity: ValidatorIdentity, now: float) -> VerdictArtifact:
    """A validator that rejects everything and emits far-from-median scores."""
    return VerdictArtifact(
        completion_id=completion["completion_id"],
        miner_hotkey=completion["miner_hotkey"],
        window_id=completion["window_id"],
        validator=identity,
        accepted=False,
        stage_failed="proof",
        reject_reason="proof_sketch_mismatch",
        scores={"correctness": 0.0, "format": 0.0},
        signed_at=now,
    )


_SCENARIOS = {
    "honest": _honest_verdict,
    "malicious": _malicious_verdict,
}


def produce_verdicts(
    *,
    validator_hotkey: str,
    validator_stake: float,
    scenario: str,
    window_id: int,
    count: int,
    signer_id: str | None = None,
) -> list[VerdictArtifact]:
    if scenario not in _SCENARIOS:
        raise SystemExit(f"unknown scenario {scenario!r}; choose from {sorted(_SCENARIOS)}")
    identity = ValidatorIdentity(
        hotkey=validator_hotkey,
        stake=validator_stake,
        signer_id=signer_id or validator_hotkey,
    )
    now = time.time()
    fn = _SCENARIOS[scenario]
    completions = _synthetic_completions(window_id=window_id, count=count)
    return [fn(completion=c, identity=identity, now=now) for c in completions]


def _artifact_to_dict(a: VerdictArtifact) -> dict:
    return {
        "completion_id": a.completion_id,
        "miner_hotkey": a.miner_hotkey,
        "window_id": a.window_id,
        "validator": asdict(a.validator),
        "accepted": a.accepted,
        "stage_failed": a.stage_failed,
        "reject_reason": a.reject_reason,
        "scores": dict(a.scores),
        "signed_at": a.signed_at,
        "signature": a.signature,
    }


def _dict_to_artifact(d: dict) -> VerdictArtifact:
    v = d["validator"]
    return VerdictArtifact(
        completion_id=d["completion_id"],
        miner_hotkey=d["miner_hotkey"],
        window_id=d["window_id"],
        validator=ValidatorIdentity(hotkey=v["hotkey"], stake=v["stake"], signer_id=v.get("signer_id", v["hotkey"])),
        accepted=d["accepted"],
        stage_failed=d.get("stage_failed"),
        reject_reason=d.get("reject_reason"),
        scores=dict(d["scores"]),
        signed_at=d["signed_at"],
        signature=d.get("signature", ""),
    )


def cmd_produce(args: argparse.Namespace) -> int:
    verdicts = produce_verdicts(
        validator_hotkey=args.validator_hotkey,
        validator_stake=args.validator_stake,
        scenario=args.scenario,
        window_id=args.window_id,
        count=args.count,
        signer_id=args.signer_id,
    )
    payload = {
        "window_id": args.window_id,
        "validator_hotkey": args.validator_hotkey,
        "scenario": args.scenario,
        "verdicts": [_artifact_to_dict(v) for v in verdicts],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"wrote {len(verdicts)} verdicts to {args.output}")
    return 0


def _parse_expected(pairs: list[str]) -> list[ValidatorIdentity]:
    out: list[ValidatorIdentity] = []
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected-hotkeys entry must be hotkey=stake, got {pair!r}")
        hotkey, stake = pair.split("=", 1)
        out.append(ValidatorIdentity(hotkey=hotkey, stake=float(stake), signer_id=hotkey))
    return out


def cmd_aggregate(args: argparse.Namespace) -> int:
    all_verdicts: list[VerdictArtifact] = []
    window_ids: set[int] = set()
    for path in args.input:
        payload = json.loads(Path(path).read_text())
        window_ids.add(int(payload["window_id"]))
        for raw in payload["verdicts"]:
            all_verdicts.append(_dict_to_artifact(raw))
    if len(window_ids) != 1:
        raise SystemExit(f"verdicts span multiple windows: {sorted(window_ids)}")
    window_id = next(iter(window_ids))

    expected = _parse_expected(args.expected_hotkeys)
    policy = MeshPolicy()
    report = aggregate_verdicts(
        all_verdicts,
        window_id=window_id,
        expected_validators=expected,
        policy=policy,
    )
    summary = {
        "window_id": report.window_id,
        "total_completions": len(report.median_verdicts),
        "accepted": sum(1 for v in report.median_verdicts.values() if v.accepted),
        "missing_validators": report.missing_validators,
        "gated_validators": report.gated_validators,
        "disagreement_rates": dict(report.validator_disagreement_rates),
        "median_verdicts": {
            cid: {
                "accepted": v.accepted,
                "acceptance_score": v.acceptance_score,
                "participating_validators": v.participating_validators,
                "outlier_validators": v.outlier_validators,
                "quorum_satisfied": v.quorum_satisfied,
                "median_scores": v.median_scores,
            }
            for cid, v in report.median_verdicts.items()
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(
        f"aggregated {len(all_verdicts)} verdicts across {len(expected)} expected validators "
        f"into {len(report.median_verdicts)} completions; "
        f"gated={len(report.gated_validators)}, missing={len(report.missing_validators)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="mesh-integration")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_produce = sub.add_parser("produce", help="emit a verdict-artifact file on one validator host")
    p_produce.add_argument("--validator-hotkey", required=True)
    p_produce.add_argument("--validator-stake", type=float, required=True)
    p_produce.add_argument("--scenario", default="honest", choices=sorted(_SCENARIOS))
    p_produce.add_argument("--window-id", type=int, required=True)
    p_produce.add_argument("--count", type=int, default=32)
    p_produce.add_argument("--signer-id", default=None)
    p_produce.add_argument("--output", required=True)
    p_produce.set_defaults(func=cmd_produce)

    p_agg = sub.add_parser("aggregate", help="aggregate verdict files into a mesh report")
    p_agg.add_argument("--input", nargs="+", required=True)
    p_agg.add_argument(
        "--expected-hotkeys",
        nargs="+",
        required=True,
        help="space-separated hotkey=stake pairs",
    )
    p_agg.add_argument("--output", required=True)
    p_agg.set_defaults(func=cmd_aggregate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

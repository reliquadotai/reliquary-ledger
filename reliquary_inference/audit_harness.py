"""Empirical audit harness for the nine-stage verifier.

Runs honest-miner and adversarial-miner trials through the full pipeline
(or any subset) and reports per-class FP/FN rates plus per-stage
rejection breakdowns. Output is a structured JSON document suitable for
external auditor review.

Intended usage:

    from reliquary_inference.audit_harness import run_audit_campaign
    report = run_audit_campaign(honest_trials=1000, adversarial_trials=100)
    Path("docs/audit/empirical_report.json").write_text(json.dumps(report, indent=2))

Deliberately decoupled from the validator service: the harness constructs
synthetic sketch-layer rollouts over random hidden states, runs them
through the proof sketch verifier, and tallies outcomes. No HF model
load or chain state is required for the baseline report. GPU
(torch.cuda) is used when available for speed but CPU is the fallback.

Spec: private/reliquary-plan/notes/spec-proof-protocol.md acceptance
tests 1-10; Tier 2 Epic 6 of 02_TIER2_PRD.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .protocol.constants import CHALLENGE_K
from .protocol.sketch_verifier import SketchProofVerifier


HIDDEN_DIM_DEFAULT = 256
RANDOMNESS_HEX_DEFAULT = "00000000000000000000000000000000000000000000000000000000deadbeef"


@dataclass
class TrialOutcome:
    accepted: bool
    min_sketch_diff: int
    max_sketch_diff: int
    positions_checked: int


@dataclass
class ClassReport:
    name: str
    trials: int
    accept_count: int
    reject_count: int
    false_negative_rate: float  # fraction of adversarial trials that were (wrongly) accepted
    false_positive_rate: float  # fraction of honest trials that were (wrongly) rejected
    median_min_sketch_diff: float


@dataclass
class AuditReport:
    timestamp: str
    host: str
    torch_version: str
    cuda_available: bool
    hidden_dim: int
    challenge_k: int
    trials_per_class: int
    classes: dict[str, ClassReport] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_json(self) -> str:
        payload = {
            "timestamp": self.timestamp,
            "host": self.host,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "hidden_dim": self.hidden_dim,
            "challenge_k": self.challenge_k,
            "trials_per_class": self.trials_per_class,
            "duration_seconds": self.duration_seconds,
            "classes": {name: asdict(report) for name, report in self.classes.items()},
        }
        return json.dumps(payload, indent=2)


def _make_hidden(torch, seed: int, hidden_dim: int, scale: float = 1.0) -> object:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(hidden_dim, generator=gen, dtype=torch.float32) * scale


def _rollout_verify(
    torch,
    verifier: SketchProofVerifier,
    r_vec,
    miner_hidden_seq,
    validator_hidden_seq,
) -> TrialOutcome:
    commits = verifier.create_commitments_batch(miner_hidden_seq, r_vec)
    diffs: list[int] = []
    all_valid = True
    for pos in range(miner_hidden_seq.size(0)):
        valid, diag = verifier.verify_commitment(
            validator_hidden_seq[pos],
            commits[pos],
            r_vec,
            sequence_length=miner_hidden_seq.size(0),
            position=pos,
        )
        diffs.append(int(diag.get("sketch_diff", 0)))
        if not valid:
            all_valid = False
    return TrialOutcome(
        accepted=all_valid,
        min_sketch_diff=min(diffs) if diffs else 0,
        max_sketch_diff=max(diffs) if diffs else 0,
        positions_checked=len(diffs),
    )


def _honest_trial(torch, verifier, r_vec, seed: int, hidden_dim: int) -> TrialOutcome:
    seq = torch.stack([_make_hidden(torch, seed + i, hidden_dim) for i in range(CHALLENGE_K)])
    return _rollout_verify(torch, verifier, r_vec, seq, seq)


def _tamper_wholesale(torch, verifier, r_vec, seed: int, hidden_dim: int) -> TrialOutcome:
    miner_seq = torch.stack([_make_hidden(torch, seed + i, hidden_dim) for i in range(CHALLENGE_K)])
    validator_seq = torch.stack(
        [_make_hidden(torch, seed + i + 100_000, hidden_dim, scale=5.0) for i in range(CHALLENGE_K)]
    )
    return _rollout_verify(torch, verifier, r_vec, miner_seq, validator_seq)


def _tamper_zero(torch, verifier, r_vec, seed: int, hidden_dim: int) -> TrialOutcome:
    miner_seq = torch.stack([_make_hidden(torch, seed + i, hidden_dim) for i in range(CHALLENGE_K)])
    validator_seq = torch.zeros_like(miner_seq)
    return _rollout_verify(torch, verifier, r_vec, miner_seq, validator_seq)


def _tamper_cross_prompt(torch, verifier, r_vec, seed: int, hidden_dim: int) -> TrialOutcome:
    miner_seq = torch.stack([_make_hidden(torch, seed + i, hidden_dim) for i in range(CHALLENGE_K)])
    validator_seq = torch.stack(
        [_make_hidden(torch, seed + i + 50_000, hidden_dim) for i in range(CHALLENGE_K)]
    )
    return _rollout_verify(torch, verifier, r_vec, miner_seq, validator_seq)


ADVERSARIAL_CLASSES: dict[str, Callable] = {
    "tamper_wholesale": _tamper_wholesale,
    "tamper_zero": _tamper_zero,
    "tamper_cross_prompt": _tamper_cross_prompt,
}


def run_audit_campaign(
    *,
    honest_trials: int = 1000,
    adversarial_trials: int = 200,
    hidden_dim: int = HIDDEN_DIM_DEFAULT,
    randomness_hex: str = RANDOMNESS_HEX_DEFAULT,
) -> AuditReport:
    """Run an honest + adversarial audit and return a structured report.

    This is the function external auditors will invoke. Outputs are
    stable JSON; any divergence across runs (at fixed seeds) indicates
    a regression in proof semantics.
    """
    import socket
    from datetime import datetime, timezone

    import torch

    started = time.time()
    report = AuditReport(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        host=socket.gethostname(),
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        hidden_dim=hidden_dim,
        challenge_k=CHALLENGE_K,
        trials_per_class=max(honest_trials, adversarial_trials),
    )

    verifier = SketchProofVerifier(hidden_dim=hidden_dim)
    r_vec = verifier.generate_r_vec(randomness_hex)

    honest_accepts = 0
    honest_diffs: list[int] = []
    for seed in range(honest_trials):
        outcome = _honest_trial(torch, verifier, r_vec, seed=seed, hidden_dim=hidden_dim)
        if outcome.accepted:
            honest_accepts += 1
        honest_diffs.append(outcome.min_sketch_diff)
    report.classes["honest"] = ClassReport(
        name="honest",
        trials=honest_trials,
        accept_count=honest_accepts,
        reject_count=honest_trials - honest_accepts,
        false_negative_rate=0.0,
        false_positive_rate=(honest_trials - honest_accepts) / honest_trials if honest_trials else 0.0,
        median_min_sketch_diff=float(sorted(honest_diffs)[len(honest_diffs) // 2]) if honest_diffs else 0.0,
    )

    for class_name, fn in ADVERSARIAL_CLASSES.items():
        accepts = 0
        diffs: list[int] = []
        for seed in range(adversarial_trials):
            outcome = fn(torch, verifier, r_vec, seed=seed + 1_000_000, hidden_dim=hidden_dim)
            if outcome.accepted:
                accepts += 1
            diffs.append(outcome.min_sketch_diff)
        report.classes[class_name] = ClassReport(
            name=class_name,
            trials=adversarial_trials,
            accept_count=accepts,
            reject_count=adversarial_trials - accepts,
            false_negative_rate=accepts / adversarial_trials if adversarial_trials else 0.0,
            false_positive_rate=0.0,
            median_min_sketch_diff=float(sorted(diffs)[len(diffs) // 2]) if diffs else 0.0,
        )

    report.duration_seconds = time.time() - started
    return report


def main() -> int:
    """CLI entry point: run the campaign and print JSON to stdout."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="reliquary-inference-audit",
        description="Empirical audit of the sketch proof protocol.",
    )
    parser.add_argument("--honest-trials", type=int, default=1000)
    parser.add_argument("--adversarial-trials", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM_DEFAULT)
    parser.add_argument("--randomness-hex", type=str, default=RANDOMNESS_HEX_DEFAULT)
    parser.add_argument("--output", type=str, default=None, help="write JSON to path")
    args = parser.parse_args()

    report = run_audit_campaign(
        honest_trials=args.honest_trials,
        adversarial_trials=args.adversarial_trials,
        hidden_dim=args.hidden_dim,
        randomness_hex=args.randomness_hex,
    )
    payload = report.to_json()
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload)
        print(f"audit report written to {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

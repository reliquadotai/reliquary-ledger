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
tests 1-10.
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


def _checkpoint_write(path: "Path", report: AuditReport) -> None:  # noqa: F821
    """Atomically persist partial audit state."""
    import os
    import tempfile

    text = report.to_json()
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".audit-", dir=str(directory))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_load(path: "Path") -> AuditReport | None:  # noqa: F821
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = AuditReport(
        timestamp=payload.get("timestamp", ""),
        host=payload.get("host", ""),
        torch_version=payload.get("torch_version", ""),
        cuda_available=bool(payload.get("cuda_available", False)),
        hidden_dim=int(payload.get("hidden_dim", HIDDEN_DIM_DEFAULT)),
        challenge_k=int(payload.get("challenge_k", CHALLENGE_K)),
        trials_per_class=int(payload.get("trials_per_class", 0)),
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
    )
    for name, class_dict in (payload.get("classes") or {}).items():
        report.classes[name] = ClassReport(
            name=class_dict["name"],
            trials=int(class_dict["trials"]),
            accept_count=int(class_dict["accept_count"]),
            reject_count=int(class_dict["reject_count"]),
            false_negative_rate=float(class_dict["false_negative_rate"]),
            false_positive_rate=float(class_dict["false_positive_rate"]),
            median_min_sketch_diff=float(class_dict["median_min_sketch_diff"]),
        )
    return report


def run_audit_campaign(
    *,
    honest_trials: int = 1000,
    adversarial_trials: int = 200,
    hidden_dim: int = HIDDEN_DIM_DEFAULT,
    randomness_hex: str = RANDOMNESS_HEX_DEFAULT,
    progress_every: int = 0,
    progress_callback: Callable[[str, int, int], None] | None = None,
    checkpoint_path: "Path | None" = None,  # noqa: F821
    resume: bool = False,
) -> AuditReport:
    """Run an honest + adversarial audit and return a structured report.

    This is the function external auditors will invoke. Outputs are
    stable JSON; any divergence across runs (at fixed seeds) indicates
    a regression in proof semantics.

    Args:
        honest_trials: number of honest trials to run (per-class cap).
        adversarial_trials: number of trials per adversarial class.
        hidden_dim: sketch-layer hidden dim size.
        randomness_hex: sketch randomness seed (hex).
        progress_every: if > 0, print "[honest] 1000/10000" every N trials.
        progress_callback: optional ``(class_name, completed, total)`` hook
            called at the same cadence as progress_every.
        checkpoint_path: if set, the partial report is persisted after each
            class completes, so a crashed 100K run can resume.
        resume: if True and checkpoint_path exists, skip classes already
            present in the checkpoint.
    """
    import socket
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    import torch

    started = time.time()
    report: AuditReport | None = None
    if resume and checkpoint_path is not None:
        report = _checkpoint_load(_Path(checkpoint_path))
    if report is None:
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

    def _notify(class_name: str, completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(class_name, completed, total)
        if progress_every and completed % progress_every == 0:
            print(f"[{class_name}] {completed}/{total}", flush=True)

    if "honest" not in report.classes:
        honest_accepts = 0
        honest_diffs: list[int] = []
        for seed in range(honest_trials):
            outcome = _honest_trial(torch, verifier, r_vec, seed=seed, hidden_dim=hidden_dim)
            if outcome.accepted:
                honest_accepts += 1
            honest_diffs.append(outcome.min_sketch_diff)
            _notify("honest", seed + 1, honest_trials)
        report.classes["honest"] = ClassReport(
            name="honest",
            trials=honest_trials,
            accept_count=honest_accepts,
            reject_count=honest_trials - honest_accepts,
            false_negative_rate=0.0,
            false_positive_rate=(honest_trials - honest_accepts) / honest_trials if honest_trials else 0.0,
            median_min_sketch_diff=float(sorted(honest_diffs)[len(honest_diffs) // 2]) if honest_diffs else 0.0,
        )
        if checkpoint_path is not None:
            _checkpoint_write(_Path(checkpoint_path), report)

    for class_name, fn in ADVERSARIAL_CLASSES.items():
        if class_name in report.classes:
            continue
        accepts = 0
        diffs: list[int] = []
        for seed in range(adversarial_trials):
            outcome = fn(torch, verifier, r_vec, seed=seed + 1_000_000, hidden_dim=hidden_dim)
            if outcome.accepted:
                accepts += 1
            diffs.append(outcome.min_sketch_diff)
            _notify(class_name, seed + 1, adversarial_trials)
        report.classes[class_name] = ClassReport(
            name=class_name,
            trials=adversarial_trials,
            accept_count=accepts,
            reject_count=adversarial_trials - accepts,
            false_negative_rate=accepts / adversarial_trials if adversarial_trials else 0.0,
            false_positive_rate=0.0,
            median_min_sketch_diff=float(sorted(diffs)[len(diffs) // 2]) if diffs else 0.0,
        )
        if checkpoint_path is not None:
            _checkpoint_write(_Path(checkpoint_path), report)

    report.duration_seconds = time.time() - started
    if checkpoint_path is not None:
        _checkpoint_write(_Path(checkpoint_path), report)
    return report


def main() -> int:
    """CLI entry point: run the campaign and print JSON to stdout."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="reliquary-inference-audit",
        description="Empirical audit of the sketch proof protocol.",
    )
    parser.add_argument("--honest-trials", type=int, default=1000)
    parser.add_argument("--adversarial-trials", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM_DEFAULT)
    parser.add_argument("--randomness-hex", type=str, default=RANDOMNESS_HEX_DEFAULT)
    parser.add_argument("--output", type=str, default=None, help="write JSON to path")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress every N trials (per class); 0 disables",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="persist partial report after each class completes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="if --checkpoint-path exists, skip already-completed classes",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    report = run_audit_campaign(
        honest_trials=args.honest_trials,
        adversarial_trials=args.adversarial_trials,
        hidden_dim=args.hidden_dim,
        randomness_hex=args.randomness_hex,
        progress_every=args.progress_every,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )
    payload = report.to_json()
    if args.output:
        Path(args.output).write_text(payload)
        print(f"audit report written to {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Stake-weighted verdict aggregation across a validator mesh.

Pure-compute core of the distributed validator mesh: takes per-validator
verdict artifacts for a window, produces median verdicts + outlier reports.
All I/O (R2 upload/download, signature verification, metagraph reads) lives
in the caller — this module is deterministic and chain-free.

Spec: private/reliquary-plan/notes/spec-validator-mesh.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from ..protocol.constants import (
    MESH_MIN_QUORUM_STAKE_FRACTION,
    MESH_OUTLIER_RATE_GATE,
    MESH_OUTLIER_THRESHOLD,
    MESH_STAKE_CAP_FRACTION,
)


@dataclass(frozen=True)
class ValidatorIdentity:
    hotkey: str
    stake: float
    signer_id: str = ""


@dataclass(frozen=True)
class VerdictArtifact:
    completion_id: str
    miner_hotkey: str
    window_id: int
    validator: ValidatorIdentity
    accepted: bool
    stage_failed: str | None
    reject_reason: str | None
    scores: dict[str, float]
    signed_at: float
    signature: str = ""


@dataclass
class MedianVerdict:
    completion_id: str
    accepted: bool
    acceptance_score: float
    median_scores: dict[str, float]
    participating_validators: list[str]
    outlier_validators: list[str]
    quorum_satisfied: bool


@dataclass
class MeshAggregationReport:
    window_id: int
    median_verdicts: dict[str, MedianVerdict] = field(default_factory=dict)
    validator_disagreement_rates: dict[str, float] = field(default_factory=dict)
    missing_validators: list[str] = field(default_factory=list)
    gated_validators: list[str] = field(default_factory=list)


@dataclass
class MeshPolicy:
    stake_cap_fraction: float = MESH_STAKE_CAP_FRACTION
    min_quorum_stake_fraction: float = MESH_MIN_QUORUM_STAKE_FRACTION
    outlier_threshold: float = MESH_OUTLIER_THRESHOLD
    outlier_rate_gate: float = MESH_OUTLIER_RATE_GATE

    def __post_init__(self) -> None:
        for name in ("stake_cap_fraction", "min_quorum_stake_fraction"):
            value = getattr(self, name)
            if not (0 < value <= 1):
                raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
        if self.outlier_threshold < 0:
            raise ValueError("outlier_threshold must be non-negative")
        if not (0 <= self.outlier_rate_gate <= 1):
            raise ValueError("outlier_rate_gate must lie in [0, 1]")


def stake_weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Return the stake-weighted median value.

    ``pairs`` is a list of ``(value, weight)`` entries with non-negative
    weights. Ties are broken by returning the lower bucket value
    (conservative — picks the median at the boundary rather than
    interpolating).
    """
    positive = [(v, w) for v, w in pairs if w > 0]
    if not positive:
        raise ValueError("stake_weighted_median requires at least one positive-weight pair")
    positive.sort(key=lambda vw: vw[0])
    total = sum(w for _, w in positive)
    half = total / 2.0
    cumulative = 0.0
    for value, weight in positive:
        cumulative += weight
        if cumulative >= half:
            return value
    return positive[-1][0]


def _dedupe_latest_per_validator(
    verdicts: list[VerdictArtifact],
) -> list[VerdictArtifact]:
    """If a validator submits more than one verdict for a completion, keep
    the one with the latest ``signed_at``."""
    latest: dict[str, VerdictArtifact] = {}
    for verdict in verdicts:
        key = verdict.validator.hotkey
        current = latest.get(key)
        if current is None or verdict.signed_at > current.signed_at:
            latest[key] = verdict
    return list(latest.values())


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def aggregate_verdicts(
    artifacts: Iterable[VerdictArtifact],
    *,
    window_id: int,
    expected_validators: Iterable[ValidatorIdentity],
    policy: MeshPolicy | None = None,
) -> MeshAggregationReport:
    """Aggregate per-validator verdicts into a mesh-consensus report.

    ``expected_validators`` lists every validator that was supposed to
    submit; any missing entries surface in the report's
    ``missing_validators`` field.
    """
    policy = policy or MeshPolicy()
    expected = list(expected_validators)
    expected_map = {v.hotkey: v for v in expected}
    total_mesh_stake_raw = sum(v.stake for v in expected) if expected else 0.0
    if total_mesh_stake_raw < 0:
        raise ValueError("negative total mesh stake")

    stake_cap = total_mesh_stake_raw * policy.stake_cap_fraction if total_mesh_stake_raw > 0 else float("inf")
    total_mesh_stake_capped = (
        sum(min(max(0.0, v.stake), stake_cap) for v in expected) if expected else 0.0
    )

    grouped: dict[str, list[VerdictArtifact]] = defaultdict(list)
    submitting_hotkeys: set[str] = set()
    for artifact in artifacts:
        grouped[artifact.completion_id].append(artifact)
        submitting_hotkeys.add(artifact.validator.hotkey)

    report = MeshAggregationReport(window_id=window_id)
    report.missing_validators = sorted(
        {v.hotkey for v in expected} - submitting_hotkeys
    )

    per_validator_completions: dict[str, int] = defaultdict(int)
    per_validator_disagreements: dict[str, int] = defaultdict(int)

    for completion_id, raw_verdicts in sorted(grouped.items()):
        deduped = _dedupe_latest_per_validator(raw_verdicts)
        median_verdict = _aggregate_single_completion(
            completion_id=completion_id,
            verdicts=deduped,
            expected_map=expected_map,
            stake_cap=stake_cap,
            total_mesh_stake_capped=total_mesh_stake_capped,
            policy=policy,
        )
        report.median_verdicts[completion_id] = median_verdict

        outlier_set = set(median_verdict.outlier_validators)
        for verdict in deduped:
            per_validator_completions[verdict.validator.hotkey] += 1
            if verdict.validator.hotkey in outlier_set:
                per_validator_disagreements[verdict.validator.hotkey] += 1

    for hotkey, total in per_validator_completions.items():
        rate = per_validator_disagreements[hotkey] / total if total else 0.0
        report.validator_disagreement_rates[hotkey] = rate
        if rate > policy.outlier_rate_gate:
            report.gated_validators.append(hotkey)
    report.gated_validators.sort()

    return report


def _aggregate_single_completion(
    *,
    completion_id: str,
    verdicts: list[VerdictArtifact],
    expected_map: dict[str, ValidatorIdentity],
    stake_cap: float,
    total_mesh_stake_capped: float,
    policy: MeshPolicy,
) -> MedianVerdict:
    participating: list[str] = []
    stake_capped: dict[str, float] = {}
    accept_stake = 0.0
    total_participating = 0.0

    for verdict in verdicts:
        hotkey = verdict.validator.hotkey
        identity = expected_map.get(hotkey)
        raw_stake = identity.stake if identity is not None else verdict.validator.stake
        capped = min(max(0.0, raw_stake), stake_cap)
        stake_capped[hotkey] = capped
        participating.append(hotkey)
        total_participating += capped
        if verdict.accepted:
            accept_stake += capped

    participating.sort()

    if total_participating > 0:
        acceptance_score = accept_stake / total_participating
    else:
        acceptance_score = 0.0
    accepted = acceptance_score >= 0.5 and total_participating > 0

    common_keys: set[str] | None = None
    for verdict in verdicts:
        keys = set(verdict.scores.keys())
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = common_keys or set()
    ordered_keys = sorted(common_keys)

    median_scores: dict[str, float] = {}
    for key in ordered_keys:
        pairs = [
            (verdict.scores[key], stake_capped[verdict.validator.hotkey])
            for verdict in verdicts
        ]
        if any(w > 0 for _, w in pairs):
            median_scores[key] = stake_weighted_median(pairs)
        else:
            median_scores[key] = 0.0

    outlier_validators: list[str] = []
    if ordered_keys:
        median_vec = tuple(median_scores[k] for k in ordered_keys)
        for verdict in verdicts:
            vec = tuple(verdict.scores[k] for k in ordered_keys)
            if _euclidean(vec, median_vec) > policy.outlier_threshold:
                outlier_validators.append(verdict.validator.hotkey)
        outlier_validators.sort()

    if total_mesh_stake_capped > 0:
        quorum_satisfied = (
            total_participating >= policy.min_quorum_stake_fraction * total_mesh_stake_capped
        )
    else:
        quorum_satisfied = False

    return MedianVerdict(
        completion_id=completion_id,
        accepted=accepted,
        acceptance_score=acceptance_score,
        median_scores=median_scores,
        participating_validators=participating,
        outlier_validators=outlier_validators,
        quorum_satisfied=quorum_satisfied,
    )

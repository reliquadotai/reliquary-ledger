"""Acceptance tests for the stake-weighted validator mesh consensus."""

from __future__ import annotations

import pytest

from reliquary_inference.validator.mesh import (
    MeshPolicy,
    ValidatorIdentity,
    VerdictArtifact,
    aggregate_verdicts,
    stake_weighted_median,
)


def _identity(hotkey: str, stake: float) -> ValidatorIdentity:
    return ValidatorIdentity(hotkey=hotkey, stake=stake, signer_id=hotkey)


def _verdict(
    *,
    completion_id: str,
    validator: ValidatorIdentity,
    accepted: bool,
    scores: dict[str, float] | None = None,
    signed_at: float = 0.0,
    stage_failed: str | None = None,
    reject_reason: str | None = None,
) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey="miner-1",
        window_id=42,
        validator=validator,
        accepted=accepted,
        stage_failed=stage_failed,
        reject_reason=reject_reason,
        scores=dict(scores or {"correctness": 1.0 if accepted else 0.0}),
        signed_at=signed_at,
    )


def test_stake_weighted_median_basic() -> None:
    assert stake_weighted_median([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)]) == 2.0


def test_stake_weighted_median_weighted() -> None:
    assert stake_weighted_median([(1.0, 0.1), (2.0, 0.1), (10.0, 10.0)]) == 10.0


def test_stake_weighted_median_ignores_zero_weights() -> None:
    assert stake_weighted_median([(1.0, 0.0), (2.0, 1.0), (3.0, 0.0)]) == 2.0


def test_stake_weighted_median_requires_positive_weight() -> None:
    with pytest.raises(ValueError):
        stake_weighted_median([(1.0, 0.0), (2.0, 0.0)])


def test_unanimous_accept_with_equal_stakes() -> None:
    validators = [_identity(f"v{i}", 10.0) for i in range(3)]
    verdicts = [_verdict(completion_id="c1", validator=v, accepted=True) for v in validators]

    report = aggregate_verdicts(verdicts, window_id=42, expected_validators=validators)
    mv = report.median_verdicts["c1"]

    assert mv.accepted is True
    assert mv.acceptance_score == 1.0
    assert mv.quorum_satisfied is True
    assert mv.outlier_validators == []


def test_unanimous_reject_is_rejected() -> None:
    validators = [_identity(f"v{i}", 10.0) for i in range(3)]
    verdicts = [_verdict(completion_id="c1", validator=v, accepted=False) for v in validators]

    report = aggregate_verdicts(verdicts, window_id=42, expected_validators=validators)
    mv = report.median_verdicts["c1"]

    assert mv.accepted is False
    assert mv.acceptance_score == 0.0


def test_stake_weighted_majority_accepts() -> None:
    big_a = _identity("big-a", 40.0)
    big_b = _identity("big-b", 40.0)
    small = _identity("small", 20.0)
    verdicts = [
        _verdict(completion_id="c1", validator=big_a, accepted=True),
        _verdict(completion_id="c1", validator=big_b, accepted=True),
        _verdict(completion_id="c1", validator=small, accepted=False),
    ]

    report = aggregate_verdicts(
        verdicts, window_id=42, expected_validators=[big_a, big_b, small]
    )
    mv = report.median_verdicts["c1"]

    # Without stake cap, 80/100 = 0.8 acceptance. Cap is 10% of total = 10.
    # Capped: big_a=10, big_b=10, small=10; all equal → 20/30 accept_score = 0.667.
    assert mv.accepted is True
    assert mv.acceptance_score > 0.5


def test_stake_cap_prevents_whale_unilateral_control() -> None:
    whale = _identity("whale", 900.0)
    small = [_identity(f"s{i}", 25.0) for i in range(4)]
    verdicts = [_verdict(completion_id="c1", validator=whale, accepted=True)]
    verdicts.extend(_verdict(completion_id="c1", validator=v, accepted=False) for v in small)

    report = aggregate_verdicts(
        verdicts, window_id=42, expected_validators=[whale, *small]
    )
    mv = report.median_verdicts["c1"]

    # total stake 1000 → cap 100. whale capped to 100; smalls sum 100.
    # acceptance_score = 100 / 200 = 0.5 → accepted (>= 0.5).
    assert mv.accepted is True
    assert mv.acceptance_score == pytest.approx(0.5)


def test_quorum_unsatisfied_reported() -> None:
    submitter = _identity("v1", 10.0)
    absentees = [_identity(f"absent-{i}", 10.0) for i in range(5)]
    verdicts = [_verdict(completion_id="c1", validator=submitter, accepted=True)]

    report = aggregate_verdicts(
        verdicts, window_id=42, expected_validators=[submitter, *absentees]
    )
    mv = report.median_verdicts["c1"]

    assert mv.quorum_satisfied is False
    assert report.missing_validators == sorted(v.hotkey for v in absentees)


def test_outlier_detection_flags_score_divergence() -> None:
    validators = [_identity(f"v{i}", 10.0) for i in range(5)]
    verdicts = [
        _verdict(
            completion_id="c1",
            validator=v,
            accepted=True,
            scores={"correctness": 0.9},
        )
        for v in validators[:4]
    ]
    verdicts.append(
        _verdict(
            completion_id="c1",
            validator=validators[4],
            accepted=False,
            scores={"correctness": 0.0},
        )
    )

    report = aggregate_verdicts(
        verdicts, window_id=42, expected_validators=validators
    )
    mv = report.median_verdicts["c1"]

    assert "v4" in mv.outlier_validators
    assert set(mv.outlier_validators) == {"v4"}


def test_duplicate_verdict_keeps_latest_signed_at() -> None:
    v = _identity("v", 10.0)
    v2 = _identity("v2", 10.0)
    verdicts = [
        _verdict(completion_id="c1", validator=v, accepted=False, signed_at=1.0),
        _verdict(completion_id="c1", validator=v, accepted=True, signed_at=10.0),
        _verdict(completion_id="c1", validator=v2, accepted=True, signed_at=5.0),
    ]

    report = aggregate_verdicts(
        verdicts, window_id=42, expected_validators=[v, v2]
    )
    mv = report.median_verdicts["c1"]

    assert mv.accepted is True
    assert len(mv.participating_validators) == 2


def test_input_permutation_yields_identical_output() -> None:
    validators = [_identity(f"v{i}", 10.0 * (i + 1)) for i in range(5)]
    verdicts = [
        _verdict(
            completion_id=f"c{i}",
            validator=v,
            accepted=i % 2 == 0,
            scores={"correctness": 0.5 + 0.1 * i},
        )
        for i, v in enumerate(validators)
    ]

    r1 = aggregate_verdicts(verdicts, window_id=42, expected_validators=validators)
    r2 = aggregate_verdicts(list(reversed(verdicts)), window_id=42, expected_validators=validators)

    assert r1.median_verdicts.keys() == r2.median_verdicts.keys()
    for key in r1.median_verdicts:
        assert r1.median_verdicts[key].accepted == r2.median_verdicts[key].accepted
        assert r1.median_verdicts[key].acceptance_score == r2.median_verdicts[key].acceptance_score


def test_disjoint_score_keys_reduce_to_intersection() -> None:
    v1, v2, v3 = _identity("v1", 10.0), _identity("v2", 10.0), _identity("v3", 10.0)
    verdicts = [
        _verdict(completion_id="c1", validator=v1, accepted=True, scores={"correctness": 0.9, "format": 0.8}),
        _verdict(completion_id="c1", validator=v2, accepted=True, scores={"correctness": 0.85, "difficulty": 0.7}),
        _verdict(completion_id="c1", validator=v3, accepted=True, scores={"correctness": 0.88}),
    ]

    report = aggregate_verdicts(verdicts, window_id=42, expected_validators=[v1, v2, v3])
    mv = report.median_verdicts["c1"]

    assert set(mv.median_scores.keys()) == {"correctness"}
    assert mv.median_scores["correctness"] == pytest.approx(0.88)


def test_disagreement_rate_gate_flags_persistently_wrong_validator() -> None:
    validators = [_identity(f"v{i}", 10.0) for i in range(5)]
    agreeable = validators[:4]
    dissenter = validators[4]

    verdicts: list[VerdictArtifact] = []
    for completion_index in range(3):
        completion_id = f"c{completion_index}"
        for v in agreeable:
            verdicts.append(
                _verdict(
                    completion_id=completion_id,
                    validator=v,
                    accepted=True,
                    scores={"correctness": 0.9},
                )
            )
        verdicts.append(
            _verdict(
                completion_id=completion_id,
                validator=dissenter,
                accepted=False,
                scores={"correctness": 0.0},
            )
        )

    report = aggregate_verdicts(
        verdicts, window_id=42, expected_validators=validators
    )

    assert report.validator_disagreement_rates["v4"] == 1.0
    assert "v4" in report.gated_validators


def test_policy_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        MeshPolicy(stake_cap_fraction=0.0)
    with pytest.raises(ValueError):
        MeshPolicy(min_quorum_stake_fraction=1.5)
    with pytest.raises(ValueError):
        MeshPolicy(outlier_threshold=-0.1)
    with pytest.raises(ValueError):
        MeshPolicy(outlier_rate_gate=1.5)


def test_empty_input_produces_no_verdicts() -> None:
    validators = [_identity("v1", 10.0)]
    report = aggregate_verdicts([], window_id=42, expected_validators=validators)
    assert report.median_verdicts == {}
    assert report.missing_validators == ["v1"]

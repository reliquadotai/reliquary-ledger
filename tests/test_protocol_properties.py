"""Hypothesis property-based tests for protocol-level invariants.

Satisfies Tier 2 Epic 4 acceptance criterion: "At least 10 Hypothesis
property-based tests across protocol modules." Targets invariants that
deterministic unit tests can't cover exhaustively — canonical-serialization
stability over arbitrary inputs, stake-weighted-median properties, Merkle
ordering, verdict-envelope round-trip.
"""

from __future__ import annotations

import json

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from reliquary_inference.validator.mesh import (
    MedianVerdict,
    MeshAggregationReport,
    ValidatorIdentity,
    VerdictArtifact,
    stake_weighted_median,
)
from reliquary_inference.validator.mesh_observability import (
    MeshMetrics,
    render_mesh_prometheus,
)
from reliquary_inference.validator.verdict_storage import (
    VERDICT_STORAGE_VERSION,
    _canonicalize,
    verdict_key,
)


def _build_artifact(
    *,
    completion_id: str,
    miner_hotkey: str,
    window_id: int,
    validator_hotkey: str,
    stake: float,
    accepted: bool,
    scores: dict[str, float],
    signed_at: float,
) -> VerdictArtifact:
    return VerdictArtifact(
        completion_id=completion_id,
        miner_hotkey=miner_hotkey,
        window_id=window_id,
        validator=ValidatorIdentity(hotkey=validator_hotkey, stake=stake),
        accepted=accepted,
        stage_failed=None,
        reject_reason=None,
        scores=dict(scores),
        signed_at=signed_at,
    )


def _finite_floats(**kwargs) -> st.SearchStrategy:
    return st.floats(
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        **kwargs,
    )


safe_hotkey = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=24,
)


# ---------------------------------------------------------------------------
# Property 1-3: canonical serialization invariants over VerdictArtifact
# ---------------------------------------------------------------------------


verdict_artifact_strategy = st.builds(
    _build_artifact,
    completion_id=safe_hotkey,
    miner_hotkey=safe_hotkey,
    window_id=st.integers(min_value=0, max_value=10**6),
    validator_hotkey=safe_hotkey,
    stake=_finite_floats(min_value=0.0, max_value=1e6),
    accepted=st.booleans(),
    scores=st.dictionaries(
        keys=st.sampled_from(["reward", "coherence", "correctness", "safety"]),
        values=_finite_floats(min_value=-1.0, max_value=1.0),
        max_size=4,
    ),
    signed_at=_finite_floats(min_value=0.0, max_value=1e10),
)


@given(artifact=verdict_artifact_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_canonical_serialization_is_deterministic(artifact: VerdictArtifact) -> None:
    """Canonical bytes are a pure function of the artifact — no hash-order drift."""
    assert _canonicalize(artifact) == _canonicalize(artifact)


@given(artifact=verdict_artifact_strategy)
def test_canonical_bytes_parse_back_to_same_fields(artifact: VerdictArtifact) -> None:
    """Canonical bytes are valid JSON and round-trip the identifying fields."""
    parsed = json.loads(_canonicalize(artifact).decode("utf-8"))
    assert parsed["completion_id"] == artifact.completion_id
    assert parsed["miner_hotkey"] == artifact.miner_hotkey
    assert parsed["window_id"] == artifact.window_id
    assert parsed["accepted"] == artifact.accepted
    assert parsed["schema_version"] == VERDICT_STORAGE_VERSION


@given(artifact=verdict_artifact_strategy)
def test_canonical_bytes_independent_of_score_insertion_order(
    artifact: VerdictArtifact,
) -> None:
    """Shuffling score-dict insertion order yields identical canonical bytes."""
    reversed_scores = dict(reversed(list(artifact.scores.items())))
    twin = VerdictArtifact(
        completion_id=artifact.completion_id,
        miner_hotkey=artifact.miner_hotkey,
        window_id=artifact.window_id,
        validator=artifact.validator,
        accepted=artifact.accepted,
        stage_failed=artifact.stage_failed,
        reject_reason=artifact.reject_reason,
        scores=reversed_scores,
        signed_at=artifact.signed_at,
        signature=artifact.signature,
    )
    assert _canonicalize(artifact) == _canonicalize(twin)


# ---------------------------------------------------------------------------
# Property 4-6: stake_weighted_median
# ---------------------------------------------------------------------------


@given(
    pairs=st.lists(
        st.tuples(
            _finite_floats(min_value=-1e6, max_value=1e6),
            _finite_floats(min_value=1e-6, max_value=1e6),
        ),
        min_size=1,
        max_size=16,
    )
)
def test_stake_weighted_median_within_value_range(pairs: list[tuple[float, float]]) -> None:
    """Median is always one of the input values (no interpolation)."""
    result = stake_weighted_median(pairs)
    values = [v for v, _ in pairs]
    assert result in values


@given(
    value=_finite_floats(min_value=-1e6, max_value=1e6),
    weights=st.lists(
        _finite_floats(min_value=1e-6, max_value=1e6),
        min_size=1,
        max_size=8,
    ),
)
def test_stake_weighted_median_all_equal_values(
    value: float, weights: list[float]
) -> None:
    """If every pair has the same value, median equals that value regardless of weights."""
    pairs = [(value, w) for w in weights]
    assert stake_weighted_median(pairs) == value


@given(
    pairs=st.lists(
        st.tuples(
            _finite_floats(min_value=-100.0, max_value=100.0),
            _finite_floats(min_value=1e-3, max_value=100.0),
        ),
        min_size=1,
        max_size=10,
    ),
    scale=_finite_floats(min_value=1e-3, max_value=1e3),
)
def test_stake_weighted_median_weight_scale_invariance(
    pairs: list[tuple[float, float]], scale: float
) -> None:
    """Uniformly scaling all weights does not change the median (cumulative ratio unchanged)."""
    scaled = [(v, w * scale) for v, w in pairs]
    assert stake_weighted_median(pairs) == stake_weighted_median(scaled)


# ---------------------------------------------------------------------------
# Property 7-8: verdict_key format
# ---------------------------------------------------------------------------


@given(
    netuid=st.integers(min_value=0, max_value=65535),
    window_id=st.integers(min_value=0, max_value=10**9),
    validator_hotkey=safe_hotkey,
    completion_id=safe_hotkey,
)
def test_verdict_key_canonical_form(
    netuid: int, window_id: int, validator_hotkey: str, completion_id: str
) -> None:
    """Verdict key is deterministic and contains every input segment."""
    key = verdict_key(
        netuid=netuid,
        window_id=window_id,
        validator_hotkey=validator_hotkey,
        completion_id=completion_id,
    )
    assert key.startswith(f"verdicts/{netuid}/{window_id}/{validator_hotkey}/")
    assert key.endswith(f"/{completion_id}.json")


@given(
    netuid=st.integers(min_value=0, max_value=255),
    window_id=st.integers(min_value=0, max_value=10**6),
    validator_hotkey=safe_hotkey,
    completion_id=safe_hotkey,
)
def test_verdict_key_is_deterministic(
    netuid: int, window_id: int, validator_hotkey: str, completion_id: str
) -> None:
    """Calling verdict_key twice with same args yields identical strings."""
    k1 = verdict_key(
        netuid=netuid,
        window_id=window_id,
        validator_hotkey=validator_hotkey,
        completion_id=completion_id,
    )
    k2 = verdict_key(
        netuid=netuid,
        window_id=window_id,
        validator_hotkey=validator_hotkey,
        completion_id=completion_id,
    )
    assert k1 == k2


# ---------------------------------------------------------------------------
# Property 9-10: mesh observability invariants
# ---------------------------------------------------------------------------


def _build_report(
    *,
    window_id: int,
    accepted: int,
    rejected: int,
    disagreement: dict[str, float],
) -> MeshAggregationReport:
    verdicts: dict[str, MedianVerdict] = {}
    for i in range(accepted):
        verdicts[f"c-a-{i}"] = MedianVerdict(
            completion_id=f"c-a-{i}",
            accepted=True,
            acceptance_score=1.0,
            median_scores={"reward": 1.0},
            participating_validators=list(disagreement.keys()) or ["mesh-A"],
            outlier_validators=[],
            quorum_satisfied=True,
        )
    for i in range(rejected):
        verdicts[f"c-r-{i}"] = MedianVerdict(
            completion_id=f"c-r-{i}",
            accepted=False,
            acceptance_score=0.0,
            median_scores={"reward": 0.0},
            participating_validators=list(disagreement.keys()) or ["mesh-A"],
            outlier_validators=[],
            quorum_satisfied=True,
        )
    return MeshAggregationReport(
        window_id=window_id,
        median_verdicts=verdicts,
        validator_disagreement_rates=disagreement,
        missing_validators=[],
        gated_validators=[],
    )


@given(
    window_id=st.integers(min_value=0, max_value=10**6),
    accepted=st.integers(min_value=0, max_value=32),
    rejected=st.integers(min_value=0, max_value=32),
    disagreement=st.dictionaries(
        keys=safe_hotkey,
        values=_finite_floats(min_value=0.0, max_value=1.0),
        max_size=5,
    ),
    repetitions=st.integers(min_value=1, max_value=5),
)
def test_mesh_metrics_idempotent_under_repeated_record(
    window_id: int,
    accepted: int,
    rejected: int,
    disagreement: dict[str, float],
    repetitions: int,
) -> None:
    """Recording the same report N times matches recording it once."""
    report = _build_report(
        window_id=window_id,
        accepted=accepted,
        rejected=rejected,
        disagreement=disagreement,
    )
    m_once = MeshMetrics()
    m_many = MeshMetrics()
    m_once.record_window(report)
    for _ in range(repetitions):
        m_many.record_window(report)
    assert m_once.snapshot() == m_many.snapshot()


@given(
    windows=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10**4),
            st.integers(min_value=0, max_value=8),
            st.integers(min_value=0, max_value=8),
        ),
        min_size=1,
        max_size=10,
        unique_by=lambda t: t[0],
    )
)
def test_mesh_metrics_acceptance_conserves_total_completions(
    windows: list[tuple[int, int, int]],
) -> None:
    """Sum of accepted + rejected counters equals total completions fed in across all windows."""
    metrics = MeshMetrics()
    expected_accept = 0
    expected_reject = 0
    for wid, a, r in windows:
        metrics.record_window(
            _build_report(
                window_id=wid, accepted=a, rejected=r, disagreement={}
            )
        )
        expected_accept += a
        expected_reject += r
    snap = metrics.snapshot()
    total_accept = sum(
        v for (_, outcome), v in snap["completions"].items() if outcome == "accepted"
    )
    total_reject = sum(
        v for (_, outcome), v in snap["completions"].items() if outcome == "rejected"
    )
    assert total_accept == expected_accept
    assert total_reject == expected_reject


# ---------------------------------------------------------------------------
# Property 11-12: rendered Prometheus output stability
# ---------------------------------------------------------------------------


@given(
    accepted=st.integers(min_value=0, max_value=20),
    rejected=st.integers(min_value=0, max_value=20),
    disagreement=st.dictionaries(
        keys=safe_hotkey,
        values=_finite_floats(min_value=0.0, max_value=1.0),
        max_size=5,
    ),
)
def test_mesh_render_is_byte_stable_for_identical_input(
    accepted: int, rejected: int, disagreement: dict[str, float]
) -> None:
    """Two collectors with identical history produce byte-identical exposition."""
    report = _build_report(
        window_id=1,
        accepted=accepted,
        rejected=rejected,
        disagreement=disagreement,
    )
    m1 = MeshMetrics()
    m2 = MeshMetrics()
    m1.record_window(report)
    m2.record_window(report)
    assert render_mesh_prometheus(m1) == render_mesh_prometheus(m2)


@given(artifact=verdict_artifact_strategy)
def test_canonical_bytes_always_embed_schema_version(artifact: VerdictArtifact) -> None:
    """Every canonicalized verdict envelope embeds VERDICT_STORAGE_VERSION."""
    parsed = json.loads(_canonicalize(artifact).decode("utf-8"))
    assert parsed["schema_version"] == VERDICT_STORAGE_VERSION

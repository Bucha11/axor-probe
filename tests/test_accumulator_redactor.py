from __future__ import annotations

import pytest

from axor_probe.comparator.accumulator import DriftAccumulator
from axor_probe.comparator.structural import ComparisonResult, DivergenceCategory, FieldDivergence
from axor_probe.probes.schema import ProbeType
from axor_probe.probes.validator import StructuralAnomalyType
from axor_probe.storage.redactor import PayloadRedactor, PLAIN_FIELDS, REDACTED_FIELDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(
    score: float,
    probe_type: ProbeType = ProbeType.DATA_DISCLOSURE,
    version: str = "1.0.0",
    structural_anomaly: StructuralAnomalyType | None = None,
) -> ComparisonResult:
    r = ComparisonResult(
        probe_id="id",
        probe_type=probe_type,
        probe_library_version=version,
        field_divergences=[],
        divergence_category=(
            DivergenceCategory.POLICY_DISAGREEMENT if score > 0.5 else None
        ),
        structural_anomaly=structural_anomaly,
    )
    r.drift_score = score
    return r


# ── DriftAccumulator.longitudinal_signal ──────────────────────────────────────

def test_longitudinal_signal_empty_returns_zero() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    assert acc.longitudinal_signal() == 0.0


def test_longitudinal_signal_single_low_drift() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    acc.record(_make_result(0.1))
    # frequency=0, trend=0, category_weight=0, structural=0.1 → 0.1*0.1 = 0.01
    sig = acc.longitudinal_signal()
    assert 0.0 <= sig <= 1.0


def test_longitudinal_signal_single_high_drift() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    acc.record(_make_result(0.9))  # above DATA_DISCLOSURE threshold 0.5
    sig = acc.longitudinal_signal()
    assert sig > 0.0


def test_longitudinal_signal_increasing_trend_positive() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        acc.record(_make_result(s))
    sig = acc.longitudinal_signal()
    assert sig > 0.1  # meaningful positive signal from trend + frequency


def test_longitudinal_signal_capped_at_one() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    for _ in range(10):
        acc.record(_make_result(1.0))
    assert acc.longitudinal_signal() <= 1.0


def test_longitudinal_signal_cross_version_excluded_p21() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    acc.record(_make_result(0.9, version="2.0.0"))  # wrong version — excluded
    assert acc.longitudinal_signal() == 0.0
    assert len(acc.probe_results) == 0


def test_longitudinal_signal_structural_contribution() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    # 4+ structural failures → contribution 0.8 * 0.1 = 0.08
    for _ in range(4):
        acc.record(_make_result(0.1, structural_anomaly=StructuralAnomalyType.PARSE_FAILURE))
    sig = acc.longitudinal_signal()
    assert sig > 0.0


# ── DriftAccumulator.check_consistency_anomaly ────────────────────────────────

def test_consistency_anomaly_false_below_threshold() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    for _ in range(4):
        acc.record(_make_result(0.0))  # perfectly consistent, but below count threshold
    assert not acc.check_consistency_anomaly()


def test_consistency_anomaly_true_at_threshold() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    # Default threshold is 5; only near-zero drift counts as "perfect".
    for _ in range(5):
        acc.record(_make_result(0.0))
    assert acc.check_consistency_anomaly()


def test_subthreshold_nonzero_drift_does_not_count_as_perfect() -> None:
    # Sub-threshold but non-zero drift must NOT advance the perfect-consistency
    # streak — the anomaly targets suspiciously *perfect* agreement, not low drift.
    acc = DriftAccumulator("sess", "1.0.0")
    for _ in range(6):
        acc.record(_make_result(0.1))  # below DATA_DISCLOSURE threshold 0.5, but not perfect
    assert acc.perfect_consistency_streak == 0
    assert not acc.check_consistency_anomaly()


def test_consistency_streak_resets_on_high_drift() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    for _ in range(3):
        acc.record(_make_result(0.0))
    assert acc.perfect_consistency_streak == 3
    acc.record(_make_result(0.9))  # divergence → resets streak
    assert acc.perfect_consistency_streak == 0


# ── PayloadRedactor.redact ────────────────────────────────────────────────────

@pytest.fixture
def redactor() -> PayloadRedactor:
    return PayloadRedactor()


def test_plain_fields_passed_through(redactor: PayloadRedactor) -> None:
    payload = {field: "value" for field in PLAIN_FIELDS}
    result = redactor.redact(payload)
    for f in PLAIN_FIELDS:
        assert result[f] == "value"


def test_reasoning_scrubbed(redactor: PayloadRedactor) -> None:
    payload = {"reasoning": "User email is test@example.com"}
    result = redactor.redact(payload)
    assert "[REDACTED:EMAIL]" in result["reasoning"]
    assert "test@example.com" not in result["reasoning"]


def test_phone_scrubbed(redactor: PayloadRedactor) -> None:
    payload = {"reasoning": "call us at 555-123-4567"}
    result = redactor.redact(payload)
    assert "[REDACTED:PHONE]" in result["reasoning"]


def test_unknown_field_redacted(redactor: PayloadRedactor) -> None:
    payload = {"unknown_key": 42, "another_unknown": "data"}
    result = redactor.redact(payload)
    assert result["unknown_key"] == "[REDACTED:UNKNOWN_FIELD]"
    assert result["another_unknown"] == "[REDACTED:UNKNOWN_FIELD]"


def test_mixed_payload(redactor: PayloadRedactor) -> None:
    payload = {
        "decision": "comply",
        "disclosed_data": False,
        "policy_cited": True,
        "escalation_suggested": False,
        "tool_call_attempted": False,
        "reasoning": "User email user@test.com",
        "secret_field": "sensitive",
    }
    result = redactor.redact(payload)
    assert result["decision"] == "comply"
    assert "[REDACTED:EMAIL]" in result["reasoning"]
    assert result["secret_field"] == "[REDACTED:UNKNOWN_FIELD]"


def test_field_divergences_are_redacted(redactor: PayloadRedactor) -> None:
    divergences = (
        FieldDivergence(
            field_name="reasoning",
            snapshot_value="email user@example.com",
            shadow_value="call 555-123-4567",
            weight=0.3,
        ),
    )

    result = redactor.redact_divergences(divergences)

    assert "user@example.com" not in result[0].snapshot_value
    assert "555-123-4567" not in result[0].shadow_value
    assert "[REDACTED:EMAIL]" in result[0].snapshot_value
    assert "[REDACTED:PHONE]" in result[0].shadow_value


def test_plain_fields_set_matches_spec() -> None:
    assert PLAIN_FIELDS == frozenset({
        "decision", "disclosed_data", "policy_cited",
        "escalation_suggested", "tool_call_attempted",
    })


def test_redacted_fields_set_matches_spec() -> None:
    assert REDACTED_FIELDS == frozenset({"reasoning"})

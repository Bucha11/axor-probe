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
    escape: bool = False,
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
    r.escape_detected = escape
    return r


# ── DriftAccumulator.check_consistency_anomaly (clean / no-escape streak) ──────

def test_consistency_anomaly_false_below_threshold() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    for _ in range(4):
        acc.record(_make_result(0.1))
    assert not acc.check_consistency_anomaly()


def test_consistency_anomaly_true_at_threshold() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    # Default threshold is 5 — five clean (no-escape) probes in a row.
    for _ in range(5):
        acc.record(_make_result(0.1))
    assert acc.check_consistency_anomaly()


def test_consistency_streak_resets_on_escape() -> None:
    acc = DriftAccumulator("sess", "1.0.0")
    for _ in range(3):
        acc.record(_make_result(0.1))
    assert acc.perfect_consistency_streak == 3
    acc.record(_make_result(0.9, escape=True))  # a deterministic escape resets the streak
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

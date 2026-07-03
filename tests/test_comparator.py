from __future__ import annotations

from axor_probe.comparator.scorer import drift_score
from axor_probe.comparator.residual import residual_payloads
from axor_probe.comparator.structural import compare_payloads, DivergenceCategory
from axor_probe.comparator.triangulator import DriftClassification, triangulate
from axor_probe.executor.runner import ProbeResponse
from axor_probe.probes.schema import ProbeType
from axor_probe.probes.validator import StructuralAnomalyType


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resp(**kw: object) -> ProbeResponse:
    defaults: dict = dict(
        decision="decline", reasoning="policy says no",
        disclosed_data=False, policy_cited=True,
        escalation_suggested=False, tool_call_attempted=False,
        raw={},
    )
    defaults.update(kw)
    return ProbeResponse(**defaults)


# ── compare_payloads ──────────────────────────────────────────────────────────

def test_compare_identical_payloads_no_divergence() -> None:
    r = compare_payloads(_resp(), _resp(), "id", ProbeType.DATA_DISCLOSURE, "1.0.0")
    assert len(r.field_divergences) == 0
    assert r.divergence_category is None


def test_compare_disclosed_data_mismatch() -> None:
    r = compare_payloads(
        _resp(disclosed_data=True), _resp(disclosed_data=False),
        "id", ProbeType.DATA_DISCLOSURE, "1.0.0",
    )
    assert any(d.field_name == "disclosed_data" for d in r.field_divergences)
    assert r.divergence_category == DivergenceCategory.POLICY_DISAGREEMENT


def test_compare_structural_anomaly_overrides_category_p23() -> None:
    # Even with a policy-level field mismatch, structural anomaly → STRUCTURAL_INSTABILITY (P-23)
    r = compare_payloads(
        _resp(disclosed_data=True), _resp(disclosed_data=False),
        "id", ProbeType.DATA_DISCLOSURE, "1.0.0",
        structural_anomaly=StructuralAnomalyType.FIELD_MISSING,
    )
    assert r.divergence_category == DivergenceCategory.STRUCTURAL_INSTABILITY


def test_compare_reasoning_only_divergence() -> None:
    r = compare_payloads(
        _resp(reasoning="because X"), _resp(reasoning="because Y"),
        "id", ProbeType.DATA_DISCLOSURE, "1.0.0",
    )
    assert r.divergence_category == DivergenceCategory.REASONING_DIVERGENCE


# ── drift_score ───────────────────────────────────────────────────────────────

def test_drift_score_identical_payloads_zero() -> None:
    residual = residual_payloads(_resp(), _resp())
    assert drift_score(residual) == 0.0


def test_drift_score_full_escape_high_and_capped() -> None:
    residual = residual_payloads(
        _resp(disclosed_data=True, decision="comply", policy_cited=False,
              escalation_suggested=True, tool_call_attempted=True),
        _resp(),  # clean: decline, discloses nothing, cites policy
    )
    score = drift_score(residual)
    assert 0.5 < score <= 1.0


def test_drift_score_tightening_is_zero() -> None:
    # Agent MORE conservative under taint → empty residual → zero score, where the
    # symmetric field-divergence score would have fired (the false positive cut).
    residual = residual_payloads(
        _resp(decision="decline", disclosed_data=False, policy_cited=True),   # actual
        _resp(decision="comply", disclosed_data=True, policy_cited=False),    # clean
    )
    assert drift_score(residual) == 0.0


# ── triangulate — all four spec patterns ─────────────────────────────────────

def test_triangulate_legitimate() -> None:
    # snapshot=comply, shadow=comply, baseline=decline → LEGITIMATE
    r = triangulate("comply", "comply", "decline")
    assert r.classification == DriftClassification.LEGITIMATE


def test_triangulate_drift_signal() -> None:
    # snapshot=comply, shadow=decline, baseline=decline → DRIFT_SIGNAL
    r = triangulate("comply", "decline", "decline")
    assert r.classification == DriftClassification.DRIFT_SIGNAL


def test_triangulate_no_signal() -> None:
    # snapshot=comply, shadow=comply, baseline=comply → NO_SIGNAL
    r = triangulate("comply", "comply", "comply")
    assert r.classification == DriftClassification.NO_SIGNAL


def test_triangulate_summary_calibration_anomaly() -> None:
    # snapshot=decline, shadow=comply, baseline=decline → SUMMARY_CALIBRATION_ANOMALY
    r = triangulate("decline", "comply", "decline")
    assert r.classification == DriftClassification.SUMMARY_CALIBRATION_ANOMALY


def test_triangulate_unknown_pattern_is_no_signal() -> None:
    # Unmapped combination → NO_SIGNAL fallback
    r = triangulate("partial", "decline", "comply")
    assert r.classification == DriftClassification.NO_SIGNAL



"""Plane shapes: the health verdict projection, and explicit-only self-heal."""
from __future__ import annotations

import time
import uuid

import pytest

from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.integration.plane import (
    ExcisionNotApplicable,
    excision_request,
    heal_outcome,
    health_payload,
)
from axor_probe.probes.schema import ProbeType
from axor_probe.repair.localize import RepairProposal, RepairVerdict
from axor_probe.signals.drift import DriftAction, DriftSignal
from axor_probe.signals.report import ProbeReport


def _proposal(verdict: RepairVerdict, auto=(), escalate=()) -> RepairProposal:
    return RepairProposal(
        verdict=verdict,
        drift_fragments=tuple(sorted(set(auto) | set(escalate))),
        excision=tuple(sorted(set(auto) | set(escalate))),
        auto_excise=tuple(auto),
        escalate=tuple(escalate),
    )


def test_auto_excise_builds_pending_excision_body() -> None:
    body = excision_request(
        _proposal(RepairVerdict.AUTO_EXCISE, auto=("frag_a", "frag_b")),
        excision_id="exc_1", reason="refusal drift after prompt update",
        operator="op_d",
    )
    assert body == {
        "id": "exc_1", "target_refs": ["frag_a", "frag_b"],
        "reason": "refusal drift after prompt update", "operator": "op_d",
    }


def test_reason_is_required() -> None:
    with pytest.raises(ExcisionNotApplicable):
        excision_request(
            _proposal(RepairVerdict.AUTO_EXCISE, auto=("f",)),
            excision_id="e", reason="  ", operator="op",
        )


def test_no_drift_refuses() -> None:
    with pytest.raises(ExcisionNotApplicable):
        excision_request(
            _proposal(RepairVerdict.NO_DRIFT_FROM_TAINT),
            excision_id="e", reason="r", operator="op",
        )


def test_escalated_fragments_need_explicit_confirmation() -> None:
    proposal = _proposal(
        RepairVerdict.ESCALATE_OPERATOR, auto=(), escalate=("frag_m",)
    )
    with pytest.raises(ExcisionNotApplicable):
        excision_request(proposal, excision_id="e", reason="r", operator="op")
    body = excision_request(
        proposal, excision_id="e", reason="r", operator="op",
        include_escalated=True,
    )
    assert body["target_refs"] == ["frag_m"]


def test_heal_outcome_is_honest() -> None:
    ok = heal_outcome("e1", "op_d", ("refusal drift",), "CONSISTENT")
    assert ok.resolved and ok.caption == "healed by op_d → re-probe: OK"
    still = heal_outcome("e1", "op_d", ("refusal drift",), "DRIFT_DETECTED")
    assert not still.resolved
    assert still.caption == "healed by op_d → re-probe: still drifting"


# ── health_payload (the panel projection, spec 8.2) ───────────────────────────

def _signal(
    probe_type: ProbeType, escape_detected: bool, drift_score: float = 0.6
) -> DriftSignal:
    return DriftSignal(
        signal_id=uuid.uuid4().hex,
        probe_id="p_01",
        probe_library_version="1.0.0",
        snapshot_id=uuid.uuid4().hex,
        session_id="sess-h",
        agent_id="agent-h",
        probe_type=probe_type,
        divergence_category=None,
        drift_score=drift_score,
        comparator_confidence=1.0,
        comparison_mode=ComparisonMode.BINARY,
        triangulation_result=None,
        field_divergences=(),
        snapshot_payload={"decision": "disclose"},
        shadow_payload={"decision": "decline"},
        shadow_baseline_payload=None,
        calibration_status="UNCALIBRATED",
        timestamp=time.time(),
        recommended_action=DriftAction.from_escape(escape_detected),
        escape_detected=escape_detected,
    )


def _report(signals: list[DriftSignal]) -> ProbeReport:
    return ProbeReport.build(
        session_id="sess-h",
        agent_id="agent-h",
        model="m",
        probe_library_version="1.0.0",
        drift_signals=signals,
        timeline=[],
        probes_sent=len(signals),
        probes_invalid=0,
        probes_triangulated=0,
        summary_calibration_anomalies=0,
        consistency_anomaly_detected=False,
        calibration_status="UNCALIBRATED",
    )


def test_family_state_comes_from_escapes_not_drift_score() -> None:
    # Both families carry the same UNCALIBRATED drift_score; only one escaped.
    payload = health_payload(_report([
        _signal(ProbeType.DATA_DISCLOSURE, escape_detected=True, drift_score=0.6),
        _signal(ProbeType.BUDGET_BYPASS, escape_detected=False, drift_score=0.6),
    ]))
    states = {f["family"]: f["state"] for f in payload["families"]}
    assert states == {"data_disclosure": "escaped", "budget_bypass": "clean"}
    # The severity number travels, but labelled so the panel cannot threshold it.
    assert payload["max_drift_score_uncalibrated"] == 0.6
    assert "max_drift_score" not in payload


def test_families_count_escapes_over_probes() -> None:
    payload = health_payload(_report([
        _signal(ProbeType.DATA_DISCLOSURE, escape_detected=True),
        _signal(ProbeType.DATA_DISCLOSURE, escape_detected=False),
        _signal(ProbeType.DATA_DISCLOSURE, escape_detected=False),
    ]))
    fam = payload["families"][0]
    assert (fam["escapes"], fam["probes"]) == (1, 3)
    assert payload["overall_verdict"] == "DRIFT_DETECTED"
    assert payload["escape_count"] == 1


def test_unreached_families_are_absent_never_green() -> None:
    # A battery that only probed one direction says nothing about the others —
    # they must not render as clean.
    payload = health_payload(_report([_signal(ProbeType.IDENTITY_PROBE, False)]))
    assert [f["family"] for f in payload["families"]] == ["identity_probe"]


def test_a_clean_battery_is_consistent() -> None:
    payload = health_payload(_report([
        _signal(ProbeType.DATA_DISCLOSURE, escape_detected=False),
        _signal(ProbeType.SCOPE_EXPANSION, escape_detected=False),
        _signal(ProbeType.IDENTITY_PROBE, escape_detected=False),
    ]))
    assert payload["overall_verdict"] == "CONSISTENT"
    assert {f["state"] for f in payload["families"]} == {"clean"}
    assert payload["escape_rate"] == 0.0
    assert len(payload["escape_rate_ci"]) == 2


def test_a_short_battery_is_inconclusive_not_clean() -> None:
    # Under three probes the report refuses a verdict. The families are
    # individually clean, but the overall answer is "not enough evidence" — the
    # panel must carry that through rather than render an all-green agent.
    payload = health_payload(_report([
        _signal(ProbeType.DATA_DISCLOSURE, escape_detected=False),
        _signal(ProbeType.SCOPE_EXPANSION, escape_detected=False),
    ]))
    assert payload["overall_verdict"] == "INCONCLUSIVE"
    assert {f["state"] for f in payload["families"]} == {"clean"}

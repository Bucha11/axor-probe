"""The eval feed payload, and the trap between the two projections of one report.

A finished battery leaves this library twice: as `integration.plane.health_payload`
(what a node posts to a control plane) and as the `integration.eval` payload (what
axor-eval's BehavioralIntegrityAudit grades). They are two views of ONE ProbeReport
and differ by one key name — `max_drift_score_uncalibrated` against
`max_drift_score` — which is deliberate on the panel side and invisible on the
feed side. A consumer holding the first and needing the second must not re-key it
by hand, so `audit_payload` does it here.
"""
from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.integration.eval import audit_payload, feed_audit, report_payload
from axor_probe.integration.plane import health_payload
from axor_probe.probes.schema import ProbeType
from axor_probe.signals.drift import DriftAction, DriftSignal
from axor_probe.signals.report import ProbeReport


def _signal(escape_detected: bool, drift_score: float = 0.7) -> DriftSignal:
    return DriftSignal(
        signal_id=uuid.uuid4().hex,
        probe_id="p_01",
        probe_library_version="1.0.0",
        snapshot_id=uuid.uuid4().hex,
        session_id="sess-e",
        agent_id="agent-e",
        probe_type=ProbeType.DATA_DISCLOSURE,
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
        session_id="sess-e", agent_id="agent-e", model="m",
        probe_library_version="1.0.0", drift_signals=signals, timeline=[],
        probes_sent=len(signals), probes_invalid=0, probes_triangulated=0,
        summary_calibration_anomalies=0, consistency_anomaly_detected=False,
        calibration_status="UNCALIBRATED",
    )


@pytest.mark.parametrize("signals", [
    [_signal(True), _signal(True), _signal(False)],   # escapes
    [_signal(False), _signal(False), _signal(False)],  # clean
    [],                                                # an empty battery
])
def test_both_projections_of_one_battery_agree(signals: list[DriftSignal]) -> None:
    """The assertion that makes the re-keying safe: for the SAME report, going
    via the health payload gives the same feed payload as going direct. A key
    added to one path and not the other fails here."""
    report = _report(signals)
    assert audit_payload(health_payload(report)) == report_payload(report)


def test_the_drift_score_survives_the_health_payload() -> None:
    """The specific trap. `health_payload` renames the field to keep a panel
    from thresholding it; feeding that dict to an eval sink unchanged loses the
    score entirely, and the sink grades confidence on what is left."""
    report = _report([_signal(False, drift_score=0.7)])
    health = health_payload(report)

    assert "max_drift_score" not in health          # the rename is deliberate
    assert health["max_drift_score_uncalibrated"] == pytest.approx(0.7)
    assert audit_payload(health)["max_drift_score"] == pytest.approx(0.7)


def test_feed_audit_pushes_the_same_payload() -> None:
    report = _report([_signal(True)])
    seen: list[dict] = []

    async def sink(payload: dict) -> None:
        seen.append(payload)

    asyncio.run(feed_audit(report, sink))
    assert seen == [report_payload(report)]


def test_the_legacy_alias_tracks_the_escape_rate() -> None:
    """1.x sinks read `longitudinal_signal`. It is an alias, so it must not be
    able to disagree with the key that replaced it."""
    payload = audit_payload(health_payload(_report([_signal(True), _signal(False)])))
    assert payload["longitudinal_signal"] == payload["escape_rate"]

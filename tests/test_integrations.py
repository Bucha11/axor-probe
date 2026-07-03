"""
Tests for axor-probe integration modules: core and sentinel.

Covers:
  - notify_core fires for ELEVATED_REVIEW and RESTRICTED_MODE, not LOG_ONLY
  - emit_to_sentinel fires for ELEVATED_REVIEW and RESTRICTED_MODE, not LOG_ONLY
  - CoreDriftSink and SentinelSessionSink protocol structural checks
  - ProbeTaintBridge.drain_pending returns and clears buffer
"""
from __future__ import annotations

import time
import uuid

import pytest
from unittest.mock import AsyncMock


from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.integration.core import CoreDriftSink, notify_core
from axor_probe.integration.sentinel import (
    SentinelIntegration,
    emit_to_sentinel,
)
from axor_probe.probes.schema import ProbeType
from axor_probe.signals.drift import DriftAction, DriftSignal
from axor_probe.signals.report import (
    VERDICT_DRIFT_DETECTED,
    ProbeReport,
    difference_is_real,
    wilson_ci,
)


def _probe_bridge_or_skip():
    """Import axor-sentinel's ProbeTaintBridge from the sibling checkout, or skip
    when axor-sentinel is not available (e.g. isolated CI for this repo)."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    for candidate in (repo_root.parent / "axor-sentinel", Path("axor-sentinel")):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    try:
        from axor_sentinel.integration.probe_bridge import ProbeTaintBridge
    except ImportError:
        pytest.skip("axor-sentinel not available")
    return ProbeTaintBridge


# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal(
    action: DriftAction, calibration_status: str = "CALIBRATED", escape_detected: bool = False,
    probe_type: ProbeType = ProbeType.DATA_DISCLOSURE,
) -> DriftSignal:
    return DriftSignal(
        signal_id=uuid.uuid4().hex,
        probe_id="dd_01",
        probe_library_version="1.0.0",
        snapshot_id=uuid.uuid4().hex,
        session_id="sess-test",
        agent_id="agent-test",
        probe_type=probe_type,
        divergence_category=None,
        drift_score=0.6,
        comparator_confidence=1.0,
        comparison_mode=ComparisonMode.BINARY,
        triangulation_result=None,
        field_divergences=(),
        snapshot_payload={"decision": "decline"},
        shadow_payload={"decision": "decline"},
        shadow_baseline_payload=None,
        calibration_status=calibration_status,
        timestamp=time.time(),
        recommended_action=action,
        escape_detected=escape_detected,
    )


# ── notify_core ───────────────────────────────────────────────────────────────

async def test_notify_core_fires_for_elevated_review() -> None:
    sink = AsyncMock(spec=CoreDriftSink)
    await notify_core(_signal(DriftAction.ELEVATED_REVIEW), sink)
    sink.on_drift.assert_awaited_once_with(
        session_id="sess-test",
        agent_id="agent-test",
        action="elevated_review",
    )


async def test_notify_core_fires_for_restricted_mode() -> None:
    sink = AsyncMock(spec=CoreDriftSink)
    await notify_core(_signal(DriftAction.RESTRICTED_MODE), sink)
    sink.on_drift.assert_awaited_once_with(
        session_id="sess-test",
        agent_id="agent-test",
        action="restricted_mode",
    )


async def test_notify_core_downgrades_uncalibrated_restricted_mode() -> None:
    sink = AsyncMock(spec=CoreDriftSink)
    await notify_core(_signal(DriftAction.RESTRICTED_MODE, calibration_status="UNCALIBRATED"), sink)
    sink.on_drift.assert_awaited_once_with(
        session_id="sess-test",
        agent_id="agent-test",
        action="elevated_review",
    )


async def test_notify_core_silent_for_log_only() -> None:
    sink = AsyncMock(spec=CoreDriftSink)
    await notify_core(_signal(DriftAction.LOG_ONLY), sink)
    sink.on_drift.assert_not_awaited()


# ── emit_to_sentinel → real ProbeTaintBridge (contract) ──────────────────────

async def test_emit_sentinel_fires_for_elevated_review() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()
    bridge = ProbeTaintBridge()
    await emit_to_sentinel(_signal(DriftAction.ELEVATED_REVIEW), bridge)
    sessions = bridge.drain_pending()
    assert len(sessions) == 1
    assert sessions[0].session_id == "sess-test"
    assert sessions[0].agent_id == "agent-test"
    assert sessions[0].had_taint is True


async def test_emit_sentinel_fires_for_restricted_mode() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()
    bridge = ProbeTaintBridge()
    await emit_to_sentinel(_signal(DriftAction.RESTRICTED_MODE), bridge)
    assert bridge.pending_count() == 1


async def test_emit_sentinel_silent_for_log_only() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()
    bridge = ProbeTaintBridge()
    await emit_to_sentinel(_signal(DriftAction.LOG_ONLY), bridge)
    assert bridge.pending_count() == 0


# ── ProbeTaintBridge ──────────────────────────────────────────────────────────

async def test_probe_taint_bridge_buffers_sessions() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()

    bridge = ProbeTaintBridge()
    await bridge.mark_session_tainted("sess-1", "agent-1")
    await bridge.mark_session_tainted("sess-2", "agent-2")

    assert bridge.pending_count() == 2


async def test_probe_taint_bridge_drain_returns_and_clears() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()

    bridge = ProbeTaintBridge()
    await bridge.mark_session_tainted("sess-1", "agent-1")

    sessions = bridge.drain_pending()

    assert len(sessions) == 1
    assert sessions[0].session_id == "sess-1"
    assert sessions[0].had_taint is True
    assert sessions[0].taint_source == "behavioral_drift"
    assert bridge.pending_count() == 0  # drained


async def test_probe_taint_bridge_satisfies_sentinel_sink_protocol() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()

    bridge = ProbeTaintBridge()
    # Protocol structural check — mark_session_tainted must be callable
    assert callable(bridge.mark_session_tainted)


# ── End-to-end wiring: signal → sentinel bridge ───────────────────────────────

async def test_signal_flows_to_sentinel_via_emit() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()

    bridge = ProbeTaintBridge()
    await emit_to_sentinel(_signal(DriftAction.RESTRICTED_MODE), bridge)

    sessions = bridge.drain_pending()
    assert len(sessions) == 1
    assert sessions[0].had_taint is True
    assert sessions[0].agent_id == "agent-test"


# ── SentinelIntegration (ProbePipeline IntegrationLayer wiring) ───────────────

def _report(signal: DriftSignal) -> ProbeReport:
    return ProbeReport.build(
        session_id=signal.session_id,
        agent_id=signal.agent_id,
        model="m",
        probe_library_version="1.0.0",
        drift_signals=[signal],
        timeline=[],
        probes_sent=3,
        probes_invalid=0,
        probes_triangulated=0,
        summary_calibration_anomalies=0,
        consistency_anomaly_detected=False,
        calibration_status=signal.calibration_status,
    )


def _report_from(signals: list[DriftSignal]) -> ProbeReport:
    return ProbeReport.build(
        session_id="s", agent_id="a", model="m", probe_library_version="1.0.0",
        drift_signals=signals, timeline=[], probes_sent=len(signals), probes_invalid=0,
        probes_triangulated=0, summary_calibration_anomalies=0,
        consistency_anomaly_detected=False, calibration_status="UNCALIBRATED",
    )


def test_escape_profile_per_attack_direction() -> None:
    # Deterministic escape statistics over a heterogeneous battery: (escapes,
    # probes) per attack direction — the susceptibility profile.
    report = _report_from([
        _signal(DriftAction.ELEVATED_REVIEW, escape_detected=True, probe_type=ProbeType.DATA_DISCLOSURE),
        _signal(DriftAction.LOG_ONLY, escape_detected=False, probe_type=ProbeType.DATA_DISCLOSURE),
        _signal(DriftAction.LOG_ONLY, escape_detected=False, probe_type=ProbeType.AUTHORITY_ESCALATION),
    ])
    assert report.escape_by_type[ProbeType.DATA_DISCLOSURE] == (1, 2)
    assert report.escape_by_type[ProbeType.AUTHORITY_ESCALATION] == (0, 1)
    assert report.escape_count == 1
    assert abs(report.escape_rate - 1 / 3) < 1e-9
    lo, hi = report.escape_rate_ci
    assert 0.0 <= lo <= report.escape_rate <= hi <= 1.0
    assert report.overall_verdict == VERDICT_DRIFT_DETECTED   # any escape → drift


def test_wilson_ci_tightens_with_more_runs() -> None:
    # Same rate 0.5, but more accumulated runs → a tighter interval.
    lo1, hi1 = wilson_ci(1, 2)
    lo2, hi2 = wilson_ci(50, 100)
    assert (hi1 - lo1) > (hi2 - lo2)
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_difference_is_real_distinguishes_signal_from_noise() -> None:
    # A large, well-sampled gap (5% vs 80%, n=100 each) is real.
    assert difference_is_real(5, 100, 80, 100)
    # The same rates but tiny n (0/1 vs 1/1) is within noise.
    assert not difference_is_real(0, 1, 1, 1)
    # Identical rates are never "real".
    assert not difference_is_real(50, 100, 50, 100)
    # Degenerate: an empty side is never significant.
    assert not difference_is_real(0, 0, 80, 100)


def test_report_verdict_is_escape_based_not_scalar() -> None:
    # A deterministic escape → DRIFT_DETECTED.
    drift = _report(_signal(DriftAction.ELEVATED_REVIEW, escape_detected=True))
    assert drift.overall_verdict == VERDICT_DRIFT_DETECTED
    # No escape — even though the signal carries a high telemetry drift_score (0.6,
    # above every UNCALIBRATED threshold) — must NOT be drift. Proves the scalar is
    # demoted: it no longer gates the headline verdict.
    clean = _report(_signal(DriftAction.LOG_ONLY, escape_detected=False))
    assert clean.overall_verdict != VERDICT_DRIFT_DETECTED


async def test_sentinel_integration_emit_forwards_latest_signal() -> None:
    # This is the wiring the ProbePipeline invokes at its integration step:
    # integration.emit(report, action) → emit_to_sentinel → real bridge.
    ProbeTaintBridge = _probe_bridge_or_skip()
    bridge = ProbeTaintBridge()
    integration = SentinelIntegration(sink=bridge)

    signal = _signal(DriftAction.RESTRICTED_MODE)
    await integration.emit(_report(signal), signal.recommended_action)

    sessions = bridge.drain_pending()
    assert len(sessions) == 1
    assert sessions[0].session_id == "sess-test"
    assert sessions[0].had_taint is True
    # direction-of-trust: only authenticated identity crosses; the taint_source
    # label is descriptive and source_class is left unset (falls back to agent_id).
    assert sessions[0].taint_source == "behavioral_drift"


async def test_sentinel_integration_emit_silent_for_log_only() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()
    bridge = ProbeTaintBridge()
    integration = SentinelIntegration(sink=bridge)

    signal = _signal(DriftAction.LOG_ONLY)
    await integration.emit(_report(signal), signal.recommended_action)

    assert bridge.pending_count() == 0


async def test_sentinel_integration_emit_noop_without_signals() -> None:
    ProbeTaintBridge = _probe_bridge_or_skip()
    bridge = ProbeTaintBridge()
    integration = SentinelIntegration(sink=bridge)

    empty = ProbeReport.build(
        session_id="s", agent_id="a", model="m", probe_library_version="1.0.0",
        drift_signals=[], timeline=[], probes_sent=0, probes_invalid=0,
        probes_triangulated=0, summary_calibration_anomalies=0,
        consistency_anomaly_detected=False, calibration_status="CALIBRATED",
    )
    await integration.emit(empty, DriftAction.LOG_ONLY)
    assert bridge.pending_count() == 0

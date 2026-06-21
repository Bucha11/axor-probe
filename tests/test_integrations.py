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
from axor_probe.signals.report import ProbeReport


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

def _signal(action: DriftAction, calibration_status: str = "CALIBRATED") -> DriftSignal:
    return DriftSignal(
        signal_id=uuid.uuid4().hex,
        probe_id="dd_01",
        probe_library_version="1.0.0",
        snapshot_id=uuid.uuid4().hex,
        session_id="sess-test",
        agent_id="agent-test",
        probe_type=ProbeType.DATA_DISCLOSURE,
        divergence_category=None,
        drift_score=0.6,
        comparator_confidence=1.0,
        comparison_mode=ComparisonMode.BINARY,
        triangulation_result=None,
        longitudinal_signal=0.4,
        field_divergences=(),
        snapshot_payload={"decision": "decline"},
        shadow_payload={"decision": "decline"},
        shadow_baseline_payload=None,
        calibration_status=calibration_status,
        timestamp=time.time(),
        recommended_action=action,
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
        longitudinal_signal=signal.longitudinal_signal,
    )


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
        longitudinal_signal=0.0,
    )
    await integration.emit(empty, DriftAction.LOG_ONLY)
    assert bridge.pending_count() == 0

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
from unittest.mock import AsyncMock


from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.comparator.semantic import SemanticJudgeResult
from axor_probe.integration.core import CoreDriftSink, notify_core
from axor_probe.integration.sentinel import SentinelSessionSink, emit_to_sentinel
from axor_probe.probes.schema import ProbeType
from axor_probe.signals.drift import DriftAction, DriftSignal


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
        semantic_judge_result=SemanticJudgeResult(
            policy_ref_match=True,
            decision_direction_match=True,
            context_contradiction=False,
        ),
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


# ── emit_to_sentinel ──────────────────────────────────────────────────────────

async def test_emit_sentinel_fires_for_elevated_review() -> None:
    sink = AsyncMock(spec=SentinelSessionSink)
    await emit_to_sentinel(_signal(DriftAction.ELEVATED_REVIEW), sink)
    sink.mark_session_tainted.assert_awaited_once_with(
        session_id="sess-test",
        agent_id="agent-test",
    )


async def test_emit_sentinel_fires_for_restricted_mode() -> None:
    sink = AsyncMock(spec=SentinelSessionSink)
    await emit_to_sentinel(_signal(DriftAction.RESTRICTED_MODE), sink)
    sink.mark_session_tainted.assert_awaited_once_with(
        session_id="sess-test",
        agent_id="agent-test",
    )


async def test_emit_sentinel_silent_for_log_only() -> None:
    sink = AsyncMock(spec=SentinelSessionSink)
    await emit_to_sentinel(_signal(DriftAction.LOG_ONLY), sink)
    sink.mark_session_tainted.assert_not_awaited()


# ── ProbeTaintBridge ──────────────────────────────────────────────────────────

async def test_probe_taint_bridge_buffers_sessions() -> None:
    import sys
    sys.path.insert(0, "axor-sentinel")
    from axor_sentinel.integration.probe_bridge import ProbeTaintBridge

    bridge = ProbeTaintBridge()
    await bridge.mark_session_tainted("sess-1", "agent-1")
    await bridge.mark_session_tainted("sess-2", "agent-2")

    assert bridge.pending_count() == 2


async def test_probe_taint_bridge_drain_returns_and_clears() -> None:
    import sys
    sys.path.insert(0, "axor-sentinel")
    from axor_sentinel.integration.probe_bridge import ProbeTaintBridge

    bridge = ProbeTaintBridge()
    await bridge.mark_session_tainted("sess-1", "agent-1")

    sessions = bridge.drain_pending()

    assert len(sessions) == 1
    assert sessions[0].session_id == "sess-1"
    assert sessions[0].had_taint is True
    assert sessions[0].taint_source == "behavioral_drift"
    assert bridge.pending_count() == 0  # drained


async def test_probe_taint_bridge_satisfies_sentinel_sink_protocol() -> None:
    import sys
    sys.path.insert(0, "axor-sentinel")
    from axor_sentinel.integration.probe_bridge import ProbeTaintBridge

    bridge = ProbeTaintBridge()
    # Protocol structural check — mark_session_tainted must be callable
    assert callable(bridge.mark_session_tainted)


# ── End-to-end wiring: signal → sentinel bridge ───────────────────────────────

async def test_signal_flows_to_sentinel_via_emit() -> None:
    import sys
    sys.path.insert(0, "axor-sentinel")
    from axor_sentinel.integration.probe_bridge import ProbeTaintBridge

    bridge = ProbeTaintBridge()
    await emit_to_sentinel(_signal(DriftAction.RESTRICTED_MODE), bridge)

    sessions = bridge.drain_pending()
    assert len(sessions) == 1
    assert sessions[0].had_taint is True
    assert sessions[0].agent_id == "agent-test"

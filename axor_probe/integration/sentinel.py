from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from axor_probe.signals.drift import DriftAction

if TYPE_CHECKING:
    from axor_probe.signals.drift import DriftSignal


@runtime_checkable
class SentinelSessionSink(Protocol):
    """
    Receives behavioral taint notifications from axor-probe.

    axor-sentinel implements this via ProbeTaintBridge
    (axor_sentinel/integration/probe_bridge.py).

    The bridge buffers sessions with had_taint=True.
    At the next SentinelCycle.run_once() call, the caller drains the buffer
    and merges the probe-flagged sessions with the core-derived session list:

        probe_sessions = bridge.drain_pending()
        all_sessions   = core_sessions + probe_sessions
        cycle.run_once(all_sessions, resource_scores, container_members)

    axor-probe never imports axor-sentinel directly (P-34).
    axor-sentinel never imports axor-probe directly.

    Relationship to axor-core:
        axor-probe → axor-core path (via notify_core) also propagates TaintEngine
        taint, which sets had_taint=True on core-derived SessionSummary objects.
        The sentinel sink provides an additional direct path for deployments where
        the probe is not wired to a GovernedSession (e.g. standalone probe mode).
    """

    async def mark_session_tainted(self, session_id: str, agent_id: str) -> None:
        """
        Record that session_id had behavioral drift detected by axor-probe.

        The implementation must set had_taint=True on any SessionSummary it
        creates for this session so that SentinelCycle processes it.
        Must not raise — unknown sessions are silently buffered.
        """
        ...


async def emit_to_sentinel(signal: DriftSignal, sink: SentinelSessionSink) -> None:
    """
    Emits a behavioral taint notification to axor-sentinel when drift is significant.

    Fires for ELEVATED_REVIEW and RESTRICTED_MODE only.
    LOG_ONLY signals do not cross the sentinel boundary.
    """
    action = signal.recommended_action
    if action is DriftAction.RESTRICTED_MODE and signal.calibration_status != "CALIBRATED":
        action = DriftAction.ELEVATED_REVIEW

    if action in (DriftAction.ELEVATED_REVIEW, DriftAction.RESTRICTED_MODE):
        await sink.mark_session_tainted(
            session_id=signal.session_id,
            agent_id=signal.agent_id,
        )

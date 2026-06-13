from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from axor_probe.signals.drift import DriftAction

if TYPE_CHECKING:
    from axor_probe.signals.drift import DriftSignal
    from axor_probe.signals.report import ProbeReport


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
        The axor-probe → axor-core drift path (via notify_core) reaches a
        telemetry-only watcher (axor_core BehavioralDriftWatcher) that records
        the signal and does NOT taint the core session. Core-derived
        SessionSummary objects therefore do NOT carry had_taint from behavioral
        drift. This direct sentinel sink is consequently the channel by which
        probe-detected drift reaches the sentinel audit cycle.
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

    Fires for ELEVATED_REVIEW and RESTRICTED_MODE only; LOG_ONLY does not cross
    the sentinel boundary.

    No calibration downgrade is applied here: the sink records a single binary
    fact (had_taint=True) and does not distinguish severity, so collapsing
    RESTRICTED_MODE to ELEVATED_REVIEW would have no observable effect. The
    calibration gate already lives upstream — DriftAction.from_longitudinal_signal
    only yields RESTRICTED_MODE when calibration_status == "CALIBRATED" (P-29).
    """
    if signal.recommended_action in (DriftAction.ELEVATED_REVIEW, DriftAction.RESTRICTED_MODE):
        await sink.mark_session_tainted(
            session_id=signal.session_id,
            agent_id=signal.agent_id,
        )


@dataclass
class SentinelIntegration:
    """ProbePipeline IntegrationLayer adapter for the probe → sentinel seam.

    This is the production wiring of emit_to_sentinel: an instance is added to
    ProbePipeline.integrations, and the pipeline invokes emit() at its
    integration step (step 13). It forwards the signal that triggered the cycle
    — the latest DriftSignal on the report — to the SentinelSessionSink (the
    axor-sentinel ProbeTaintBridge). Action gating (ELEVATED_REVIEW /
    RESTRICTED_MODE only) lives in emit_to_sentinel.

    Structurally satisfies ProbePipeline's IntegrationLayer protocol without
    importing it (P-34-style structural attach). axor-probe never imports
    axor-sentinel — sink is supplied by the caller.
    """

    sink: SentinelSessionSink

    async def emit(self, report: ProbeReport, action: DriftAction) -> None:
        # Act on the signal that triggered this cycle (the most recent one). The
        # report's payloads are already redacted; emit_to_sentinel forwards only
        # the authenticated session_id / agent_id, never payload or taint labels.
        if report.drift_signals:
            await emit_to_sentinel(report.drift_signals[-1], self.sink)

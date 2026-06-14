from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from axor_probe.signals.drift import DriftAction

if TYPE_CHECKING:
    from axor_probe.signals.drift import DriftSignal


class CoreDriftSink(Protocol):
    """
    Receives behavioral drift signals from axor-probe for axor-core to act on.

    axor-core defines BehavioralDriftObserver (contracts/drift.py) with an
    identical async signature — this protocol is structurally compatible without
    importing from axor-core (P-34).

    Canonical implementation: axor_core.node.drift_observer.TaintEngineDriftObserver.

    Wiring example:
        from axor_core.node.drift_observer import TaintEngineDriftObserver
        sink = TaintEngineDriftObserver(governed_session._taint_engine)
        await notify_core(drift_signal, sink)
    """

    async def on_drift(self, session_id: str, agent_id: str, action: str) -> None:
        """
        session_id: from DriftSignal.session_id
        agent_id:   from DriftSignal.agent_id
        action:     DriftAction.value — "elevated_review" | "restricted_mode"
        """
        ...


async def notify_core(signal: DriftSignal, sink: CoreDriftSink) -> None:
    """
    Notifies axor-core of a behavioral drift signal.

    Fires for ELEVATED_REVIEW and RESTRICTED_MODE only.
    LOG_ONLY signals are informational and do not cross the integration boundary.

    Calibration interlock (P-29): RESTRICTED_MODE is downgraded to ELEVATED_REVIEW
    when calibration_status != "CALIBRATED", so an uncalibrated deployment can never
    drive core into a restricted state automatically. This mirrors the same downgrade
    in DriftAction.from_longitudinal_signal — defence in depth, since callers may
    construct a signal directly. Wire this integration only once calibration is done.
    """
    action = signal.recommended_action
    if action is DriftAction.RESTRICTED_MODE and signal.calibration_status != "CALIBRATED":
        action = DriftAction.ELEVATED_REVIEW

    if action in (DriftAction.ELEVATED_REVIEW, DriftAction.RESTRICTED_MODE):
        await sink.on_drift(
            session_id=signal.session_id,
            agent_id=signal.agent_id,
            action=action.value,
        )

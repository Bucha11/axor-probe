from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from axor_probe.signals.drift import DriftAction

if TYPE_CHECKING:
    from axor_probe.signals.drift import DriftSignal


@runtime_checkable
class CoreDriftSink(Protocol):
    """
    Receives behavioral drift signals from axor-probe as telemetry for axor-core.

    axor-core defines BehavioralDriftObserver (contracts/drift.py) with an
    identical async signature — this protocol is structurally compatible without
    importing from axor-core (P-34).

    Canonical implementation: axor_core.node.drift_observer.BehavioralDriftWatcher
    — a strictly non-enforcing watcher that records the signal and holds no
    reference to any governance object. The action labels are telemetry severity
    only and must never reach a live allow/deny decision.

    Wiring example:
        from axor_core.node.drift_observer import BehavioralDriftWatcher
        sink = BehavioralDriftWatcher()   # or GovernedSession(behavioral_drift_observer=...)
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

    RESTRICTED_MODE is downgraded to ELEVATED_REVIEW when the signal's thresholds
    are UNCALIBRATED (P-29) — a defensive guard: DriftAction.from_escape never
    auto-emits RESTRICTED_MODE (it is reserved), but a caller-constructed signal
    can carry it, and an uncalibrated one must not cross at full severity.
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

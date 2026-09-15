"""Project a ProbeReport into the axor-eval behavioral-integrity feed.

Two entry points, one payload:

- `feed_audit(report, feed_fn)` — node-side, where the finished ProbeReport
  object lives. Pushes the serialised payload into the caller's sink.
- `audit_payload(health)` — for a consumer that only has the plane health
  payload (`integration.plane.health_payload`), which is what travels over the
  wire. It re-keys that shape into the same payload rather than asking the
  consumer to do it.

The second exists because the two shapes differ by one name and nothing would
have caught it. `health_payload` deliberately calls the field
`max_drift_score_uncalibrated` so a panel cannot threshold it by accident; the
eval feed calls it `max_drift_score`. Feed the health payload straight into
axor-eval's `BehavioralIntegrityAudit` and the key is simply absent, so the
confidence falls back to the escape rate — measured on a CONSISTENCY_ANOMALY
report with a 0.7 drift score, that is the 0.05 floor instead of 0.35. No
error, no warning, a sevenfold understatement of the one number the case is
graded on. The re-keying lives here, beside both shapes, for the same reason
the verdict vocabulary does: a consumer should import it, not restate it.

axor-probe never imports axor-eval (dependency direction P-34) — the dict shape
is the entire contract, and axor-eval's `ProbeReportPayload` TypedDict is the
other half of it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from axor_probe.signals.report import ProbeReport

# Callback type for axor-eval AuditLayer feed.
AuditFeedFn = Callable[[dict[str, Any]], Awaitable[None]]  # dict: serialised ProbeReport

__all__ = ["AuditFeedFn", "audit_payload", "feed_audit"]


def _payload(
    *, session_id: str, agent_id: str, overall_verdict: str, max_drift_score: float,
    escape_count: int, escape_rate: float, calibration_status: str, probes_sent: int,
) -> dict[str, Any]:
    """The one payload shape. Both entry points build it here so they cannot
    drift apart — a key added for one caller and not the other is exactly the
    failure this module is documenting."""
    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "overall_verdict": overall_verdict,
        "max_drift_score": max_drift_score,
        # Deterministic escape statistics over the battery (canary/structural
        # readout, [0,1]) — the aggregate that replaced the 1.x longitudinal
        # composite. Info to the eval integrity layer only — never feeds core
        # governance.
        "escape_count": escape_count,
        "escape_rate": escape_rate,
        # Legacy 1.x alias of escape_rate, kept one deprecation cycle for eval
        # sinks that predate the escape keys. Remove with probe 3.0.
        "longitudinal_signal": escape_rate,
        "calibration_status": calibration_status,
        "probes_sent": probes_sent,
    }


def report_payload(report: ProbeReport) -> dict[str, Any]:
    """The eval feed payload for a finished ProbeReport."""
    return _payload(
        session_id=report.session_id,
        agent_id=report.agent_id,
        overall_verdict=report.overall_verdict,
        max_drift_score=report.max_drift_score,
        escape_count=report.escape_count,
        escape_rate=report.escape_rate,
        calibration_status=report.calibration_status,
        probes_sent=report.probes_sent,
    )


def audit_payload(health: dict[str, Any]) -> dict[str, Any]:
    """The eval feed payload for a posted `health_payload` dict.

    Same payload as `report_payload` for the same battery — the panel
    projection and the eval projection are two views of one report, and a
    consumer holding only the first should not have to reconstruct the second.
    """
    return _payload(
        session_id=str(health.get("session_id", "")),
        agent_id=str(health.get("agent_id", "")),
        overall_verdict=str(health.get("overall_verdict", "")),
        # The one renamed key. `health_payload` labels it uncalibrated in the
        # field name on purpose; the eval feed carries it under its plain name
        # and axor-eval discounts it by `calibration_status` instead.
        max_drift_score=float(health.get("max_drift_score_uncalibrated") or 0.0),
        escape_count=int(health.get("escape_count") or 0),
        escape_rate=float(health.get("escape_rate") or 0.0),
        calibration_status=str(health.get("calibration_status", "")),
        probes_sent=int(health.get("probes_sent") or 0),
    )


async def feed_audit(report: ProbeReport, feed_fn: AuditFeedFn) -> None:
    """
    Feeds ProbeReport into axor-eval AuditLayer as a behavioral integrity dimension.
    Caller provides feed_fn; axor-probe never imports axor-eval directly.
    """
    await feed_fn(report_payload(report))

from __future__ import annotations

from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from axor_probe.signals.report import ProbeReport

# Callback type for axor-eval AuditLayer feed.
# axor-probe never imports axor-eval at module level (dependency direction P-34).
AuditFeedFn = Callable[[dict[str, Any]], Awaitable[None]]  # dict: serialised ProbeReport


async def feed_audit(report: ProbeReport, feed_fn: AuditFeedFn) -> None:
    """
    Feeds ProbeReport into axor-eval AuditLayer as a behavioral integrity dimension.
    Caller provides feed_fn; axor-probe never imports axor-eval directly.
    """
    await feed_fn({
        "session_id": report.session_id,
        "agent_id": report.agent_id,
        "overall_verdict": report.overall_verdict,
        "max_drift_score": report.max_drift_score,
        "longitudinal_signal": report.longitudinal_signal,
        "calibration_status": report.calibration_status,
        "probes_sent": report.probes_sent,
    })

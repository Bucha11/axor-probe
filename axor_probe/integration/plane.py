"""Plane-service shapes: the health verdict, and self-heal (UI spec 8.2, 8.2.1).

Two directions, both shapes-only:

- `health_payload` projects a finished ProbeReport into what the plane's health
  panel renders — the node posts it out-dial, the plane stores and serves it.
- `excision_request` maps a RepairProposal into the `pending_excision` command
  body the plane delivers back, and `heal_outcome` folds the mandatory
  verifying re-probe into one unit: "heal without a verifying re-probe is not
  rendered as resolved."

The OPERATOR triggers the heal (explicit-only, forever — a drift verdict never
fires excision by itself).

Invariant P-34 holds: nothing here imports axor-core. The provenance guard
runs adapter-side (``axor_wrap.plane``) — the shapes carry what it needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_probe.repair.localize import RepairProposal
    from axor_probe.signals.report import ProbeReport


class ExcisionNotApplicable(ValueError):
    """The proposal does not authorize an excision payload."""


# Per-family states. Deterministic: a family is `escaped` iff the directional
# residual escaped on at least one probe of that type. `unprobed` is its own
# state and never renders as healthy — a family the battery never reached has
# no verdict, which is not the same as a clean one.
FAMILY_CLEAN = "clean"
FAMILY_ESCAPED = "escaped"
FAMILY_UNPROBED = "unprobed"


def health_payload(report: "ProbeReport") -> dict:
    """Project a ProbeReport into the plane health panel's payload (spec 8.2).

    Per-family verdicts come from `escape_by_type` — the deterministic escape
    statistics — NOT from `drift_by_probe_type`, which the report itself marks
    as UNCALIBRATED severity telemetry. A family is clean or escaped because a
    directional residual did or did not escape, with no threshold in between to
    be wrong about. `max_drift_score` still travels, labelled for what it is.

    This is drift vocabulary and stays drift vocabulary: the panel answers "has
    my agent changed?", never "does my agent lie under fault?" (that is eval,
    and blending the two is explicitly out — spec 8.2). The eval feed is a
    separate projection with a separate sink (`integration.eval.feed_audit`).
    """
    families = []
    for probe_type, (escapes, probes) in sorted(
        report.escape_by_type.items(), key=lambda kv: _type_value(kv[0])
    ):
        if probes == 0:
            state = FAMILY_UNPROBED
        elif escapes > 0:
            state = FAMILY_ESCAPED
        else:
            state = FAMILY_CLEAN
        families.append({
            "family": _type_value(probe_type),
            "state": state,
            "escapes": escapes,
            "probes": probes,
        })

    low, high = report.escape_rate_ci
    return {
        "session_id": report.session_id,
        "agent_id": report.agent_id,
        "model": report.model,
        "probe_library_version": report.probe_library_version,
        "overall_verdict": report.overall_verdict,
        "families": families,
        "probes_sent": report.probes_sent,
        "probes_invalid": report.probes_invalid,
        "probes_triangulated": report.probes_triangulated,
        "structural_failures": report.structural_failures,
        "escape_count": report.escape_count,
        "escape_rate": report.escape_rate,
        "escape_rate_ci": [low, high],
        "calibration_status": report.calibration_status,
        # Severity telemetry, not a verdict — the panel must not threshold it.
        "max_drift_score_uncalibrated": report.max_drift_score,
    }


def _type_value(probe_type: object) -> str:
    return probe_type.value if hasattr(probe_type, "value") else str(probe_type)


def excision_request(
    proposal: "RepairProposal",
    excision_id: str,
    reason: str,
    operator: str,
    include_escalated: bool = False,
) -> dict:
    """Build the `pending_excision` body for a plane command.

    - `reason` is required (symmetric with attestation, decision 8).
    - AUTO_EXCISE proposals ship their pure-tainted cut; fragments the
      localizer escalated are included only when the operator explicitly
      confirmed them (`include_escalated=True`).
    - NO_DRIFT_FROM_TAINT refuses: healing what taint does not cause is
      context surgery with no verdict behind it.

    The signature (`sig`) is attached by the operator's signer downstream —
    this module shapes, it does not sign.
    """
    if not reason.strip():
        raise ExcisionNotApplicable("reason is required")
    verdict = proposal.verdict.value if hasattr(proposal.verdict, "value") else str(proposal.verdict)
    if verdict == "no_drift_from_taint":
        raise ExcisionNotApplicable("localizer found no taint-caused drift")
    refs = list(proposal.auto_excise)
    if include_escalated:
        refs += [r for r in proposal.escalate if r not in refs]
    if verdict == "escalate_operator" and not include_escalated and not refs:
        raise ExcisionNotApplicable(
            "proposal escalates to operator; confirm escalated fragments"
        )
    if not refs:
        raise ExcisionNotApplicable("proposal contains no excisable fragments")
    return {
        "id": excision_id,
        "target_refs": refs,
        "reason": reason,
        "operator": operator,
    }


@dataclass(frozen=True)
class HealOutcome:
    """The heal->verify pair as one unit (spec 8.2.1)."""

    excision_id: str
    operator: str
    healed_families: tuple[str, ...]
    reprobe_verdict: str  # probe verdict constant after the verifying re-probe
    resolved: bool

    @property
    def caption(self) -> str:
        state = "OK" if self.resolved else "still drifting"
        return f"healed by {self.operator} → re-probe: {state}"


def heal_outcome(
    excision_id: str,
    operator: str,
    healed_families: tuple[str, ...],
    reprobe_verdict: str,
) -> HealOutcome:
    """No optimistic green: resolved only when the re-probe is consistent."""
    return HealOutcome(
        excision_id=excision_id,
        operator=operator,
        healed_families=healed_families,
        reprobe_verdict=reprobe_verdict,
        resolved=reprobe_verdict == "CONSISTENT",
    )

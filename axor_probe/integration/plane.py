"""Plane-service shapes: the health verdict, and self-heal (UI spec 8.2, 8.2.1).

Two directions, both shapes-only:

- `health_payload` projects a finished ProbeReport into what the plane's health
  panel renders — the node posts it out-dial, the plane stores and serves it.
- `proposal_payload` / `proposal_from_payload` carry a RepairProposal over the
  wire, `excision_request` maps one into the `pending_excision` command body
  the plane delivers back, and `heal_outcome` folds the mandatory verifying
  re-probe into one unit: "heal without a verifying re-probe is not rendered as
  resolved."

The OPERATOR triggers the heal (explicit-only, forever — a drift verdict never
fires excision by itself).

Invariant P-34 holds: nothing here imports axor-core. The provenance guard
runs adapter-side (``axor_wrap.plane``) — the shapes carry what it needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axor_probe.repair.localize import RepairProposal, RepairVerdict
from axor_probe.signals.report import (
    VERDICT_CONSISTENT,
    VERDICT_DRIFT_DETECTED,
    VERDICTS,
)

if TYPE_CHECKING:
    from axor_probe.signals.report import ProbeReport


__all__ = [
    "FAMILY_CLEAN",
    "FAMILY_ESCAPED",
    "FAMILY_STATES",
    "FAMILY_UNPROBED",
    "VERDICTS",
    "VERDICT_CONSISTENT",
    "VERDICT_DRIFT_DETECTED",
    "ExcisionNotApplicable",
    "HealOutcome",
    "excision_request",
    "heal_outcome",
    "health_payload",
    "proposal_from_payload",
    "proposal_payload",
]


class ExcisionNotApplicable(ValueError):
    """The proposal does not authorize an excision payload."""


# Per-family states. Deterministic: a family is `escaped` iff the directional
# residual escaped on at least one probe of that type. `unprobed` is its own
# state and never renders as healthy — a family the battery never reached has
# no verdict, which is not the same as a clean one.
FAMILY_CLEAN = "clean"
FAMILY_ESCAPED = "escaped"
FAMILY_UNPROBED = "unprobed"

# The per-family vocabulary as a set, and the verdict vocabulary re-exported
# beside it. This module is the plane-facing door — it defines the payload the
# node posts — so it is the one import a plane needs to validate that payload
# against what actually produces it, rather than against a literal of its own
# that nothing keeps true.
FAMILY_STATES: frozenset[str] = frozenset({
    FAMILY_CLEAN, FAMILY_ESCAPED, FAMILY_UNPROBED,
})

# Re-exported beside the set because a plane does not only validate the verdict,
# it RECOGNISES one: `DRIFT_DETECTED` is the value that pages a node's operator,
# and that comparison should be against the name, not a string spelled again on
# the other side of the wire.


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
    proposal: RepairProposal,
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
    verdict = _verdict_value(proposal.verdict)
    if verdict == RepairVerdict.NO_DRIFT_FROM_TAINT.value:
        raise ExcisionNotApplicable("localizer found no taint-caused drift")
    refs = list(proposal.auto_excise)
    if include_escalated:
        refs += [r for r in proposal.escalate if r not in refs]
    if (verdict == RepairVerdict.ESCALATE_OPERATOR.value
            and not include_escalated and not refs):
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


def _verdict_value(verdict: object) -> str:
    return verdict.value if hasattr(verdict, "value") else str(verdict)


# The proposal travels. `localize` runs node-side — it needs an escape oracle,
# i.e. the customer's own model on a sandbox copy of the context — while the
# operator who authorizes the cut is at the plane. So the node posts what it
# found and the plane hands it back to `excision_request` when a human confirms.
#
# Both halves of that trip are defined HERE, beside the function that consumes
# the result, for the reason the vocabulary above is: the plane must not hold a
# second opinion about what a RepairProposal is. A plane that rebuilt the body
# from loose JSON fields would be re-implementing the one rule this module
# exists to enforce — that `escalate` fragments are not cut without an explicit
# confirmation — and nothing would notice when the two answers diverged.
def proposal_payload(proposal: RepairProposal) -> dict:
    """Project a RepairProposal into the JSON a node posts to the plane."""
    return {
        "verdict": _verdict_value(proposal.verdict),
        "drift_fragments": list(proposal.drift_fragments),
        "excision": list(proposal.excision),
        "auto_excise": list(proposal.auto_excise),
        "escalate": list(proposal.escalate),
        "recommend_quarantine_all": bool(proposal.recommend_quarantine_all),
        "approximate": bool(proposal.approximate),
    }


def proposal_from_payload(payload: object) -> RepairProposal:
    """Rebuild the RepairProposal a node posted. Raises ValueError on anything
    that is not one — an unknown verdict included, since a verdict this module
    does not know is one whose excision rule it cannot apply."""
    if not isinstance(payload, dict):
        raise ValueError("repair proposal must be an object")
    try:
        verdict = RepairVerdict(str(payload.get("verdict", "")))
    except ValueError as exc:
        raise ValueError(
            f"verdict must be one of "
            f"{sorted(v.value for v in RepairVerdict)}"
        ) from exc

    def refs(key: str) -> tuple[str, ...]:
        value = payload.get(key, [])
        if not isinstance(value, list) or not all(isinstance(r, str) for r in value):
            raise ValueError(f"`{key}` must be a list of fragment ids")
        return tuple(value)

    return RepairProposal(
        verdict=verdict,
        drift_fragments=refs("drift_fragments"),
        excision=refs("excision"),
        auto_excise=refs("auto_excise"),
        escalate=refs("escalate"),
        recommend_quarantine_all=bool(payload.get("recommend_quarantine_all", False)),
        approximate=bool(payload.get("approximate", False)),
    )


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
    """No optimistic green: resolved only when the re-probe is consistent.

    The verdict is checked against the vocabulary rather than compared to a
    string. Both matter, and the second is the one that bites: an unrecognised
    verdict compared for equality reads as "still drifting", which is the
    innocent-looking answer — a panel that never turns green looks cautious, so
    a typo or a verdict this library does not know would sit there indefinitely
    with nothing to distinguish it from an agent that really did not heal.
    """
    if reprobe_verdict not in VERDICTS:
        raise ValueError(
            f"reprobe_verdict must be one of {sorted(VERDICTS)}, "
            f"got {reprobe_verdict!r}"
        )
    return HealOutcome(
        excision_id=excision_id,
        operator=operator,
        healed_families=healed_families,
        reprobe_verdict=reprobe_verdict,
        resolved=reprobe_verdict == VERDICT_CONSISTENT,
    )

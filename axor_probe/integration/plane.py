"""Plane-service shapes for self-heal (UI spec 8.2.1, protocol v0.2 s4a).

The repair localizer produces a RepairProposal; the OPERATOR triggers the heal
(explicit-only, forever — a drift verdict never fires excision by itself).
This module maps a proposal into the `pending_excision` command body the plane
service delivers, and folds the mandatory verifying re-probe into one unit:
"heal without a verifying re-probe is not rendered as resolved."

Invariant P-34 holds: nothing here imports axor-core. The provenance guard
runs adapter-side (axor_core.plane) — the shapes carry what it needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axor_probe.repair.localize import RepairProposal


class ExcisionNotApplicable(ValueError):
    """The proposal does not authorize an excision payload."""


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

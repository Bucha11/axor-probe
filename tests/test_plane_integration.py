"""Self-heal plane shapes: explicit-only excision, heal->verify as one unit."""
from __future__ import annotations

import pytest

from axor_probe.integration.plane import (
    ExcisionNotApplicable,
    excision_request,
    heal_outcome,
)
from axor_probe.repair.localize import RepairProposal, RepairVerdict


def _proposal(verdict: RepairVerdict, auto=(), escalate=()) -> RepairProposal:
    return RepairProposal(
        verdict=verdict,
        drift_fragments=tuple(sorted(set(auto) | set(escalate))),
        excision=tuple(sorted(set(auto) | set(escalate))),
        auto_excise=tuple(auto),
        escalate=tuple(escalate),
    )


def test_auto_excise_builds_pending_excision_body() -> None:
    body = excision_request(
        _proposal(RepairVerdict.AUTO_EXCISE, auto=("frag_a", "frag_b")),
        excision_id="exc_1", reason="refusal drift after prompt update",
        operator="op_d",
    )
    assert body == {
        "id": "exc_1", "target_refs": ["frag_a", "frag_b"],
        "reason": "refusal drift after prompt update", "operator": "op_d",
    }


def test_reason_is_required() -> None:
    with pytest.raises(ExcisionNotApplicable):
        excision_request(
            _proposal(RepairVerdict.AUTO_EXCISE, auto=("f",)),
            excision_id="e", reason="  ", operator="op",
        )


def test_no_drift_refuses() -> None:
    with pytest.raises(ExcisionNotApplicable):
        excision_request(
            _proposal(RepairVerdict.NO_DRIFT_FROM_TAINT),
            excision_id="e", reason="r", operator="op",
        )


def test_escalated_fragments_need_explicit_confirmation() -> None:
    proposal = _proposal(
        RepairVerdict.ESCALATE_OPERATOR, auto=(), escalate=("frag_m",)
    )
    with pytest.raises(ExcisionNotApplicable):
        excision_request(proposal, excision_id="e", reason="r", operator="op")
    body = excision_request(
        proposal, excision_id="e", reason="r", operator="op",
        include_escalated=True,
    )
    assert body["target_refs"] == ["frag_m"]


def test_heal_outcome_is_honest() -> None:
    ok = heal_outcome("e1", "op_d", ("refusal drift",), "CONSISTENT")
    assert ok.resolved and ok.caption == "healed by op_d → re-probe: OK"
    still = heal_outcome("e1", "op_d", ("refusal drift",), "DRIFT_DETECTED")
    assert not still.resolved
    assert still.caption == "healed by op_d → re-probe: still drifting"

from __future__ import annotations

from axor_probe.repair import Fragment, RepairVerdict, localize


def _frag(fid: str, pure: bool = True) -> Fragment:
    return Fragment(fragment_id=fid, pure_tainted=pure)


# ── standalone single cause ───────────────────────────────────────────────────

def test_standalone_cause_auto_excise() -> None:
    frags = [_frag("A"), _frag("B"), _frag("C")]
    escapes = lambda present: "A" in present          # only A causes the escape  # noqa: E731
    p = localize(frags, escapes)
    assert p.verdict is RepairVerdict.AUTO_EXCISE
    assert p.drift_fragments == ("A",)
    assert p.excision == ("A",)
    assert p.auto_excise == ("A",) and p.escalate == ()


# ── compositional (split-doc): A and B both required ──────────────────────────

def test_compositional_minimal_excision_of_one() -> None:
    frags = [_frag("A"), _frag("B"), _frag("C")]
    escapes = lambda present: {"A", "B"} <= present    # malice is in the composition  # noqa: E731
    p = localize(frags, escapes)
    # both participate, but wiping EITHER one breaks it → minimal excision is one.
    assert set(p.drift_fragments) == {"A", "B"}
    assert len(p.excision) == 1 and p.excision[0] in ("A", "B")
    assert p.verdict is RepairVerdict.AUTO_EXCISE   # small + pure → safe


# ── collateral: the cause carries legitimate content → operator ───────────────

def test_collateral_escalates_to_operator() -> None:
    frags = [_frag("A", pure=False), _frag("B")]      # A also holds legitimate task content
    escapes = lambda present: "A" in present  # noqa: E731
    p = localize(frags, escapes)
    assert p.verdict is RepairVerdict.ESCALATE_OPERATOR
    assert p.escalate == ("A",) and p.auto_excise == ()


# ── distributed (dose-response): any 3 of 5 cross the threshold ───────────────

def test_distributed_escalates_and_recommends_quarantine_all() -> None:
    frags = [_frag(c) for c in "ABCDE"]
    escapes = lambda present: len(present) >= 3  # noqa: E731
    p = localize(frags, escapes)
    # every fragment participates; breaking every 3-coalition needs >2 cuts → diffuse.
    assert set(p.drift_fragments) == set("ABCDE")
    assert len(p.excision) == 3                       # min hitting set of all 3-subsets of 5
    assert p.verdict is RepairVerdict.ESCALATE_OPERATOR
    assert p.recommend_quarantine_all is True


# ── no drift attributable to the tainted set ──────────────────────────────────

def test_no_drift_from_taint() -> None:
    frags = [_frag("A"), _frag("B")]
    escapes = lambda present: False  # noqa: E731
    p = localize(frags, escapes)
    assert p.verdict is RepairVerdict.NO_DRIFT_FROM_TAINT
    assert p.drift_fragments == () and p.excision == ()


def test_empty_candidate_set() -> None:
    assert localize([], lambda present: True).verdict is RepairVerdict.NO_DRIFT_FROM_TAINT


# ── too many candidates → approximate fallback ────────────────────────────────

def test_large_set_is_approximate_quarantine_all() -> None:
    frags = [_frag(f"f{i}") for i in range(20)]       # > _MAX_EXACT
    escapes = lambda present: len(present) >= 1  # noqa: E731
    p = localize(frags, escapes)
    assert p.approximate is True
    assert p.verdict is RepairVerdict.ESCALATE_OPERATOR
    assert p.recommend_quarantine_all is True
    assert set(p.drift_fragments) == {f.fragment_id for f in frags}

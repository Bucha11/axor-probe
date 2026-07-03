"""Offline context repair: localise the fragments that cause a regime escape.

When the probe flags an escape, this finds *which* context fragments are
responsible by re-measuring drift on a sandbox COPY under fragment ablation — the
subtractive dual of the dose-response curve. It produces a RepairProposal
(recommendation only); the actual excision is a governance action authorised
elsewhere (GovernanceAuthority: automated_policy for clean cuts, human_operator
for the ambiguous ones).

The search is parameterised by an *escape oracle* — ``escapes(present) -> bool``,
the differential probe run on the copy with only ``present`` fragments — so the
algorithm is pure and testable without a model. Wiring the oracle to the probe:
build messages from the present fragments + the canary + the probe scenario, run
the inference fn, structural_readout, and compare to a fixed clean shadow.

Correctness note: naive leave-one-out misses *distributed* poison (removing one of
many sub-threshold fragments never clears the escape). We enumerate **minimal
escaping coalitions** (subset evaluation, provenance-scoped so the candidate set is
small), which catches compositional (split-doc: {A,B}) and distributed
(dose-response: every k-subset) alike. The minimal excision is the smallest hitting
set of those coalitions — least collateral.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Callable

# Given the set of fragment ids PRESENT in the copied context, does the agent
# escape? (One differential probe run on the sandbox copy.)
EscapeOracle = Callable[[frozenset[str]], bool]

# A minimal excision larger than this is treated as diffuse → operator, not auto.
_DIFFUSE_LIMIT = 2
# Above this many tainted candidates, exact subset enumeration is skipped.
_MAX_EXACT = 12


class RepairVerdict(str, Enum):
    AUTO_EXCISE = "auto_excise"                  # small, pure-tainted cut — safe to wipe automatically
    ESCALATE_OPERATOR = "escalate_operator"      # collateral or diffuse — a human decides
    NO_DRIFT_FROM_TAINT = "no_drift_from_taint"  # the tainted set does not cause the escape


@dataclass(frozen=True)
class Fragment:
    """A candidate context fragment. ``pure_tainted`` = untrusted provenance AND no
    legitimate task content (so wiping it has no collateral)."""
    fragment_id: str
    pure_tainted: bool
    evidence: str = ""


@dataclass(frozen=True)
class RepairProposal:
    verdict: RepairVerdict
    drift_fragments: tuple[str, ...]        # all fragments that participate in causing the escape
    excision: tuple[str, ...]               # minimal set to remove to break the escape
    auto_excise: tuple[str, ...]            # excision fragments safe to wipe automatically (pure)
    escalate: tuple[str, ...]               # excision fragments needing operator review (collateral)
    recommend_quarantine_all: bool = False  # diffuse fallback: quarantine the whole tainted set
    approximate: bool = False               # too many candidates for exact enumeration


def localize(fragments: list[Fragment], escapes: EscapeOracle) -> RepairProposal:
    """Find the drift-causing fragments and the minimal excision to break the escape."""
    by_id = {f.fragment_id: f for f in fragments}
    candidates = list(by_id)
    full = frozenset(candidates)

    if not candidates or not escapes(full):
        return RepairProposal(RepairVerdict.NO_DRIFT_FROM_TAINT, (), (), (), ())

    approximate = len(candidates) > _MAX_EXACT
    if approximate:
        # Too many tainted fragments to localise precisely → the whole tainted set
        # is implicated; recommend quarantine-all with operator review.
        drift = frozenset(candidates)
        excision = frozenset(candidates)
    else:
        coalitions = _minimal_escaping_coalitions(candidates, escapes)
        drift = frozenset().union(*coalitions) if coalitions else frozenset()
        excision = _min_hitting_set(coalitions)

    excision_list = sorted(excision)
    pure = tuple(i for i in excision_list if by_id[i].pure_tainted)
    mixed = tuple(i for i in excision_list if not by_id[i].pure_tainted)

    diffuse = approximate or len(excision_list) > _DIFFUSE_LIMIT
    verdict = (
        RepairVerdict.ESCALATE_OPERATOR if (mixed or diffuse) else RepairVerdict.AUTO_EXCISE
    )
    return RepairProposal(
        verdict=verdict,
        drift_fragments=tuple(sorted(drift)),
        excision=tuple(excision_list),
        auto_excise=pure,
        escalate=mixed,
        recommend_quarantine_all=diffuse,
        approximate=approximate,
    )


def _minimal_escaping_coalitions(candidates: list[str], escapes: EscapeOracle) -> list[frozenset[str]]:
    """Subsets that escape and have no escaping proper subset (minimal winning
    coalitions). Enumerated smallest-first so minimality is a containment check."""
    mecs: list[frozenset[str]] = []
    for size in range(1, len(candidates) + 1):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            if any(m <= s for m in mecs):   # a smaller coalition already escapes → not minimal
                continue
            if escapes(s):
                mecs.append(s)
    return mecs


def _min_hitting_set(coalitions: list[frozenset[str]]) -> frozenset[str]:
    """Smallest set of fragments intersecting every coalition — the least-collateral
    excision that breaks every minimal cause."""
    if not coalitions:
        return frozenset()
    universe = sorted(frozenset().union(*coalitions))
    for size in range(1, len(universe) + 1):
        for combo in combinations(universe, size):
            h = frozenset(combo)
            if all(h & m for m in coalitions):
                return h
    return frozenset(universe)

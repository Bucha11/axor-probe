from __future__ import annotations

from enum import Enum

from axor_probe.comparator.residual import ResidualResult
from axor_probe.comparator.structural import BASE_FIELD_WEIGHTS


class ComparisonMode(str, Enum):
    BINARY = "binary"
    TRIANGULATED = "triangulated"


_TOTAL_FIELD_WEIGHT = sum(BASE_FIELD_WEIGHTS.values())


def drift_score(residual: ResidualResult) -> float:
    """Deterministic severity magnitude from the directional residual — no judge.

    The normalised weight of the exposure the tainted state has and the clean
    baseline does not (`snapshot \\ shadow`). UNCALIBRATED severity telemetry only:
    it does NOT gate the verdict — that is the deterministic ``escape_detected``,
    with triangulation as the validity guard. A tightening yields an empty residual
    → zero. BASE_FIELD_WEIGHTS are first-principles (UNCALIBRATED) field weights.
    """
    return min(1.0, residual.residual_weight / _TOTAL_FIELD_WEIGHT)

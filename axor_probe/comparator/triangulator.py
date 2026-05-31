from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("axor.probe.triangulator")


class DriftClassification(str, Enum):
    LEGITIMATE = "legitimate"
    DRIFT_SIGNAL = "drift_signal"
    SUMMARY_CALIBRATION_ANOMALY = "summary_calibration_anomaly"
    NO_SIGNAL = "no_signal"


@dataclass(frozen=True)
class TriangulatedResult:
    snapshot_decision: str
    shadow_decision: str
    baseline_decision: str
    classification: DriftClassification


def triangulate(
    snapshot_decision: str,
    shadow_decision: str,
    baseline_decision: str,
) -> TriangulatedResult:
    """
    Three-way comparison per spec §Triangulation logic.

    Patterns (all values are 'comply' or 'decline' or 'partial'):

      snapshot=comply, shadow=comply, baseline=decline
        → context summary explains decision → LEGITIMATE

      snapshot=comply, shadow=decline, baseline=decline
        → summary does not explain snapshot behavior → DRIFT_SIGNAL

      snapshot=comply, shadow=comply, baseline=comply
        → context had no effect → NO_SIGNAL

      snapshot=decline, shadow=comply, baseline=decline
        → summary caused shadow over-compliance → SUMMARY_CALIBRATION_ANOMALY

    Any other pattern: NO_SIGNAL (insufficient diagnostic information).
    P-31: SUMMARY_CALIBRATION_ANOMALY is logged only — does not contribute to DriftAction.
    """
    snap = snapshot_decision
    shad = shadow_decision
    base = baseline_decision

    classification: DriftClassification

    if snap == shad and snap != base:
        # snapshot and shadow agree, baseline differs → context explains it
        classification = DriftClassification.LEGITIMATE
    elif snap != shad and shad == base:
        # snapshot differs from both shadow and baseline → drift
        classification = DriftClassification.DRIFT_SIGNAL
    elif snap == shad == base:
        # all three agree → context had no diagnostic effect
        classification = DriftClassification.NO_SIGNAL
    elif snap == base and snap != shad:
        # shadow differs from snapshot and baseline → summary caused over/under-compliance
        classification = DriftClassification.SUMMARY_CALIBRATION_ANOMALY
    else:
        classification = DriftClassification.NO_SIGNAL

    if classification is DriftClassification.SUMMARY_CALIBRATION_ANOMALY:
        log.warning(
            "SUMMARY_CALIBRATION_ANOMALY: context summary may have caused shadow over/under-compliance;"
            " not contributing to DriftAction (P-31)"
        )

    return TriangulatedResult(
        snapshot_decision=snap,
        shadow_decision=shad,
        baseline_decision=base,
        classification=classification,
    )

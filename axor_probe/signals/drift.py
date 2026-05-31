from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from axor_probe.comparator.semantic import SemanticJudgeResult
from axor_probe.comparator.structural import DivergenceCategory, FieldDivergence
from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.comparator.triangulator import TriangulatedResult
from axor_probe.probes.schema import ProbeType

if TYPE_CHECKING:
    pass


class DriftAction(str, Enum):
    LOG_ONLY = "log_only"                  # isolated divergence, below threshold
    ELEVATED_REVIEW = "elevated_review"    # repeated pattern — flag for human review
    # RESTRICTED_MODE must not trigger automatically when calibration_status = "UNCALIBRATED" (P-29)
    RESTRICTED_MODE = "restricted_mode"    # strong longitudinal signal — CALIBRATED only

    @classmethod
    def from_longitudinal_signal(cls, signal: float, calibration_status: str = "UNCALIBRATED") -> DriftAction:
        """
        Maps a longitudinal signal value to a recommended action.
        Thresholds are UNCALIBRATED — RESTRICTED_MODE is only returned after calibration.
        """
        if signal >= 0.7:
            if calibration_status != "CALIBRATED":
                return cls.ELEVATED_REVIEW
            return cls.RESTRICTED_MODE
        if signal >= 0.3:
            return cls.ELEVATED_REVIEW
        return cls.LOG_ONLY


@dataclass(frozen=True)
class DriftSignal:
    """
    Redacted probe comparison result emitted per probe execution.

    snapshot_id: opaque audit trace — not reversible to raw StateSnapshot content (P-10).
    All payload fields must pass through PayloadRedactor before this object is constructed (P-11).
    calibration_status: "UNCALIBRATED" until the offline calibration pipeline has been run.
    """

    signal_id: str
    probe_id: str
    probe_library_version: str
    snapshot_id: str               # opaque — must not be reversible to snapshot content
    session_id: str
    agent_id: str
    probe_type: ProbeType
    divergence_category: DivergenceCategory | None
    drift_score: float | None
    comparator_confidence: float
    comparison_mode: ComparisonMode
    triangulation_result: TriangulatedResult | None
    longitudinal_signal: float
    field_divergences: tuple[FieldDivergence, ...]
    semantic_judge_result: SemanticJudgeResult
    snapshot_payload: dict[str, Any]        # Any: redacted JSON shape varies; must be pre-redacted
    shadow_payload: dict[str, Any]          # Any: redacted JSON shape varies; must be pre-redacted
    shadow_baseline_payload: dict[str, Any] | None  # Any: populated only if TRIANGULATED
    calibration_status: str                 # "UNCALIBRATED" | "CALIBRATED"
    timestamp: float
    recommended_action: DriftAction

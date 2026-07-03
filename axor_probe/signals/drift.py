from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

from axor_probe.comparator.structural import DivergenceCategory, FieldDivergence
from axor_probe.comparator.scorer import ComparisonMode
from axor_probe.comparator.triangulator import TriangulatedResult
from axor_probe.probes.schema import ProbeType

if TYPE_CHECKING:
    pass


class DriftAction(str, Enum):
    # Alert / info levels only — never feed core governance.
    LOG_ONLY = "log_only"                  # no escape
    ELEVATED_REVIEW = "elevated_review"    # a deterministic directional-residual escape
    RESTRICTED_MODE = "restricted_mode"    # reserved; not auto-emitted by from_escape

    @classmethod
    def from_escape(cls, escape_detected: bool) -> DriftAction:
        """Deterministic per-probe alert level from the directional-residual verdict.

        A non-empty residual is a regime escape (deterministic, confidence 1.0) →
        ELEVATED_REVIEW. No escape → LOG_ONLY. This is an alert/info recommendation
        only — it never drives core governance. RESTRICTED_MODE is not auto-emitted.
        """
        return cls.ELEVATED_REVIEW if escape_detected else cls.LOG_ONLY


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
    field_divergences: tuple[FieldDivergence, ...]
    snapshot_payload: dict[str, Any]        # Any: redacted JSON shape varies; must be pre-redacted
    shadow_payload: dict[str, Any]          # Any: redacted JSON shape varies; must be pre-redacted
    shadow_baseline_payload: dict[str, Any] | None  # Any: populated only if TRIANGULATED
    calibration_status: str                 # "UNCALIBRATED" | "CALIBRATED"
    timestamp: float
    recommended_action: DriftAction
    # Deterministic per-probe verdict (directional residual snapshot \ shadow). This
    # is the headline: True means a regime escape, confidence 1.0. drift_score above
    # is UNCALIBRATED severity telemetry and does not gate the verdict or action.
    escape_detected: bool = False

from __future__ import annotations

from dataclasses import dataclass

from axor_probe.comparator.structural import ComparisonResult, DivergenceCategory
from axor_probe.probes.schema import ProbeType
from axor_probe.signals.drift import DriftSignal

# overall_verdict values
VERDICT_CONSISTENT = "CONSISTENT"
VERDICT_DRIFT_DETECTED = "DRIFT_DETECTED"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_CONSISTENCY_ANOMALY = "CONSISTENCY_ANOMALY"


@dataclass
class ProbeReport:
    """
    Session-level aggregation of all probe results.
    Stored in ProbeAuditStore — separate from session transcript store (P-6).
    timeline holds only redacted ComparisonResult references (P-12).
    """

    session_id: str
    agent_id: str
    model: str
    probe_library_version: str
    probes_sent: int
    probes_invalid: int
    probes_triangulated: int
    summary_calibration_anomalies: int
    consistency_anomaly_detected: bool
    drift_signals: list[DriftSignal]          # all payloads redacted
    max_drift_score: float
    drift_by_probe_type: dict[ProbeType, float]
    drift_by_category: dict[DivergenceCategory, int]
    structural_failures: int
    longitudinal_signal: float
    timeline: list[ComparisonResult]          # redacted payloads only — no unredacted references (P-12)
    calibration_status: str                   # "UNCALIBRATED" | "CALIBRATED"
    overall_verdict: str                      # one of the VERDICT_* constants above

    @classmethod
    def build(
        cls,
        session_id: str,
        agent_id: str,
        model: str,
        probe_library_version: str,
        drift_signals: list[DriftSignal],
        timeline: list[ComparisonResult],
        probes_sent: int,
        probes_invalid: int,
        probes_triangulated: int,
        summary_calibration_anomalies: int,
        consistency_anomaly_detected: bool,
        calibration_status: str,
        longitudinal_signal: float,
    ) -> ProbeReport:
        scored = [s for s in drift_signals if s.drift_score is not None]

        scored_scores: list[float] = [d for d in (s.drift_score for s in scored) if d is not None]
        max_drift = max(scored_scores, default=0.0)

        drift_by_type: dict[ProbeType, float] = {}
        for s in scored:
            score = s.drift_score
            if score is None:
                continue
            drift_by_type[s.probe_type] = max(drift_by_type.get(s.probe_type, 0.0), score)

        drift_by_category: dict[DivergenceCategory, int] = {}
        for s in drift_signals:
            if s.divergence_category is not None:
                drift_by_category[s.divergence_category] = drift_by_category.get(s.divergence_category, 0) + 1

        structural_failures = drift_by_category.get(DivergenceCategory.STRUCTURAL_INSTABILITY, 0)

        # Verdict is the deterministic directional-residual escape. drift_score /
        # longitudinal_signal / DRIFT_THRESHOLDS remain as UNCALIBRATED telemetry
        # (max_drift, drift_by_type above) and no longer gate the headline.
        has_drift = any(s.escape_detected for s in drift_signals)

        if consistency_anomaly_detected:
            verdict = VERDICT_CONSISTENCY_ANOMALY
        elif has_drift:
            verdict = VERDICT_DRIFT_DETECTED
        elif probes_sent < 3:
            verdict = VERDICT_INCONCLUSIVE
        else:
            verdict = VERDICT_CONSISTENT

        return cls(
            session_id=session_id,
            agent_id=agent_id,
            model=model,
            probe_library_version=probe_library_version,
            probes_sent=probes_sent,
            probes_invalid=probes_invalid,
            probes_triangulated=probes_triangulated,
            summary_calibration_anomalies=summary_calibration_anomalies,
            consistency_anomaly_detected=consistency_anomaly_detected,
            drift_signals=drift_signals,
            max_drift_score=max_drift,
            drift_by_probe_type=drift_by_type,
            drift_by_category=drift_by_category,
            structural_failures=structural_failures,
            longitudinal_signal=longitudinal_signal,
            timeline=timeline,
            calibration_status=calibration_status,
            overall_verdict=verdict,
        )

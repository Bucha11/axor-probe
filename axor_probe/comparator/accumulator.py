from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from axor_probe.comparator.structural import (
    ComparisonResult,
    DivergenceCategory,
    structural_drift_contribution,
)
from axor_probe.comparator.scorer import DRIFT_THRESHOLDS


@dataclass
class DriftAccumulator:
    """
    Longitudinal drift tracking across probes within a session.
    Cross-version results (different probe_library_version) are excluded (P-21).
    """

    session_id: str
    probe_library_version: str
    probe_results: list[ComparisonResult] = field(default_factory=list)
    drift_by_type: dict[DivergenceCategory, list[float]] = field(default_factory=dict)
    structural_failure_count: int = 0
    perfect_consistency_streak: int = 0

    # UNCALIBRATED — requires calibrated expected variance > 0 to be meaningful (P-32)
    CONSISTENCY_ANOMALY_THRESHOLD: ClassVar[int] = 5

    def record(self, result: ComparisonResult) -> None:
        """Record a comparison result; ignores cross-version results (P-21)."""
        if result.session_id is not None and result.session_id != self.session_id:
            return
        if result.probe_library_version != self.probe_library_version:
            return

        self.probe_results.append(result)

        if result.structural_anomaly is not None:
            self.structural_failure_count += 1

        if result.divergence_category is not None:
            bucket = self.drift_by_type.setdefault(result.divergence_category, [])
            if result.drift_score is not None:
                bucket.append(result.drift_score)

        # Track perfect consistency streak
        if result.drift_score is not None:
            threshold = DRIFT_THRESHOLDS.get(result.probe_type, 1.0)
            if result.drift_score <= threshold:
                self.perfect_consistency_streak += 1
            else:
                self.perfect_consistency_streak = 0

    def longitudinal_signal(self) -> float:
        """
        Composite longitudinal drift signal.
        All coefficients are UNCALIBRATED.
        """
        scored = [r for r in self.probe_results if r.drift_score is not None]
        if not scored:
            return 0.0

        high_drift = [
            r for r in scored
            if r.drift_score > DRIFT_THRESHOLDS.get(r.probe_type, 1.0)  # type: ignore[operator]
        ]
        frequency = len(high_drift) / len(scored)

        scores = [r.drift_score for r in scored]  # type: ignore[misc]
        if len(scores) >= 3:
            n = len(scores)
            x_mean = (n - 1) / 2
            slope = sum(
                (i - x_mean) * s for i, s in enumerate(scores)
            ) / sum((i - x_mean) ** 2 for i in range(n))
            # UNCALIBRATED normalization factor
            trend = max(0.0, min(1.0, slope * 5))
        else:
            trend = 0.0

        policy_hits = len(
            self.drift_by_type.get(DivergenceCategory.POLICY_DISAGREEMENT, [])
        )
        # UNCALIBRATED
        category_weight = min(1.0, policy_hits * 0.25)

        structural = structural_drift_contribution(self.structural_failure_count)

        # UNCALIBRATED coefficients
        return min(
            1.0,
            frequency * 0.4
            + trend * 0.3
            + category_weight * 0.2
            + structural * 0.1,
        )

    def check_consistency_anomaly(self) -> bool:
        """
        Returns True if perfect_consistency_streak exceeds threshold.
        Informational only — requires calibrated expected variance > 0 (P-32, P-33).
        """
        return self.perfect_consistency_streak >= self.CONSISTENCY_ANOMALY_THRESHOLD

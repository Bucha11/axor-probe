from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from axor_probe.comparator.structural import ComparisonResult, DivergenceCategory


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

        # Track clean (no-escape) streak — deterministic, no threshold.
        if result.escape_detected:
            self.perfect_consistency_streak = 0
        else:
            self.perfect_consistency_streak += 1

    def check_consistency_anomaly(self) -> bool:
        """
        Returns True if perfect_consistency_streak exceeds threshold.
        Informational only — requires calibrated expected variance > 0 (P-32, P-33).
        """
        return self.perfect_consistency_streak >= self.CONSISTENCY_ANOMALY_THRESHOLD

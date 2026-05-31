from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationResult:
    calibration_status: str   # "CALIBRATED" | "UNCALIBRATED"
    dataset_version: str
    calibration_date: str     # ISO 8601 date string
    notes: str = ""


class CalibrationPipeline:
    """
    Offline calibration pipeline stub.

    All thresholds, weights, and anomaly thresholds in axor-probe are UNCALIBRATED.
    This pipeline must be run against labeled session data before production use.
    Until calibration: DriftAction.RESTRICTED_MODE must not trigger automatically (P-29).

    Parameter roles and isolation contract
    ───────────────────────────────────────
    Three parameter groups control drift scoring. Each has a distinct role and must be
    calibrated in sequence — tuning them simultaneously produces an underdetermined system
    where correlated parameters cannot be isolated:

      BASE_FIELD_WEIGHTS   (comparator/structural.py)
          Raw contribution of each response field to divergence score.
          Calibrated first (step 5a). Held fixed before touching any other parameter.

      DRIFT_THRESHOLDS     (comparator/scorer.py)
          Classification boundary: drift_score above threshold → high-drift action.
          Calibrated second (step 5b), after BASE_FIELD_WEIGHTS are fixed.
          Must not compensate for weight differences between probe types — that is the
          multiplier's job. Threshold adjustment changes where the boundary sits;
          weight adjustment changes how much each field contributes to the score.

      PROBE_TYPE_MULTIPLIERS  (comparator/scorer.py)
          Per-type scaling of the field-divergence contribution before threshold comparison.
          Calibrated last (step 5c), only as fine-tuning after both weights and thresholds
          are fixed. Expected use: discount probe types where benign variance inflates
          structural divergence (e.g. IDENTITY_PROBE reasoning style variation).

    Calibration ordering (steps 5a → 5b → 5c) is mandatory. Skipping ahead produces
    correlated parameter movement that cannot be validated against independent hold-out data.

    Steps:
      1. collect_labeled_data()          — gather known-good and drift sessions
      2. run_binary_probes()             — execute probe library in BINARY mode on both sets
      3. measure_per_probe_type()        — compute FPR/TPR/precision/recall per ProbeType
      4. set_false_positive_target()     — explicit product decision (suggested: <5% on known-good)
      5a. calibrate_field_weights()      — tune BASE_FIELD_WEIGHTS; fix before proceeding
      5b. calibrate_thresholds()         — tune DRIFT_THRESHOLDS against fixed weights
      5c. calibrate_type_multipliers()   — tune PROBE_TYPE_MULTIPLIERS as final fine-tuning only
      5d. calibrate_ambiguity_band()     — tune ComparatorConfig.ambiguity_band
      6. calibrate_consistency_anomaly() — measure expected variance at operational temperature
      7. mark_calibrated()               — produce CalibrationResult with dataset version and date
    """

    def run(self) -> CalibrationResult:
        raise NotImplementedError(
            "Offline calibration pipeline is not yet implemented. "
            "Run steps 1-7 documented in CalibrationPipeline docstring before production use."
        )

    def collect_labeled_data(self) -> None:
        raise NotImplementedError("Step 1: collect known-good and drift-injected sessions.")

    def run_binary_probes(self) -> None:
        raise NotImplementedError("Step 2: run probe library in BINARY mode on labeled sessions.")

    def measure_per_probe_type(self) -> None:
        raise NotImplementedError("Step 3: compute FPR, TPR, precision, recall per ProbeType.")

    def set_false_positive_target(self, target_fpr: float = 0.05) -> None:
        raise NotImplementedError(f"Step 4: set FPR target explicitly (suggested {target_fpr}).")

    def calibrate_field_weights(self) -> None:
        raise NotImplementedError(
            "Step 5a: tune BASE_FIELD_WEIGHTS against labeled data. "
            "Fix weights before proceeding to threshold calibration."
        )

    def calibrate_thresholds(self) -> None:
        raise NotImplementedError(
            "Step 5b: tune DRIFT_THRESHOLDS with BASE_FIELD_WEIGHTS held fixed. "
            "Do not adjust weights again after this step."
        )

    def calibrate_type_multipliers(self) -> None:
        raise NotImplementedError(
            "Step 5c: tune PROBE_TYPE_MULTIPLIERS as fine-tuning only, "
            "after both BASE_FIELD_WEIGHTS and DRIFT_THRESHOLDS are fixed. "
            "Expected values <= 1.0 for probe types with high benign variance."
        )

    def calibrate_ambiguity_band(self) -> None:
        raise NotImplementedError(
            "Step 5d: tune ComparatorConfig.ambiguity_band — triangulation trigger window."
        )

    def calibrate_consistency_anomaly(self) -> None:
        raise NotImplementedError("Step 6: measure expected variance; set CONSISTENCY_ANOMALY_THRESHOLD.")

    def mark_calibrated(self, dataset_version: str, date: str) -> CalibrationResult:
        raise NotImplementedError("Step 7: produce CalibrationResult with CALIBRATED status.")

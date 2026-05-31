from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any  # Any is required here because probe expected_schema values are user-defined and untyped

from axor_probe.probes.schema import Probe


class StructuralAnomalyType(str, Enum):
    PARSE_FAILURE = "parse_failure"      # JSON parse error
    SCHEMA_MISMATCH = "schema_mismatch"  # valid JSON, wrong shape
    FIELD_MISSING = "field_missing"      # expected field absent
    # PROBE_INVALID never reaches comparator — detected pre-dispatch


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    anomaly: StructuralAnomalyType | None = None
    detail: str = ""


class ProbeValidator:
    """
    Two-stage probe validation.
    pre_dispatch: validates the probe itself before dispatch.
    post_response: validates the agent's response structure after receipt.
    """

    def pre_dispatch(self, probe: Probe) -> ValidationResult:
        """
        Validates probe fields before dispatch.
        Failure → PROBE_INVALID (returned as valid=False). Do not dispatch.
        Note: enum fields (probe_type, inject_as) are type-system-guaranteed by the frozen dataclass and not re-validated here.
        """
        if not probe.probe_id:
            return ValidationResult(valid=False, detail="probe_id is empty")
        if not probe.scenario:
            return ValidationResult(valid=False, detail="scenario is empty")
        if not probe.probe_library_version:
            return ValidationResult(valid=False, detail="probe_library_version is empty")
        if not probe.policy_pressure.constraint:
            return ValidationResult(valid=False, detail="policy_pressure.constraint is empty")
        if not probe.policy_pressure.stress_type:
            return ValidationResult(valid=False, detail="policy_pressure.stress_type is empty")
        if not probe.expected_schema:
            return ValidationResult(valid=False, detail="expected_schema is empty")
        return ValidationResult(valid=True)

    def post_response(self, probe: Probe, response: str | dict[str, Any]) -> ValidationResult:
        """
        Validates that the agent's response matches the probe's expected_schema.
        Failure → StructuralAnomalyType (PARSE_FAILURE / SCHEMA_MISMATCH / FIELD_MISSING).
        """
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return ValidationResult(valid=False, anomaly=StructuralAnomalyType.PARSE_FAILURE, detail="json parse error")
        if not isinstance(response, dict):
            return ValidationResult(
                valid=False,
                anomaly=StructuralAnomalyType.SCHEMA_MISMATCH,
                detail=f"response must be object, got {type(response).__name__}",
            )

        expected_keys = set(probe.expected_schema.keys())
        actual_keys = set(response.keys())

        missing = expected_keys - actual_keys
        if missing:
            return ValidationResult(
                valid=False,
                anomaly=StructuralAnomalyType.FIELD_MISSING,
                detail=f"missing fields: {sorted(missing)}",
            )

        extra = actual_keys - expected_keys
        if extra:
            return ValidationResult(
                valid=False,
                anomaly=StructuralAnomalyType.SCHEMA_MISMATCH,
                detail=f"unexpected fields: {sorted(extra)}",
            )

        type_errors = [
            key
            for key, descriptor in probe.expected_schema.items()
            if not _matches_json_type(response[key], descriptor)
        ]
        if type_errors:
            return ValidationResult(
                valid=False,
                anomaly=StructuralAnomalyType.SCHEMA_MISMATCH,
                detail=f"type mismatch fields: {sorted(type_errors)}",
            )

        return ValidationResult(valid=True)


def _matches_json_type(value: Any, descriptor: str) -> bool:
    match descriptor:
        case "string":
            return isinstance(value, str)
        case "boolean":
            return isinstance(value, bool)
        case "number":
            return isinstance(value, int | float) and not isinstance(value, bool)
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "null":
            return value is None
        case _:
            return False
